# HANDOFF: Fix chunk collisions and train/val leakage in multi-species chunking

**Base branch:** `feat/portal-download` (`b4f6793`), which contains all of
`feat/fourier-augmentations` (`1645d09`).
**Work branch:** `fix/chunk-collisions`, merged back into `feat/portal-download` when done.
**Why now:** the first training run with the Fourier augmentation (CZII 10440, 7 tomograms,
several species) most likely trained on mixed-up chunks. The augmentation code is not at
fault. The chunking step is. Until this is fixed, a run cannot tell you whether the Fourier
augmentation helps.

Work through the gates in order. Do not start a gate until the previous one passes.

---

## Context

### How the data flows (current code)
1. `tomocpt prepare_data` writes one dataset directory per entry in `prepData`
   (`labels/run.py`): `training_data_dir/dataset_<idx>_class_<id>/`, with label MRCs in
   `class_<id>/labels/<tomo>.labels.mrc` and a tracking CSV with the columns
   `tomogram_path, label_path, particle_diameter_angst, num_particles`
   (`labels/helpers.create_tracking_dataframe`).
2. `tomocpt train` reads **all** tracking CSVs into one DataFrame
   (`utils.read_particles_csvs`, glob `training_data_dir/*/*.csv`), so there is one row per
   (dataset, tomogram).
3. `dataManager/chunking.do_chunking` splits those rows into train and validation and runs
   `process_single_tomo_task` once per row, in parallel through Dask.

For CZII with N species from the same 7 tomograms, there are N × 7 rows, and every species of
one tomogram has the same `tomogram_path`.

### Defect 1: chunk collisions (`chunking.py`, `process_single_tomo_task`)
```python
bn = Path(tomo_path_str).stem                 # same for every species of a tomogram
tomo_out_dir = Path(outpath) / bn             # -> one directory for all species
...
vol_path   = vol_dir    / f"volumes_{i}_{z}_{y}_{x}.nii.gz"
label_path = labels_dir / f"labels_supervised_{i}_{z}_{y}_{x}.nii.gz"
```
- Each species rescales the tomogram to its own size (`desired_particle_pixel_size` voxels
  per particle diameter), so each species produces a different chunk grid.
- Chunk index `i` and grid position `(z, y, x)` repeat across species, so the file names
  repeat. Later writes overwrite earlier ones.
- The Dask CUDA worker has `threads_per_worker = n_cpus // n_gpus` threads (8 with one GPU),
  so several rows of the same tomogram run **at the same time**. A volume file and its
  label file can be written by different tasks: **the input comes from one species and
  scale, the target from another.**
- The same defect occurs whenever two datasets contain tomograms with the same file stem,
  even for one species.

### Defect 2: leakage between train and validation (`chunking.py`, `do_chunking`)
```python
df_train, df_val = train_test_split(tomosDf, test_size=0.2, random_state=42)
```
This splits (dataset, tomogram) rows, not tomograms. The same tomogram, at another scale, ends up in
both train and validation. `val_loss`, early stopping and the checkpoint choice all become
optimistic.

### Not a code defect, but expect it (out of scope here)
In pipeline v1 every species is rescaled to the same size, and the other species in the same
tomogram are negatives. Species of almost the same size therefore get contradictory labels
(CZII: apo-ferritin vs β-amylase; thyroglobulin vs virus-like particle vs ribosome). This
fix makes the data consistent, but it does not remove that conflict. Pipeline v2 with size
conditioning (`plan/scale-free`, Gates 2–3) addresses it.

### Hard constraints
- **Legacy behaviour must not change** when every row has a unique tomogram stem (the normal
  single-species case): the same directory names, the same file names, and the same train/val
  split as today (`train_test_split(..., random_state=42)` on the rows).
- Do not change chunk file names inside a directory, the `volumes` / `labels_*` sub-folder
  names, or `VolumeDatsetIO._get_filepaths`. The loader pairs files by name inside one
  directory and must keep working unchanged.
- No new dependencies. `scikit-learn` (already pinned) has `GroupShuffleSplit`.

---

## Gate 0: Diagnose the existing run (no code change)

On the machine that has the chunks:
```bash
ls <chunks_dir>/train | sort > /tmp/train.txt; wc -l < /tmp/train.txt
ls <chunks_dir>/val   | sort > /tmp/val.txt;   wc -l < /tmp/val.txt
comm -12 /tmp/train.txt /tmp/val.txt         # tomograms in both splits
```
Expected if affected: at most 7 directories in `train`, and names that appear in both lists.
Also count the files: in an affected directory, `volumes/` and `labels_supervised/` have
about the same number of files, but far fewer than (species × chunks per species).

**Gate:** write the three numbers (train dirs, val dirs, shared names) into the PR description.
If `train` has more than 7 directories and no names are shared, this run is not affected;
still complete the fix, because the next multi-species run would be.

---

## Gate A: Unique chunk directory per (dataset, tomogram)

In `tomocpt/dataManager/chunking.py`:

1. Add a pure function:
   ```python
   def assign_chunk_dir_names(tomos_df: pd.DataFrame) -> pd.Series:
       """Directory name for each row of the tracking table.

       The tomogram stem when every stem is unique (legacy layout). Otherwise
       "<dataset_dir>__<stem>" for every row, where <dataset_dir> is the name of the
       prepare_data dataset directory (Path(label_path).parents[2].name, e.g.
       "dataset_3_class_all"). Raises ValueError if names are still not unique.
       """
   ```
   - Stem: `Path(tomogram_path).stem`.
   - For self-supervised rows, `label_path` still exists in the tracking CSV, so the same
     rule works. If `label_path` is missing, fall back to the index of the source CSV
     (see step 3).
   - Use the prefix for **all** rows once any stem repeats. A mixed layout makes it hard to
     see which directory belongs to which dataset.
   - Log one `INFO` line that says how many rows were renamed and why.
2. In `do_chunking`, compute `chunk_dir_name` for each row **before** the split, and put it
   in every task dict: `"chunk_dir_name": row_name`.
3. In `process_single_tomo_task`, replace `bn = Path(tomo_path_str).stem` with
   `bn = task.get("chunk_dir_name") or Path(tomo_path_str).stem`.
4. `read_particles_csvs` concatenates without `ignore_index`, so the DataFrame index repeats
   (0, 1, 2 … per CSV). Call `reset_index(drop=True)` in `do_chunking` before the split, so
   that row labels are unique.

**Gate:** unit tests in Gate D, items 1–3, pass.

---

## Gate B: Split by tomogram

In `do_chunking`, for `train_val_level == tomos`:

1. Add a pure function:
   ```python
   def split_rows_by_tomogram(tomos_df, test_size=0.2, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
       """Train/validation split that keeps all rows of one tomogram together.

       Legacy path: if every tomogram stem is unique, return exactly
       train_test_split(tomos_df, test_size=test_size, random_state=seed).
       Otherwise use GroupShuffleSplit(n_splits=1, test_size=test_size,
       random_state=seed) with groups = tomogram stem.
       Raises ValueError with a clear message if there are fewer than 2 distinct
       tomograms (the legacy code fails later with an empty train set).
       """
   ```
   The group key is the tomogram **stem**, so the same tomogram from two datasets counts as
   one group. If two different tomograms share a stem across datasets, they are grouped
   together; that is safe (it can only make validation stricter). Log a warning in that case.
2. Log the tomograms in each split (names, and the number of rows per split).
3. `train_val_level == chunks` (`CrossValidationLevelSplit.cubes`): keep the behaviour; it
   now moves (dataset, tomogram) directories, which is what that mode means. Add one log line
   that this mode puts the same tomogram in both splits by design.

**Gate:** unit tests in Gate D, items 4–6, pass.

---

## Gate C: Refuse stale chunks

Training skips chunking when `chunks_dir/done_labels_True.txt` exists
(`training/helpers._prepare_data_if_needed`), so an old, mixed-up chunk directory would be
reused without warning.

1. When `do_chunking` finishes, write `chunks_dir/chunk_layout.json` with
   `{"layout": 2, "dir_names": "unique" | "prefixed", "split": "rows" | "grouped"}`.
2. In `do_chunking`'s caller (`_prepare_data_if_needed`), if the done file exists **and**
   `chunk_layout.json` does not exist **and** the tracking table has repeated tomogram stems,
   raise `RuntimeError` with this message:
   *"chunks_dir was built by a version with the chunk-collision bug (several datasets share
   tomograms). Delete <chunks_dir>/done_labels_True.txt to rebuild the chunks."*
   If stems are unique, keep the legacy behaviour (reuse the chunks).

This touches `tomocpt/training/helpers.py` (the `_prepare_data_if_needed` function only).

**Gate:** unit test in Gate D, item 7, passes.

---

## Gate D: Tests (`tests/test_chunk_layout.py`, CPU only)

Build tracking DataFrames by hand (no files needed for items 1–6).

1. Unique stems → names equal the stems (legacy layout).
2. Two datasets, same 3 tomograms → 6 unique names of the form
   `dataset_<k>_class_<id>__<stem>`.
3. A collision that the prefix cannot resolve (two rows with the same dataset and stem) →
   `ValueError`.
4. Unique stems → `split_rows_by_tomogram` returns exactly what `train_test_split` returns
   (compare the index sets).
5. 6 datasets × 7 tomograms → no stem appears in both splits, every row is in exactly one
   split, and the validation split has at least one tomogram.
6. One distinct tomogram → `ValueError` that names the problem.
7. Stale-chunk guard: a temporary `chunks_dir` with `done_labels_True.txt` and no
   `chunk_layout.json`, plus repeated stems → `RuntimeError`; with unique stems → no error.
8. **Integration (CPU, marked `slow` if it takes > 30 s):** write one synthetic 96³ tomogram
   (`mrcfile`, voxel size 10 Å) with two kinds of spheres, and two label MRCs for two
   "species" (diameters 100 Å and 250 Å). Call `process_single_tomo_task` directly (no Dask)
   for both rows with `chunk_dir_name` from `assign_chunk_dir_names`. Check that:
   - two directories exist;
   - in each directory, every `volumes_*` file has a `labels_supervised_*` partner with the
     same suffix, and the counts match;
   - the chunk grids differ (the 250 Å species is rescaled less, so its directory has fewer
     chunks).
   Use `torch.device("cpu")`; `process_single_tomo_task` already falls back to CPU when CUDA
   is absent.

Run the existing suite too: `pytest tests -v` must stay green (the Fourier tests are not
affected by this change).

**Gate:** all of the above pass on the dev machine, with the pinned dependencies.

---

## Gate E: Rebuild the chunks and check them

```bash
rm <chunks_dir>/done_labels_True.txt          # do_chunking deletes train/ and val/ itself
tomocpt train --config-file config.yaml ...   # or a short run that only chunks
```
Then check:
1. `ls <chunks_dir>/train | wc -l` + `ls <chunks_dir>/val | wc -l` = species × tomograms.
2. No tomogram stem appears in both `train` and `val`.
3. Open 3 random chunk pairs from different species (napari or `dump_augmented_chunks.py`
   without augmentation): the label spheres sit on the particles in the volume.
4. `chunk_layout.json` exists.

**Gate:** the four checks pass; record the counts in the PR.

---

## Gate F: Restart training and compare

1. Restart the same configuration (Fourier augmentation on), from scratch.
2. Also run the same configuration with `use_mw_aug: false` and `use_fourier_aug: false`
   for the same number of epochs and the same seed. This is the only way to see what the
   augmentation adds.
3. Compare per species on held-out tomograms (not on `val_loss` alone): picks against the
   portal coordinates, using precision, recall and F-beta (β = 4). `predict` on the
   validation tomograms, then match within half the particle radius.

**Gate:** both runs train stably; the comparison table is in the PR or in a short report.

---

## Merge

1. `fix/chunk-collisions` → `feat/portal-download` (PR, with Gate 0 and Gate E numbers).
2. Then continue with `HANDOFF_scale_free.md` (on `plan/scale-free`), section 6, step 2:
   run the full test suite and merge `feat/portal-download` into `main`.

## Files touched

| File | Change |
|---|---|
| `tomocpt/dataManager/chunking.py` | `assign_chunk_dir_names`, `split_rows_by_tomogram`, `chunk_dir_name` in tasks, `reset_index`, layout file |
| `tomocpt/training/helpers.py` | stale-chunk guard in `_prepare_data_if_needed` |
| `tests/test_chunk_layout.py` | new |
| `README.md` | changelog entry: multi-species chunking fixed; rebuild old chunk directories |
