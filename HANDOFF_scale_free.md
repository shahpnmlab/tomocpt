# HANDOFF — TomoCPT scale-free upgrade (state on 25 Oct 2026)

Read this first in any new session. Then read `CLAUDE.md` and `.claude/agents/00_ORCHESTRATOR.md`
(both on branch `plan/scale-free`).

## 1. Where things are

| Branch | Head | Content | Status |
|---|---|---|---|
| `main` | `0402392` | Legacy TomoCPT 0.6.0 | unchanged |
| `feat/fourier-augmentations` | `1645d09` | membrain-seg Fourier + intensity augmentation port, config back-fill, contrast inversion, **plus two review fixes** (below) | tests not yet run by a human after the fixes |
| `feat/portal-download` | `b4f6793` | `feat/fourier-augmentations` + `scripts/download_portal_data.py` + tests | **download confirmed working on real data** (CZII 10440, 7 runs) |
| `plan/scale-free` | this commit | `CLAUDE.md`, `.claude/agents/*` (orchestrator, test author, Gates 1–7), `.claude/state/gate_status.json`, this file | plan only, no code |

`feat/portal-download` contains all of `feat/fourier-augmentations`. Merging it into `main`
brings both.

## 2. What was decided and why (short)

- **Goal stage 1 (Gates 1–5):** remove the fixed-particle rescale (every particle → 20 voxels).
  Pipeline v2 keeps the particle inside a band of 10–32 voxels with the least resampling
  and tells the network the size (FiLM conditioning). One backbone pass serves several sizes.
- **Goal stage 2 (Gates 6–7):** one model, one `predict` run: particle centroids plus maps for
  membrane, filament, organelle and void, trained from partial labels (Gate 6). Organelles and
  void run at a coarser voxel size (default 40 Å/px) through the same network, because a 64³
  chunk at 10 Å/px is only 64 nm wide.
- **MoE (Gate 7)** is an experiment on top of Gate 6: blob, line, sheet and smooth experts,
  router conditioned on size and voxel size, optional label-guided routing. Ships only if it
  beats the dense model and a "same convolutions, no router" control.
- **Segmentation does not emerge from router maps without labels.** That part of the original
  MoE plan was dropped. Labels come from portal masks, MemBrain-seg pseudo-labels or the user.
- **SSL v2 (Gate 5):** the 4-way rotation task is removed. Rotating by 90° about Z moves the
  missing wedge, so the head can only learn wedge orientation (accuracy ≈ 50 %, loss ≈ 0.69).
  It is replaced by masked reconstruction with a clean target, contrastive learning between
  differently degraded views, and a relative-scale task. The Fourier augmentation degrades the
  **input only**; polarity inversion goes on input and target together.

## 3. Review fixes already on `feat/fourier-augmentations` (`1645d09`)

1. **Wedge orientation.** membrain-seg uses `(x, y, z)` arrays; tomocpt uses `(z, y, x)`. The
   ported mask removed the frequencies a real tomogram measures (kept 31 % of them; fixed: 71 %).
   New `oriented_wedge_mask(..., array_order="zyx")`; `wedge_mask` itself is unchanged.
2. **Output scale.** The transform returned std 1.0; tomocpt chunks have std ≈ 0.30, so degraded
   chunks were 3.3× larger than clean ones. Now `match_input_stats=True` keeps mean and std.

Tests: `tests/test_fourier_orientation_and_scale.py` (11 checks; the numpy logic was verified
in a sandbox without torch; full pytest not yet run).

## 4. The training run started on 25 Oct 2026 — check this

Started from `feat/portal-download` with `use_mw_aug` and `use_fourier_aug` on. The wedge and
scale fixes are in that branch, so the augmentation itself is correct.

**Likely defect: chunk collisions when one tomogram is used for several species.**
If `prepData` lists several particle types that come from the same 7 tomograms (one dataset per
species, all with the same `raw_data_dir`), then:
- `dataManager/chunking.process_single_tomo_task` writes chunks to
  `chunks_dir/<split>/<tomogram stem>/` with names `volumes_<i>_<z>_<y>_<x>.nii.gz`. Every
  species uses the same tomogram stem, so all species write into **one directory with the same
  file names**, each at its own rescale. Later writes overwrite earlier ones. The Dask worker
  runs several tasks at once (`threads_per_worker = n_cpus // n_gpus`), so a volume chunk and its
  label chunk can come from **different species and scales**.
- The tomogram-level split (`train_test_split` over the rows of all tracking CSVs) splits
  (tomogram, species) rows, so the same tomogram can be in train and in validation.

Check (on the machine with the chunks):
```bash
ls <chunks_dir>/train | wc -l     # 6 species x ~5-6 tomograms expected; <= 7 means merged
ls <chunks_dir>/val
```
If the counts are ≤ 7 and the same tomogram names appear in `train` and `val`, the run is
affected. Fix (not written yet, first task below): name chunk directories
`<dataset dir>__<tomogram stem>` when stems repeat, and split by tomogram stem, keeping the
legacy behaviour when stems are unique.

**Science caveat for multi-species v1 training.** In v1 every species is rescaled to 20 voxels,
and the other species in the same tomogram are negatives. Species of almost the same size then
get contradictory labels. With the CZII challenge radii (apo-ferritin 60 Å, β-amylase 65 Å,
thyroglobulin 130 Å, virus-like particle 135 Å, ribosome 150 Å; check against the portal), the
pairs apo-ferritin/β-amylase and thyroglobulin/VLP/ribosome are within about 1–3 voxels after
rescaling. Expect weaker precision for those. Pipeline v2 with size conditioning and the PU mode
(Gates 2–3) is the real fix. Note: β-amylase was, I believe, weighted 0 in the challenge score.

## 5. Decisions the user still has to make

1. `config.yaml`: `infer.predictions_coord_format: relion` is invalid → `relion_31` or `relion_50`.
2. Contrast inversion probability: 0.8 (current, always on) or 0.5 (suggested).
3. v1 fine-tuning: should the `out` layer train? (It is in no optimizer group today.)
4. Gate 6 classes: membrane, filament, organelle, void — split microtubules and actin?
5. Gate 6 label sources: portal masks, MemBrain-seg pseudo-labels, own masks.

## 6. Next steps, in order

1. Fix the chunk collision and grouped split (branch from `feat/portal-download`, e.g.
   `fix/chunk-collisions`), with a test that runs two species on one tomogram. Re-chunk
   (delete `chunks_dir/done_labels_True.txt`) and restart training if the check in §4 fails.
2. Run `pip install -e '.[dev,portal]' && pytest tests -v` on `feat/portal-download`
   (expect 45 passed + 26 skipped without the membrain-seg reference, plus the new tests).
   Run once with the pinned numpy 1.26.4 / scipy 1.14.1.
3. Visual check: `python scripts/dump_augmented_chunks.py --chunks-dir <chunks>/train --report --png -n 20`.
   The wedge smearing should run along Z in every sample.
4. Merge: `feat/portal-download` → `main`, then `plan/scale-free` → `main`.
5. Tag `gate-0-base` on that `main` and start Gate 1 with the orchestrator
   (`.claude/agents/00_ORCHESTRATOR.md`).

## 7. Environment notes

- The cloud workspace used for this work cannot install PyTorch (PyPI and the PyTorch index are
  blocked there), so no torch tests have been run by Claude.
- The chat was linked to `macbookpro-home`, but that is **not** the development machine. Run
  tests, chunking and training on the dev/GPU machine (open the session there, or use Claude Code
  in the repo on that machine).
- Data downloaded with:
  `python scripts/download_portal_data.py --dataset-id 10440 --out data/czii --voxel-spacing 10.012 --processing filtered --max-runs 7`
