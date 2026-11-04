# 04c_AGENT_CV_ROUTINE.md — Gate 4c: built-in k-fold CV, ensemble and CV report (TODO 3a)

Added 4 Nov 2026 by human decision (TODO_v2.md item 3a, first after Gate 7's CZII CV): k-fold
training, the out-of-fold (OOF) ensemble and the CV summary move from `experiments/run_cv.sh`
into the package, opt-in. Pipeline v2 picking only. Extends Gate 4b (`04b_AGENT_CV_ENSEMBLE.md`).

## Why
- Gate 7 ran 4 CZII CV arms through `experiments/run_cv.sh` plus hand-written scripts. Every arm
  re-chunked the same 7 tomograms once per fold (5 x 8.1 GB, ~40 GB per arm; the Gate 7 arms
  worked around this with symlinks by hand).
- The validation tomograms are predicted twice per fold: once by the post-fit calibration
  (`training/run._calibrate_best_checkpoint`) and again by `build_ensemble` (~5 min per tomogram).
  Neither saves the picks, so every diagnostic (per-fold F4, false-positive breakdown, galleries)
  needed a third prediction pass (`benchmarks/out/czii_g7_*/oof_analysis/`).
- The peak-shape question (TODO 3a.3: is the narrow core of the target learned?) and the
  localisation of picks inside the 0.5 R match radius are invisible in F4 alone.
- Loading several checkpoints with different `train` settings in one process (no CLI snapshot)
  fails: the first load's settings leak into the second through the global `mainConfig`
  (CLAUDE.md 1.10; found 4 Nov with a contrastive and a plain checkpoint: missing
  `contrastive_head.*`). Ensembles load many checkpoints per process.

## Design

### D1. CV mode in `tomocpt train` (opt-in)
`train.cv_all_folds = True` together with `train.cv_folds = K >= 2` makes `tomocpt train` run the
whole CV. `cv_all_folds = False` (default) leaves `tomocpt train` unchanged, including the Gate 4b
single-fold mode (`cv_folds = K`, `cv_fold = k`). Invalid combinations are listed in D6.

Steps, all in the parent `tomocpt train` process except step 2:
1. **Shared chunks (D2).** Chunk every tomogram once into the pool, then build one fold
   `chunks_dir` per fold from it.
2. **Folds.** For `k = 0 .. K-1`, run one **separate process** `python -m tomocpt.main train`
   with the parent's configuration plus `cv_all_folds=False`, `cv_fold=k`, the fold's
   `chunks_dir`, `model_dir` and `experiment_name`. A separate process isolates the global
   `mainConfig` between folds and lets Lightning start DDP as for any `tomocpt train`. The
   fold process is an ordinary Gate 4b single-fold run, including the post-fit calibration
   (which writes the validation picks, D3). stdout/stderr go to the fold's `train.log`.
   A non-zero exit stops the CV with `RuntimeError` naming the fold and its log.
3. **Ensemble.** `build_ensemble` over the fold checkpoints (D3, D4), written to
   `<cv dir>/ensemble.json`, with the infer section of the parent config.
4. **Report.** `<cv dir>/cv_summary.md` and `cv_summary.json` (D5), plus `peak_profiles.png`.

Output layout (`<cv dir>` = `<model_dir>/<experiment_name>`):
```
<cv dir>/cv_config.json                       folds K, seed, fold -> validation stems, fold status
<cv dir>/folds/fold{k}/train.log
<cv dir>/folds/fold{k}/checkpoints/weights.ckpt      (Lightning, model_dir=<cv dir>/folds, experiment_name=fold{k})
<cv dir>/folds/fold{k}/checkpoints/weights.val_picks.npz
<cv dir>/folds/fold{k}/done                   written after the fold process exits 0 with a checkpoint
<cv dir>/ensemble.json, oof_picks.npz, cv_summary.md, cv_summary.json, peak_profiles.png
<chunks_dir>/pool/...                         the shared chunks (D2)
<chunks_dir>/fold{k}/...                      one fold chunks_dir per fold (links into the pool)
```
**Resume:** running the same command again skips folds with a `done` marker and an existing
checkpoint; an unfinished fold restarts from scratch (its folder is cleared first). The ensemble
and the report are always rebuilt (cheap when the validation picks exist, D3).
`cv_config.json` stores K and the seed; a rerun with a different K or seed raises `ValueError`
(use another `experiment_name`).

### D2. Shared chunks
Pipeline v2 chunks of one tomogram do not depend on the split: chunk directory names come from
`assign_chunk_dir_names` over all rows and every random draw is seeded by
`(seed, chunk dir name, level, ...)`. So:
- `chunking.chunk_pool_v2(tomos_df, pool_dir, ...)` chunks every tomogram once into
  `<pool_dir>/train/<dir>/` with the same task fields as `_do_chunking_v2` (no split, no
  `val_tomograms`), and writes the pool manifest (`chunk_layout.json` with `"kind": "pool"`
  plus every v2 key except `val_tomograms` / `cv`) and its done file last.
- `chunking.link_fold_chunks(pool_dir, fold_dir, tomos_df, folds, fold, ...)` builds a fold
  `chunks_dir` whose `train/` and `val/` hold **symlinks** to the pool's per-tomogram chunk
  directories, split by `split_rows_by_fold`, and writes the same `chunk_layout.json` (including
  `"cv"` and `val_tomograms`, with `val_coords/` CSVs where a tomogram has several coordinate
  files) and done file that `_do_chunking_v2(..., cv=(K, k))` would write.
- `_prepare_data_if_needed` is unchanged for fold processes: they find a complete fold
  `chunks_dir` with the matching `cv` entry and skip chunking.
- The pool is rebuilt only if its done file is missing; a pool whose class table, band,
  resample, chunk size / stride or seed differs from the request raises `ValueError` (as the
  existing class-table check).

### D3. Validation picks are saved (TODO 3a.2)
`calibrate_checkpoint` additionally writes `<ckpt stem>.val_picks.npz` next to the checkpoint
(temporary file, then rename) with every pick at the lowest grid threshold, the truths, the
radii, the per-tomogram class availability, the peak profiles (D5) and a fingerprint of the
infer settings that change picks (`peak_spacing_fraction`, `distance_threshold`,
`cross_class_nms`, `use_tta`, `length`, `accumulate_dtype`, `PATCH_OVERLAP_FACTOR`, and the
forced values of D4). Format (no pickle):
- `tomograms`: str `[T]` (tomogram paths); `class_numbers`: int64 `[C]`; `radii`: float64 `[C]`
  (native voxels);
- `picks`: float64 `[P, 6]` rows `tomogram index, class number, x, y, z, score`;
- `truths`: float64 `[M, 5]` rows `tomogram index, class number, x, y, z`;
- `available`: bool `[T, C]`; `fingerprint`: str (JSON);
- profile arrays of D5.
`tomocpt.predict.calibration.read_val_picks(path) -> dict` returns
`(picks, truths, radii, grid)` in the format of `calibrate_thresholds` plus the extras.
`build_ensemble` reuses a fold's file when it exists, its tomograms equal the checkpoint's
`val_tomograms` and its fingerprint equals the current one; otherwise it predicts as before.
It writes the pooled picks of all folds to `oof_picks.npz` (same format plus a `fold` column
first in `picks` / `truths`) next to the ensemble file and records `"oof_picks": "oof_picks.npz"`
(path relative to the ensemble file) in the record. The thresholds are identical with and without
reuse.

### D4. Checkpoint loads isolated from each other (TODO 3a.4)
`tomocpt.predict.ensemble.load_member_isolated(ckpt, device)` loads a pipeline-v2 checkpoint
with the checkpoint's own settings: it sets the CLI snapshot to a copy of the default config for
the duration of the load (so no `train` value from the live `mainConfig` or from the command
line is applied to the checkpoint) and restores the previous snapshot and `mainConfig.train`
afterwards. `build_ensemble`, `load_ensemble` and `calibrate_checkpoint` use it. A single
checkpoint loaded by `tomocpt predict` keeps the legacy path (invariant 1).

### D5. CV report (TODO 3a.1, 3a.3)
`tomocpt.predict.cv_report.write_cv_report(ensemble_path, out_dir, class_weights=None)` reads the
ensemble record and `oof_picks.npz` and writes:
- per class: true particles, threshold, OOF F-beta, precision, recall, TP / FP / FN at the
  calibrated threshold; per fold F-beta;
- **localisation:** per class, mean and median distance (native voxels and in units of the class
  radius R) from matched picks to their true centres, and the mean offset per axis (x, y, z).
  Until TODO 3b.1 the truths are the coordinate CSVs (whole voxels; the report says so);
- **peak profiles** (from D3): per class, the mean predicted probability around true centres as
  a radial profile (bins of 0.05 R, 0 to 1.5 R, by distance from the true centre to the voxel
  centre) and as line profiles through the true centre along x, y and z (trilinear, steps of
  0.05 R, -1.5 R to 1.5 R), with the target profile (`labels.helpers` two-Gaussian profile,
  cut at R) for comparison; peak height = mean over true particles of the map maximum within
  0.5 R of the centre; FWHM of the radial profile (2 x the radius where it falls to half its
  value at r = 0, linear interpolation, in units of R; target 0.53 R). Profile accumulators are
  sums and counts per bin, so folds and tomograms pool exactly;
- the weighted mean F-beta when `class_weights` is given (`train.cv_class_weights`,
  `"name:weight,..."`; classes not named weigh 1), else the unweighted mean;
- `peak_profiles.png`: one panel per particle class (radial and x / y / z profiles, target).
Filament classes are listed without F-beta (they are not calibrated) and without profiles.

### D6. Errors
`ValueError` (before any work) for: `cv_all_folds` with `cv_folds < 2`; with
`data_pipeline_version != 2`; with `mode != picking`; with `resume_from` or `fine_tune_from`
(not supported in CV mode; fine-tuning every fold from one checkpoint is future work);
`cv_folds` larger than the number of distinct tomograms (from `split_rows_by_fold`).

### D7. `experiments/run_cv.sh`
Becomes a thin wrapper: writes the base config with `cv_folds`, `cv_all_folds=True`, and the
class head / cross-class settings it already takes, calls `tomocpt train`, appends
`cv_summary.md` to `experiments/czii_v2_results.md` and optionally predicts `PREDICT_DIR`.
CZII Kaggle weights are passed as `train.cv_class_weights`.

## Config (`defaultConfigs/train_config.py`)
| Path | Type / default |
|---|---|
| `train.cv_all_folds` | bool = `False` (run every fold of `cv_folds`, then ensemble and report) |
| `train.cv_class_weights` | Optional[str] = `None` (`"beta-amylase:0,thyroglobulin:2"`; report only) |

## Allowed write paths
- `tomocpt/training/cv.py` (new: the driver, `run_cv(config)`), `tomocpt/training/run.py`
  (dispatch to `run_cv` when `cv_all_folds`; nothing else)
- `tomocpt/dataManager/chunking.py` (`chunk_pool_v2`, `link_fold_chunks`, and refactoring
  `_do_chunking_v2` into shared helpers without changing its output)
- `tomocpt/defaultConfigs/train_config.py` (the two fields)
- `tomocpt/predict/{calibration,ensemble,cv_report}.py`
- `experiments/run_cv.sh`, `README.md` (section "k-fold cross-validation")
- `.claude/reports/gate_04c_report.json`

## Public names
`tomocpt.training.cv.{run_cv, prepare_cv_chunks}` (plus the replaceable `_run_fold_process`,
`_build_ensemble_and_report`);
`tomocpt.dataManager.chunking.{chunk_pool_v2, link_fold_chunks}`;
`tomocpt.predict.calibration.{read_val_picks, val_picks_path, peak_profile_accumulate}` and the
keyword `profiles=` of `collect_validation_picks`;
`tomocpt.predict.ensemble.{load_member_isolated, read_oof_picks}`;
`tomocpt.predict.cv_report.{write_cv_report, peak_profile_stats}`.
Exact signatures and file formats: docstring of `tests/gate4c/g4c_helpers.py` (TA). `oof_picks.npz`
also holds `tomogram_fold` `[T]` and `class_names` `[C]`.

## Pass criteria
- **G4c-C1** Defaults: every earlier test passes unchanged; `tomocpt train` without
  `cv_all_folds` never imports `tomocpt.training.cv`.
- **G4c-C2** Shared chunks equal per-fold chunks: on synthetic multi-class data (K = 3), for every
  fold, `link_fold_chunks` gives the same `chunk_layout.json` (all keys) and byte-identical chunk
  files (`volumes/`, `points/`, `chunk_meta.json`) at the same relative paths as
  `_do_chunking_v2(..., cv=(3, k))`; fold `train/` and `val/` entries are symlinks into the pool;
  `_prepare_data_if_needed` on a fold dir skips chunking; a pool built with another class table
  or chunk stride raises `ValueError`.
- **G4c-C3** `val_picks.npz` round trip: `read_val_picks` returns exactly what
  `collect_validation_picks` produced (picks, scores, truths, radii, availability);
  `build_ensemble` with and without the files gives identical records (thresholds, F-beta,
  `oof_tomograms`); a changed fingerprint (e.g. `peak_spacing_fraction`) or a different
  tomogram list makes it predict again (counted with a spy); `oof_picks.npz` holds every fold's
  picks with the right `fold` column.
- **G4c-C4** Isolation: two tiny pipeline-v2 checkpoints that differ in `train` settings that
  change the module set (contrastive head on / off) load in either order in one process through
  `load_member_isolated`, `load_ensemble` and `build_ensemble` without error and with each
  checkpoint's own settings; the live `mainConfig.train` and the CLI snapshot are the same before
  and after.
- **G4c-C5** Report: on hand-made picks and truths, per-class F-beta, precision, recall, TP / FP /
  FN, weighted mean (with and without weights, unknown class name → `ValueError`), localisation
  (mean / median distance, per-axis offset) equal hand-computed values; `peak_profile_stats` on a
  synthetic map that is exactly the target profile around known centres gives radial and line
  profiles equal to the target (atol 0.02), peak height 1 (atol 0.02) and FWHM 0.53 R (atol 0.05);
  on a map that is only the wide Gaussian, FWHM 0.78 R (atol 0.05).
- **G4c-C6** CV mode errors (D6) raise `ValueError` before chunking or training.
- **G4c-C7** Resume logic with the fold process stubbed (no training): finished folds are
  skipped, an unfinished fold is cleared and rerun, a failing fold raises `RuntimeError` with the
  fold number and log path, a different K or seed in `cv_config.json` raises `ValueError`; the
  fold command line carries `cv_all_folds=False`, `cv_fold=k` and the fold's directories.
- **G4c-C8** (slow) End to end on synthetic multi-class data, K = 3, tiny model, 1 epoch per fold:
  `tomocpt train` with `cv_all_folds=True` trains 3 folds from one chunk pool, writes
  `ensemble.json` (loads with `load_ensemble`), `oof_picks.npz`, `cv_summary.md`,
  `cv_summary.json`, `peak_profiles.png`; the ensemble build predicts no tomogram again (spy or
  log count); a second identical command trains nothing and rebuilds the same thresholds.

## Acceptance for humans
- `tomocpt train --config_file <cfg> train.cv_folds=5 train.cv_all_folds=True` reproduces the
  Gate 7 baseline workflow end to end in one command, with one chunk pool (~8 GB instead of ~40).
- The CZII baseline CV of TODO 3b.4 is run with this mode (no separate check run needed).
