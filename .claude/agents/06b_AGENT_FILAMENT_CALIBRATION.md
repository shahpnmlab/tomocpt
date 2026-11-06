# 06b_AGENT_FILAMENT_CALIBRATION.md — Gate 6b: tracing threshold calibration (TODO 3c, handoff step 2)

Added 6 Nov 2026 (`HANDOFF_filaments.md` step 2). Point calibration finds no points for filament
classes: it spends the prediction time and leaves the tracing threshold at the fixed fallback.

## Design
- **Metric moves to the package.** `tomocpt.metrics.centerline_scores` (library code must not import
  `benchmarks`); `benchmarks.filaments` re-exports it. Same values as before, but distances go
  through a KD-tree on the densified reference, with the exact segment distance only for points in
  the ambiguous band `tol < d_dense <= sqrt(tol² + (step/2)²)`.
- **Manifest.** `val_tomograms` entries of tomograms with filament files gain `filaments_path`
  (several files are merged into `<chunks_dir>/val_coords/<dir>.filaments.csv`, ids unique). Older
  chunks lack it: those classes stay uncalibrated.
- **Point calibration skips filament rows**: their `truths` are `None`, `picks` empty.
- **Scores per threshold.** `filament_threshold_scores(prob, true_polylines, diameter_ang, apix,
  infer_cfg, grid)` returns `[G, 5]`: covered predicted length, predicted length, covered true length,
  true length (Å, so voxel sizes pool), number of true filaments. Tolerance `0.5 × diameter`, as the
  benchmark. Tracing uses the infer settings (`labels.filaments._trace_and_fit`), so the score is of
  what `predict` would output. Grid `FILAMENT_GRID` = 0.1 .. 0.9 (tracing is expensive).
- **Guard.** A threshold whose region exceeds `FILAMENT_MAX_REGION_FRAC` = 0.25 of the volume is not
  traced (skeletonising a near-full volume takes minutes, and no such trace can be right): predicted
  length 0 for it.
- **Choice.** `calibrate_filament_thresholds(scores, grid, beta)`: pools the lengths over tomograms,
  maximises F-beta, ties pick the higher threshold. No true length, or best F-beta 0: `threshold`
  None. Entry: `threshold, fbeta, n_true (true filaments), kind="filament", precision, recall, f1,
  true_length_ang, curve ([[thr, precision, recall, fbeta], ...])`.
- **`infer.filament_calibration_beta` = 1.0**, separate from `calibration_beta`: F4 is the CZII particle
  convention and almost ignores false traces, which matter for subtomogram averaging.
- **Storage.** The entries go in `tomocpt_calibration["classes"]`, so `resolve_thresholds` picks them up
  unchanged. Saved picks (`.val_picks.npz`, `oof_picks.npz`) gain `filament_numbers`, `filament_grid`,
  `filament_scores` (`[F, T, G, 5]`, NaN where unavailable) and `filament_settings` (JSON of the three
  tracing settings); `picks_fingerprint` is unchanged, so particle picks already saved stay reusable.
  `build_ensemble` predicts again if a model with filament rows has saved picks with other settings.
- **CV report.** Filament rows report length-weighted precision, recall, F1, F-beta and per-fold F-beta;
  no point counts or localisation. Their `threshold_used` is the calibrated one or None.

## Allowed write paths
`tomocpt/metrics.py` (`centerline_scores`), `tomocpt/predict/{calibration,ensemble,cv_report}.py`,
`tomocpt/dataManager/chunking.py` (`_v2_val_tomograms`), `tomocpt/defaultConfigs/infer_config.py` (one
field), `benchmarks/filaments.py`, `README.md`, `CLAUDE.md` (tables), `TODO_v2.md`,
`HANDOFF_filaments.md`, `.claude/reports/gate_06b_report.json`.

## Pass criteria
- **G6b-C1** `tomocpt.metrics.centerline_scores` equals a brute-force exact computation, including points
  in the ambiguous band; `benchmarks.filaments.centerline_scores` is the same object.
- **G6b-C2** On a map with one strong and one weak tube, the scores show the weak tube's false length at
  low thresholds and recall loss at 0.9; the calibration picks a threshold that keeps only the strong one.
- **G6b-C3** `calibrate_filament_thresholds`: pooling by length, ties, unavailable tomograms, beta, None cases.
- **G6b-C4** The guard does not trace over-full regions.
- **G6b-C5** `_v2_val_tomograms`: `filaments_path` absent, single, merged with unique ids.
- **G6b-C6** `calibrate_checkpoint` on a particle + filament checkpoint (oracle maps): the filament
  threshold is calibrated and used by `resolve_thresholds`; the particle row is as before; saved picks
  round-trip the scores.
- **G6b-C7** A particle-only table: no filament keys in the record, fingerprint unchanged.
- **G6b-C8** `build_ensemble` pools filament scores and `write_cv_report` reports them, with stale tracing
  settings predicted again.
