# 04b_AGENT_CV_ENSEMBLE.md — Gate 4b: k-fold training, out-of-fold calibration, fold ensemble

Added 29 Oct 2026 by human decision ("Implement 2"): train one picker per fold of tomograms,
calibrate the per-class thresholds on out-of-fold (OOF) predictions pooled over all folds, and
predict with the average of the fold models. Pipeline v2 only.

## Design
- **Folds.** `train.cv_folds = K` (0 = off, legacy split) and `train.cv_fold = k`. The distinct
  tomogram stems are sorted, permuted with `numpy.random.default_rng(seed)` (seed 42), and stem
  `i` of the permutation goes to fold `i % K`. Fold `k` is the validation set; the rest train.
  Rows that share a stem stay together. The same K and seed give the same folds in every run.
- **Chunks.** `chunk_layout.json` records `"cv": {"folds": K, "fold": k}` (absent when off).
  Training refuses existing chunks whose `cv` entry differs from the request (`RuntimeError`
  naming both), so one `chunks_dir` per fold is needed (the driver does this).
- **Per-fold post-fit calibration** (Gate 4) still runs on the fold's validation tomograms.
- **Ensemble file** (JSON): `{"kind": "tomocpt_ensemble", "version": 1, "checkpoints": [abs
  paths], "calibration": {"version", "beta", "grid", "classes": {class number: {"threshold",
  ...}}, "oof_tomograms": [...]}}`. `build_ensemble(checkpoints, out_path, infer_cfg)` loads each
  fold checkpoint, finds its validation tomograms in the `chunk_layout.json` of its
  `train.chunks_dir`, collects its picks there at the lowest grid threshold (the Gate 4
  calibration path), pools picks and truths over all folds and calls `calibrate_thresholds`
  once (`infer_cfg.calibration_beta`, `calibration_min_particles`). It raises `ValueError` if the
  checkpoints differ in class table or scale settings, or if two folds share a validation
  tomogram. The OOF thresholds of single fold models are used for the ensemble (each tomogram
  has only one OOF model).
- **Ensemble prediction.** `infer.weights` may name an ensemble file. `EnsembleModel` holds the
  fold models; its `predict_step(x, batch_idx=0, apix=None, classes=None)` returns the mean of
  the members' `predict_step`; class table, patch size and checkpoint settings come from the
  first member. Thresholds: `infer.class_thresholds` > ensemble calibration >
  `infer.confidence_threshold`. A one-member ensemble predicts exactly as its checkpoint.
- **Driver.** `experiments/run_cv.sh`: for k in 0..K-1 run `tomocpt train` with
  `train.cv_folds=K train.cv_fold=k`, its own `chunks_dir` and `model_dir`; then
  `python -m tomocpt.predict.ensemble <ckpts...> --out ensemble.json --config_file <yaml>`; then
  `tomocpt predict --weights ensemble.json` on the test tomograms and score them.

## Config
`train.cv_folds: int = 0`, `train.cv_fold: int = 0` (`defaultConfigs/train_config.py`).
`cv_folds > 0` with pipeline v1 raises `ValueError` at data preparation.

## Allowed write paths
- `tomocpt/dataManager/chunking.py` (fold split and the `cv` layout entry only)
- `tomocpt/defaultConfigs/train_config.py` (`cv_folds`, `cv_fold` only)
- `tomocpt/training/helpers.py` (`_prepare_data_if_needed` only)
- `tomocpt/predict/**` (new `predict/ensemble.py`; calibration refactor; loading an ensemble file)
- `experiments/run_cv.sh`, `README.md` (a short section)
- `.claude/reports/gate_04b_report.json`

## Public names
`tomocpt.dataManager.chunking.split_rows_by_fold(tomos_df, folds, fold, seed=42) -> (train_df, val_df)`,
`tomocpt.predict.ensemble.{build_ensemble, is_ensemble_file, load_ensemble, EnsembleModel}`.

## Pass criteria
- **G4b-C1** Defaults (`cv_folds = 0`): every earlier test passes unchanged.
- **G4b-C2** `split_rows_by_fold`: over k = 0..K-1 every stem is in exactly one validation fold;
  train and validation of one fold are disjoint and cover all rows; rows with the same stem stay
  together; deterministic for a seed; `K < 2`, `K >` number of stems, or `k` outside `[0, K)`
  raise `ValueError`.
- **G4b-C3** v2 chunking with `cv_folds = K, cv_fold = k` writes `"cv": {"folds": K, "fold": k}`
  and `val_tomograms` equal to fold k; preparing data with a different `cv` on those chunks raises
  `RuntimeError`; `cv_folds > 0` with pipeline v1 raises `ValueError`.
- **G4b-C4** `build_ensemble` on fold checkpoints: the file loads with `load_ensemble`, lists the
  checkpoints, and its thresholds equal `calibrate_thresholds` on the pooled per-fold picks;
  mismatched class tables or overlapping validation tomograms raise `ValueError`.
- **G4b-C5** `EnsembleModel.predict_step` equals the mean of the members' outputs (atol 1e-6),
  sigmoid and softmax heads; a one-member ensemble equals the single model.
- **G4b-C6** `tomocpt predict` with `--weights ensemble.json` writes the per-class outputs, and with
  a one-member ensemble and the same thresholds gives the same coordinates as the checkpoint;
  `infer.class_thresholds` overrides the ensemble calibration.
- **G4b-C7** (slow) Synthetic multi-class data, K = 3, tiny model: all folds train, the ensemble
  builds with non-null thresholds for classes with enough particles, and ensemble prediction on
  a held-out tomogram gives mean F1 ≥ that of the worst single fold model.
