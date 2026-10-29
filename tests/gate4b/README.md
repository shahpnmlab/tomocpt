# Gate 4b tests (k-fold training, OOF calibration, fold ensemble)

Spec: `.claude/agents/04b_AGENT_CV_ENSEMBLE.md`. G4b-C1 is the existing suite.

| Criterion | Tests |
|---|---|
| G4b-C2 `split_rows_by_fold` | `test_g4b_folds.py::test_G4b_C2_folds_partition_rows_and_stems[K]`, `test_G4b_C2_assignment_follows_the_spec_rule_and_is_deterministic`, `test_G4b_C2_invalid_arguments_raise[...]` |
| G4b-C3 cv chunks | `test_G4b_C3_cv_chunks_record_fold_and_refuse_other_cv`, `test_G4b_C3_cv_requested_on_chunks_without_cv_raises`, `test_G4b_C3_cv_with_pipeline_v1_raises` |
| G4b-C4 `build_ensemble` | `test_g4b_ensemble.py::test_G4b_C4_build_ensemble_pools_oof_picks`, `test_G4b_C4_mismatched_class_tables_raise`, `test_G4b_C4_overlapping_validation_tomograms_raise` |
| G4b-C5 `EnsembleModel` | `test_G4b_C5_ensemble_is_the_mean_of_members[sigmoid/softmax]`, `test_G4b_C5_one_member_equals_the_model[sigmoid/softmax]` |
| G4b-C6 predict with an ensemble file | `test_G4b_C6_predict_with_ensemble_file`, `test_G4b_C6_class_thresholds_override_ensemble_calibration` |
| G4b-C7 (slow) | `test_g4b_learning.py::test_G4b_C7_cv_folds_and_ensemble` |

Fast suite: 21 tests, about 20 s on the development machine (CPU). All fail on the current
`v2-dev` (no `split_rows_by_fold`, no `cv` entry or check, no `tomocpt.predict.ensemble`).
C7 has not been timed (needs the implementation); budget < 10 min.

## Interface choices where the spec is open
1. **Fold rule (C2, C3).** Implemented literally: `numpy.random.default_rng(seed).permutation`
   of the sorted distinct stems, stem `i` -> fold `i % K`. Seed 42 by default; seed 7 also checked.
   `split_rows_by_fold` returns DataFrames that keep the original index.
2. **C3** goes through `_prepare_data_if_needed` with `train.cv_folds` / `train.cv_fold` and the
   in-process chunking patch of Gate 2. A different `cv` on existing chunks -> `RuntimeError`
   for: another fold, another K, cv off (`cv_folds = 0`) on cv chunks, and a cv request on chunks
   without a `cv` entry. v1 + `cv_folds > 0` -> `ValueError` whose message mentions `cv` or `fold`
   (so an unrelated chunking error does not pass). The fold's validation tomograms are compared
   by stem with `chunk_layout.json["val_tomograms"]` and with the `val` chunk directories.
3. **C4 picks.** Fold checkpoints are copies of one tiny v2 checkpoint whose pickled
   `train.chunks_dir` (and `shared.chunks_dir`) points at a hand-written `chunk_layout.json`
   (`pipeline_version` 2, class table, `val_tomograms`, `cv`). The Gate 4 `NegInputOracle` is
   swapped in through `InferenceModel.load_from_checkpoint` (public), so **`build_ensemble` and
   ensemble prediction must load members with `InferenceModel.load_from_checkpoint`**. Reference
   thresholds = `calibrate_checkpoint` on a copy with all OOF tomograms pooled (identical models,
   so pooled per-fold picks equal these picks; it calls `calibrate_thresholds` once).
   `cal["classes"]` keys may be ints or strings; `oof_tomograms` compared by stem.
4. **Mismatched class table** = second checkpoint's pickled `prepData.class_diameters` changed
   (same channel count). **Overlap** = fold 1's layout also lists fold 0's tomogram.
5. **`load_ensemble(path)`** is called with the path only; C4 only requires a non-None result.
   C5 takes an `EnsembleModel` from `load_ensemble` if it returns one (directly, in a tuple/list,
   or as a dict value), else builds `EnsembleModel([InferenceModel, ...])`. `predict_step` is
   called with keywords (`apix=`, `classes=`); with `classes` the output must have the members'
   shape (all channels or the subset, whichever the members return).
6. **Softmax head (C5)** = a copy of the sigmoid checkpoint with pickled
   `train.particles.class_head = "softmax"` (Gate 3d style).
7. **C6** runs `tomocpt.predict.run.predict()` through the Gate 4 harness (`run_predict`, CPU,
   in-process Dask) with a hand-written ensemble file in the spec format; outputs are the Gate 4
   names `tomocpt_coords_<class>.star`. "Same coordinates" = same rows (xyz atol 1e-4, score atol
   1e-5).
8. **C7.** Six training tomograms (two per fold), one held-out tomogram, trained through
   `tomocpt.training.run.train` per fold; best checkpoint = newest calibrated `*.ckpt` under the
   fold's `model_dir`. Fold models predict with their own calibration, the ensemble with its
   file. F1 (beta 1, match radius 0.5 x class radius) averaged over classes; ensemble may be
   0.02 below the worst fold (margin, see file docstring). `calibration_min_particles = 10`
   for `build_ensemble`.
