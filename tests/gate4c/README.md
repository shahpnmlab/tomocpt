# Gate 4c tests (built-in k-fold CV, saved picks, isolated loads, CV report; TODO 3a)

Spec: `.claude/agents/04c_AGENT_CV_ROUTINE.md`. The interface the tests fix is in the docstring
of `g4c_helpers.py`. G4c-C1 also means: the existing suite passes unchanged.

| Criterion | Tests |
|---|---|
| G4c-C1 defaults | `test_g4c_cv_mode.py::test_G4c_C1_defaults` (+ full suite) |
| G4c-C2 shared chunks | `test_g4c_chunks.py` (fold names, equality per fold k = 0..2, no re-chunking, pool reuse, pool mismatch) |
| G4c-C3 saved picks | `test_g4c_picks.py` (round trip, reuse by `build_ensemble`, stale files not reused, `oof_picks.npz`) |
| G4c-C4 isolated loads | `test_g4c_isolation.py` (either order, with a CLI snapshot, `load_ensemble`, `build_ensemble`) |
| G4c-C5 report | `test_g4c_report.py` (counts, F-beta, per fold, weights, localisation, peak profiles) |
| G4c-C6 errors | `test_g4c_cv_mode.py::test_G4c_C6_invalid_cv_mode_raises_before_work[...]` |
| G4c-C7 driver / resume | `test_g4c_cv_mode.py::test_G4c_C7_*` (fold process and ensemble step stubbed) |
| G4c-C8 (slow) | `test_g4c_learning.py::test_G4c_C8_cv_mode_end_to_end` |

## Interface choices where the spec is open
1. **Spies.** `collect_validation_picks` is counted by replacing
   `tomocpt.predict.calibration.collect_validation_picks`; the implementation must call it
   through the module attribute (not a name bound at import time) in `calibrate_checkpoint` and
   in `build_ensemble`, and must pass `profiles=` by keyword.
2. **Fold process.** `tomocpt.training.cv._run_fold_process(cmd, log_path) -> int` and
   `_build_ensemble_and_report(config, checkpoints, cv_dir)` are module attributes the tests
   replace. The fold's checkpoint is `<model_dir>/<experiment_name>/checkpoints/weights.ckpt` of
   the fold config (Lightning's layout in this repo).
3. **C2 equality.** Chunk `.nii.gz` files are compared by voxel array and affine (gzip headers may
   hold a time stamp); every other file byte for byte; `chunk_layout.json` as parsed JSON.
4. **C5 thresholds.** The report reads the thresholds of the ensemble record and the match radius
   `0.5 R` of Gate 4 (`R` from `oof_picks.npz`, native voxels); F-beta uses the record's `beta`.
5. **C6** asserts nothing was chunked and no fold folder exists after the error.
6. **C7** uses real pool chunking (in-process, tiny phantoms) and stubbed fold processes.
