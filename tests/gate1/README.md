# Gate 1 tests

Written by the Test Author from `.claude/agents/01_AGENT_BASELINE.md` before the implementer starts.
Run the fast suite with `pytest tests/gate1 -m "not slow"` and the slow one with `pytest tests/gate1 -m slow`.

## Criteria and tests

| Criterion | Test(s) | File |
|---|---|---|
| G1-C1 Gaussian target golden | `test_G1_C1_gaussian_sphere_golden[5,10,17]` | `test_g1_goldens.py` |
| G1-C2 Loss golden | `test_G1_C2_loss_golden` | `test_g1_goldens.py` |
| G1-C3 Forward golden (tiny `MySwinUNETR`) | `test_G1_C3_tiny_forward_golden` | `test_g1_goldens.py` |
| G1-C4 Legacy checkpoints load | `test_G1_C4_legacy_picking_checkpoint_loads` (`BasePickingModel` + `InferenceModel(strict=True)`), `test_G1_C4_legacy_ssl_checkpoint_loads`, `test_G1_C4_ssl_to_picking_through_create_model`, `test_G1_C4_legacy_checkpoint_on_gpu` (slow) | `test_g1_checkpoints.py` |
| G1-C5 Resume | `test_G1_C5_legacy_checkpoint_resumes` | `test_g1_checkpoints.py` |
| G1-C6 Distillation with a legacy teacher | `test_G1_C6_distillation_with_legacy_teacher` | `test_g1_checkpoints.py` |
| G1-C7 Config | `test_G1_C7_sections_instantiate_alone`, `test_G1_C7_main_config_sections_are_isolated`, `test_G1_C7_update_config_keeps_target_defaults_for_missing_source_fields`, `test_G1_C7_backfill_adds_every_field_a_legacy_checkpoint_lacks`, `test_G1_C7_safe_getattr` | `test_g1_config_cli.py` |
| G1-C8 Default augmentation pipeline | `test_G1_C8_default_augmentation_pipeline_unchanged` | `test_g1_goldens.py` |
| G1-C9 `tomocpt init` round trip | `test_G1_C9_init_writes_a_config_that_loads_back` | `test_g1_config_cli.py` |
| G1-C10 Metrics | `test_G1_C10_match_exact`, `_match_one_miss_and_one_false_positive`, `_match_just_outside_radius`, `_match_two_predictions_compete_for_one_truth`, `_match_is_greedy_by_increasing_distance`, `_match_empty_inputs`, `_prf_beta`, `_score_classes_per_class_and_macro`, `_score_classes_perfect_and_missing_prediction_class` | `test_g1_metrics.py` |
| G1-C11 Zenodo checkpoint (optional) | `test_G1_C11_zenodo_checkpoint_loads_for_predict` (skips unless `TOMOCPT_ZENODO_CKPT` is set) | `test_g1_checkpoints.py` |
| T1.4 helpers | `test_G1_helpers_*` (self-checks of `tests/utils`) | `test_g1_helpers.py` |

## Goldens and legacy checkpoints

- `tests/golden/make_goldens_gate1.py` was run in a worktree of `v2-gate-0-base` (commit ce6f7b0):
  `cd ../tomocpt-base && PYTHONPATH=$PWD python ../tomocpt/tests/golden/make_goldens_gate1.py --expect-root $PWD`.
  It checks that `tomocpt` comes from the worktree. Outputs: `gate1_spheres.npz` (C1), `gate1_forward.npz` (C3),
  `gate1_goldens.json` (C2 loss value, C8 transform description, provenance).
- `tests/golden/gate1_common.py` builds the inputs (seeded `np.random.RandomState`) and the transform description.
  The generator and the tests both use it, so the 2×64³ loss inputs are rebuilt, not stored. Their sums are also
  stored in the golden as a check.
- Legacy checkpoints (`utils.legacy`) are built on first use by `tests/fixtures/make_legacy_ckpt.py` in
  `../tomocpt-base`. If the worktree is missing it is created. The checkpoints are cached in `tests/.cache/legacy/`,
  which is git-ignored by the existing `.cache` rule. A failed build raises an error, so the test fails and does not skip.

## Runtime (RTX A6000 box, CPU tests, 64 cores)

- `pytest tests/gate1 -m "not slow"`: about 29 s with a warm cache, about 50 s cold (the cold run includes building
  both legacy checkpoints in the base worktree).
- `pytest tests/gate1 -m slow`: about 4 s (one GPU test).
- Full `pytest tests`: 132 passed, 27 skipped, 10 failed (the expected failures below). Before Gate 1 it was 109 passed, 26 skipped.

## Results on `v2-gate-0-base` (= current HEAD ce6f7b0)

The tests were run against base code from a scratch copy of the worktree with these test dirs added.
Results are identical on HEAD: **23 passed, 1 skipped (C11), 10 failed**. The 10 failures are expected because they cover new behaviour:

- `test_G1_C7_safe_getattr`: `tomocpt.configManager.configManager.safe_getattr` does not exist yet (T1.1).
- All 9 `test_G1_C10_*`: `tomocpt.metrics` does not exist yet (T1.3).

Every other test (C1–C6, C8, C9, the rest of C7) passes on base. These are legacy locks and must keep passing.

## Notes for the orchestrator / implementer

1. **CLI option spelling (spec and README disagree with the code).** G1-C9 and the README write `--output-path` and
   `--config-file`. The legacy CLI keeps underscores and only accepts `--output_path` / `--config_file`; the hyphenated
   form fails with "No such option". The YAML is also merged only when the literal `--config_file` is in
   `sys.argv`, at command registration (`ConfigurableApp.command`). The C9 test uses the spelling the code accepts.
   CLAUDE.md Section 1 does not mention the spelling, so nothing in Section 1 is contradicted.
2. **Legacy checkpoints need a logger.** `TA_TEST_AUTHOR.md` says to save them with `pl.Trainer(..., logger=False)`.
   Legacy `training_step` writes images to `self.logger.experiment`, so `logger=False` crashes. The fixture uses a
   throwaway `TensorBoardLogger`.
3. **Return-structure assumptions for `tomocpt.metrics`** (the spec gives names and arguments only): `tp_pairs` can be
   any `(pred_idx, true_idx)` rows; `score_classes` per-class dicts can be at `res[name]` or `res["per_class"][name]`;
   the macro F-beta can be at `res["macro"]["fbeta"]`, `res["macro"]` (a float) or `res["macro_fbeta"]`. A class with
   truth but no key in `pred` counts as fully missed. Matching is inclusive of the radius only in the sense that
   distances 4.99 / 5.01 against radius 5 must match / not match; the exact boundary is not tested. The tests were
   checked against a scratch reference implementation (all 9 pass).
4. **`make_phantom` returns 4 values.** It returns the NamedTuple `Phantom(volume, coords, polylines, masks)`. The TA
   spec's signature shows 3 values, but the spec also says "also return masks".
5. **Test-layout rule for later gates.** `tests/gateN/` needs an `__init__.py`. Without it, its `conftest.py` replaces
   `tests/conftest.py` as module `conftest`, and `tests/test_contrast_inversion.py` and
   `tests/test_fourier_augmentations.py` fail to collect. Import helpers as `utils.<module>`, not `tests.utils`: a
   site-packages package called `tests` shadows the repo's.
6. Legacy model constructors and checkpoint loading change the global `mainConfig` and `tomocpt.constants`. The autouse
   fixture in `gate1/conftest.py` (`utils.tiny.applied_main_config`) restores both after every test.
