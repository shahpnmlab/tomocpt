# Gate 3 tests (multi-class picker, voxel-size conditioning)

The Test Author wrote these tests from `.claude/agents/03_AGENT_MULTICLASS_PICKER.md` before the implementer started.
Fast tier: `pytest tests/gate3 -m "not slow"`. Slow tier: `pytest tests/gate3 -m slow` (two learning tests, need a CUDA GPU).

New public names are imported inside the test functions, so the v1 locks still run on code that does not
have the new names yet.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse), `tiny_global`, legacy checkpoints, `v2_ckpts` (pipeline-v2 checkpoints written by the code under test, built once per session) |
| `g3_helpers.py` | Helpers (no tests): tiny v2 configs, v2 batches (CLAUDE.md 5.1 keys, targets from `render_targets`), CPU trainer, one-step fit that captures the loss and the gradients after backward, parameter-group reader |
| `test_g3_v1_lock.py` | C1, the v1 part of C9 |
| `test_g3_modules.py` | C2, C3, C7 (module level) |
| `test_g3_loss.py` | C4, C5, C6, and T3.3 (v2 `training_step` / `validation_step`, per-class val metrics) |
| `test_g3_checkpoints.py` | C7 (class-table change), C8, C9 (v2), C10, C11 |
| `test_g3_learning.py` | C12, C13 (slow) |
| `../golden/gate3_common.py` | Shared by the golden generator and the C1 test: fixed v1 batch, deterministic re-initialisation, one-batch fit |
| `../golden/make_goldens_gate3.py` | Golden generator (run in the `v2-gate-0-base` worktree) |
| `../golden/gate3_goldens.json` | C1 golden |

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| G3-C1 v1 lock | `test_G3_C1_v1_training_step_loss_matches_base_golden`, `test_G3_C1_v1_with_several_single_class_datasets_keeps_one_channel` |
| G3-C2 Zero-init | `test_G3_C2_structure_scale_embedding_and_film_zero_init`, `test_G3_C2_zero_init_matches_legacy_logits`, `test_G3_C2_apix_none_is_exactly_unconditioned_even_with_trained_film`, `test_G3_C2_scalar_apix_matches_tensor_apix`, `test_G3_C2_load_legacy_state_dict_raises_on_bad_keys` |
| G3-C3 Gradients with conditioning | `test_G3_C3_every_parameter_gets_a_gradient_with_conditioning`; through the LightningModule: `test_G3_C5_ddp_every_parameter_has_a_gradient_in_a_v2_training_step` |
| G3-C4 Loss equivalence | `test_G3_C4_single_class_equals_v1_loss[True, False]`, `test_G3_C4_checkpointed_per_class_equals_vectorised_and_reference`, `test_G3_C4_class_weights_multiply_each_class`, `test_G3_C4_hessian_is_normalised_by_d_squared`, `test_G3_C4_v2_training_step_loss_is_the_multiclass_loss[False, True]` (also checks the T3.3 `val_loss_<class name>` metrics) |
| G3-C5 Partial labels | `test_G3_C5_unavailable_class_has_exactly_zero_gradient_and_ignores_its_target[True, False]`, `test_G3_C5_unavailable_channel_through_the_model`, `test_G3_C5_no_available_pair_gives_zero_loss_and_zero_gradient`, `test_G3_C5_ddp_every_parameter_has_a_gradient_in_a_v2_training_step` |
| G3-C6 Per-class weights | `test_G3_C6_rare_class_gets_larger_positive_weights_per_sample_and_class` |
| G3-C7 Warm start | `test_G3_C7_single_class_warm_start_every_channel_equals_old_picker`, `test_G3_C7_class_table_change_copies_matching_channels`, `test_G3_C7_same_class_table_keeps_every_channel` |
| G3-C8 Legacy -> v2 fine-tune | `test_G3_C8_legacy_checkpoint_into_v2_model[C1_cond, C3, C3_cond]`, `test_G3_C8_ssl_checkpoint_into_v2_model` |
| G3-C9 Parameter groups | `test_G3_C9_v1_parameter_groups_unchanged`, `test_G3_C9_v2_parameter_groups[False, True]`, `test_G3_C9_groups_after_fine_tune_from_legacy` |
| G3-C10 Errors | `test_G3_C10_conditioning_without_apix_raises`, `test_G3_C10_resume_with_a_different_architecture_raises[class_table, scale_conditioning]`, `test_G3_C10_resume_with_the_same_architecture_works`, `test_G3_C10_resume_legacy_checkpoint_with_conditioning_raises`, `test_G3_C10_distillation_with_v2_student_raises[v2_cond, v2_multi, v2_single]`, `test_G3_C10_distillation_through_create_model_raises`, `test_G3_C10_unet_with_scale_conditioning_raises` |
| G3-C11 Inference loads v2 | `test_G3_C11_inference_model_loads_v2_checkpoint[c2, c2_cond]` |
| G3-C12 Scale disambiguation (slow) | `test_G3_C12_scale_disambiguation` |
| G3-C13 Same size, different shape (slow) | `test_G3_C13_same_size_different_shape` |

44 fast test cases and 2 slow ones.

## Golden (G3-C1)

`tests/golden/make_goldens_gate3.py` was run in the `v2-gate-0-base` worktree (commit ce6f7b0):
`cd ../tomocpt-base && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python ../tomocpt/tests/golden/make_goldens_gate3.py --expect-root $PWD`
(the script checks that `tomocpt` comes from the worktree; the worktree's tracked files were not modified).

- Model: tiny `BasePickingModel` (`feature_size=12`, dropout 0, `use_checkpoint=False`, CPU fp32), built from the
  tiny config. Its weights are then overwritten by `gate3_common.reinit_` (seeded `np.random.RandomState`, sorted
  parameter names), so the golden does not depend on how much torch RNG a constructor consumes.
- Batch: `gate3_common.v1_batch()` (input `RandomState(31)` normal clipped to [-3, 3]; target = the G1-C2 Gaussian
  spheres). The input and target sums are stored as a check that the rebuilt batch is the same.
- The loss is the value `training_step` returns on the first batch of a one-batch `trainer.fit` (a throwaway
  `TensorBoardLogger`, because legacy `training_step` writes images to `self.logger.experiment`; one validation batch
  so the `val_loss_smooth` plateau monitor exists). Golden: `7.267982482910156`, compared with rtol 1e-6.
  The eval-mode logits sum is also stored (rtol 1e-5) to catch a weight mismatch before the loss.

## Slow tests (G3-C12, G3-C13)

Training data are built **in memory** from `utils.phantoms.make_phantom` (not through chunking: one chunking run per
tomogram set costs more than it adds here, and Gate 2 already tests the chunk -> batch path). Each phantom is
robust-normalised (`clip((x - median) / (p95 - p5), -3, 3)`); every training item is a random 64^3 crop with random
axis flips (they keep the wedge orientation), with the CLAUDE.md 5.1 keys and targets drawn by
`labels.helpers.render_targets`. Training runs `BasePickingModel.training_step` through a Lightning trainer
(GPU, bf16-mixed, gradient clip 1.0, RAdam lr 1e-3, batch 4). Evaluation: held-out 64^3 phantom chunks at 10 A/px
(one forward each, `predict_step`), peaks per channel with `peak_local_max_torch` (threshold 0.3, min distance
`int(d/2)`), hits within `d/4` voxels with `tomocpt.metrics.match_points`. `TOMOCPT_G3_SEED` selects the seed.

- **C12**: classes A (150 A) and B (300 A). Training: 6 phantoms at 10 A/px (64x128x128, 14 A + 4 B, both
  annotated) and 6 at 20 A/px (14 B, 15 voxels; A unavailable there, as the band rule would make it: 7.5 voxels <
  `band_min` for a 10 A/px native), sampled 50/50, SNR 1, wedge 60 deg, 2000 steps. Held out: 12 chunks of 64^3
  at 10 A/px with 1 B + 6 A each. Asserted for the conditioned model: F1_A >= 0.8, F1_B >= 0.8, mean B prediction
  at A centres < 0.2. The unconditioned model is trained the same way and only reported (`G3_RESULT` line on
  stdout; `TOMOCPT_G3_SKIP_UNCOND=1` skips it, used for the seed sweep only).
- **C13**: four classes at 10 A/px: small 100 A solid, "solid" 200 A, "hollow" 200 A (wall 0.2 R), large 300 A,
  scale conditioning off. Training: 8 phantoms (64x128x128; 3 large, 5 solid, 5 hollow, 8 small), SNR 2, wedge 60,
  1800 steps. Held out: 12 chunks (1 large, 2 solid, 2 hollow, 3 small each). Asserted: F1 >= 0.8 per class;
  cross-pick (fraction of true particles of one same-size class with a peak of the other's channel within d/4)
  < 0.15 both ways.

### Seed sweep on the reference implementation (`TOMOCPT_G3_SEED`, default 0)

C12 (2000 steps; seeds 0-2 also trained the unconditioned model, report only):

| seed | F1_A | F1_B | mean B at A (cond) | uncond F1_A / F1_B / mean B at A |
|---|---|---|---|---|
| 0 | 0.923 | 1.0 | 0.130 | 1.000 / 0.923 / 0.263 |
| 1 | 0.923 | 1.0 | 0.122 | 1.000 / 1.000 / 0.124 |
| 2 | 1.000 | 1.0 | 0.028 | 0.954 / 1.000 / 0.109 |
| 3 | 0.923 | 1.0 | 0.161 | - |
| 4 | 0.923 | 1.0 | 0.115 | - |
| 5 | 1.000 | 1.0 | 0.124 | - |
| 6 | 1.000 | 1.0 | 0.116 | - |
| 7 | 1.000 | 1.0 | **0.216** (fails < 0.2) | - |
| 8 | 1.000 | 1.0 | 0.071 | - |
| 9 | 1.000 | 1.0 | 0.000 | - |

9 of 10 seeds pass: exactly the required rate, with a thin margin on "mean B at A" (worst passing seed 0.161).
F1 has a wide margin (min 0.923; the 0.923 cases are one A false positive per chunk, at the B particle).
The unconditioned model is higher at seed 0 (0.263) but close to the conditioned one at seeds 1-2:
on these phantoms the network partly infers the scale from the image itself, so the benefit of
conditioning is smaller than the spec expects. This is a report-only number.

C13 (1800 steps; only seeds 0-2 were run, the budget ran out):

| seed | F1 small / solid / hollow / large | cross solid->hollow / hollow->solid | result |
|---|---|---|---|
| 0 | 0.847 / 1.0 / 1.0 / 1.0 | 0.0 / 0.0 | pass |
| 1 | 0.857 / **0.571** / 1.0 / 1.0 | 0.0 / **1.0** | **fail** |
| 2 | 1.0 / 0.842 / 1.0 / 1.0 | 0.0 / 0.0 | pass |

In seed 1 the solid channel fires on every hollow particle (24 FP). Calibration runs on seed 1 with batch 2 x
3600 steps, lr 5e-4, or no wedge did not fix it within 3000 steps. So **C13 is not shown to pass in 9 of 10
seeds** with a tiny model inside the time budget. The thresholds were not loosened. The test runs seed 0,
which passes (small-class F1 0.847, margin 0.047). The orchestrator decides: accept a seed-0 test with this
known flakiness, give C13 a larger budget (more steps or a larger model), or change the criterion.

## Runtime (RTX A6000 box, 64 cores)

- `pytest tests/gate3 -m "not slow"`: about 40 s (CPU; the session builds two small v2 checkpoints and uses the
  cached legacy checkpoints).
- `pytest tests/gate3 -m slow`: about 12 min on an otherwise idle A6000 (C12: two
  models x 2000 steps, about 4.2 min each; C13: 1800 steps, about 3.8 min; tiny SwinUNETR costs about 28 ms per
  64^3 sample per step, mostly GPU-bound). Running several in parallel does not raise throughput.
- Full `pytest tests -m "not slow"` on HEAD: **391 passed, 27 skipped, 40 failed** (387 + the 4 Gate 3 v1 locks;
  the 40 failures are the Gate 3 new-behaviour tests). Against the reference implementation: 435 passed, 27 skipped.

## Results on HEAD (`v2-gate-2-passed`, 333c20a)

Fast tier: **4 passed, 40 failed**.
- Pass (v1 locks, must keep passing): `test_G3_C1_v1_training_step_loss_matches_base_golden`,
  `test_G3_C1_v1_with_several_single_class_datasets_keeps_one_channel`, `test_G3_C9_v1_parameter_groups_unchanged`.
  (`v2-gate-0-base` code produced the golden, so C1 also holds there; base predates the Gate 1 `out`-layer group fix,
  so the v1 C9 grouping lock reflects HEAD, which is the state Gate 3 must not change.)
- Fail (new behaviour): every other test. They fail on missing names (`tomocpt.networks.conditioning`,
  `ScaleConditionedSwinUNETR`, `load_legacy_state_dict`, `load_single_class_into_multiclass`,
  `tomocpt.training.multiclass_loss`, `NetworkConfig.scale_conditioning`, `TrainConfig.particles`), or, where the
  names exist, on behaviour (`BasePickingModel` builds one channel for a v2 config, distillation accepts a v2 student,
  `NetworkConfig` accepts `scale_conditioning` for UNet ...).
- Slow tier on HEAD: both tests fail (no `scale_conditioning` / multi-channel model; not run, they fail at the
  first v2 config field).

## Results on the scratch reference implementation

A throwaway implementation of the spec (written outside the repo, in the session scratchpad, and run by pointing
pytest's `pythonpath` at it) passes all 44 fast tests and the full suite (435 passed, 27 skipped).

Deliberate mutations of the reference, each caught:

| Mutation | Failing tests |
|---|---|
| FiLM not zero-initialised | C2 structure, C2 zero-init, C7 warm start (conditioned), C8 `[C1_cond]`, `[C3_cond]` |
| Hessian not normalised by (d/20)^2 | C4 reference, C4 class weights, C4 Hessian normalisation |
| (d/20)^2 applied to the prediction's Hessian only | same three C4 tests |
| Unavailable pairs not masked (and counted in the mean) | C4 reference, C4 class weights, all five C5 tests |
| Weights per sample over all classes instead of per (b, c) | C6 |
| `training_step` does not pass `apix_work` to a conditioned model | C4 `training_step[True]` |
| New class channel starts at zero instead of the mean | C7 class-table change |
| Resume does not check the architecture | C10 resume (3 tests) |
| Distillation accepts a v2 student | C10 distillation (4 tests) |
| `apix=None` runs the FiLM path with t = 0 | C2 `apix_none_is_exactly_unconditioned...` |

## Spec ambiguities and how the tests resolve them

1. **G3-C4 "normalised Hessian at d = 12 and d = 24 differs by < 10 %" is not true numerically.** With the
   Section 1.3 profile (rendered by `render_targets` or by legacy `generate_gaussian_sphere`) and the kornia `diff`
   stencil, `(d/20)^2 * H` at the centre is 0.187 (d = 12) vs 0.218 (d = 24): 14.1 % (the d = 12 core has
   sigma_s ~ 1 voxel, so the discrete second difference underestimates the curvature). Without normalisation the
   factor is 3.4. The value does not depend on the implementation. The test asserts < 15 % (and > 3x unnormalised),
   and tests the implementation's normalisation through the loss instead: with a small residual the Hessian term
   scales as ((d/20)^2)^2, so (L(40) - L(20)) / (L(20) - L(10)) = 16 (rtol 1e-3). For the orchestrator: either
   accept 15 % or change the spec's pair (d = 16 vs 24 gives 6.3 %).
2. **`y_pred` of `multiclass_picking_loss` is the sigmoid output**, as for `BasePickingModel.loss` (the C = 1
   equivalence requires it).
3. **`class_weights` argument**: one value per channel (tensor or sequence of length C); each available pair's term
   is multiplied by its class weight and the total stays the plain mean over available pairs ("multiply each class").
   `train.particles.class_weights` ("number:weight", by class number) is mapped to channels through the class table;
   classes missing from the string weigh 1.0.
4. **No available pair in the batch** (possible after zoom): the tests require loss 0 and exact zero gradients, not
   NaN (the spec's mean over an empty set is undefined; DDP needs a gradient).
5. **`apix` argument**: the training batch gives `[B, 1]` float32; the tests also require a Python float to work
   (`test_G3_C2_scalar_apix_matches_tensor_apix`), because the spec's Gate 4 note calls `forward(x, apix=level.apix_work)`
   with the float from `plan_levels`.
6. **Module structure**: `ScaleEmbedding(dim=64)` must have exactly the parameters of `Linear(17->64), SiLU,
   Linear(64->64)`; each `FiLM(C, cond_dim)` exactly those of `Linear(cond_dim -> 2C)`; `model.cond.film` holds the six
   FiLM layers in the Design order (encoder10, decoder5, decoder4, decoder3, decoder2, decoder1) with
   16fs, 8fs, 4fs, 2fs, fs, fs channels. The FiLM forward signature is not used by the tests.
7. **`load_legacy_state_dict` errors**: the spec says "raises" without a type; the tests accept any exception.
8. **Which configs build a multi-channel model**: the tests use `train.data_pipeline_version = 2` with
   `prepData.multiclass = True` for every v2 model (also C = 1). A v1 config keeps one output channel even when
   `class_table_from_config` would give several rows (legacy multi-dataset configs; v1 merges them into one heatmap).
   The tiny v1 config has `particle_length_ang` unset (MISSING), as the legacy checkpoints do, so v1 model building
   must not need the class table.
9. **Class-table change**: matched by class number; the test keeps the class name of the copied class unchanged and
   checks the result through logits (`out` is a 1x1x1 conv, so a channel copied from old channel k gives old logits k,
   and a channel initialised with the mean weights and bias gives the mean of the old logits).
10. **Resume checks** are tested only where the requested value differs from the default (class table set,
    `scale_conditioning=True`), so they do not depend on how "requested" is merged with the checkpoint config. A resume
    with the same v2 architecture must work and keep the weights.
11. **Per-class validation metric** `val_loss_<class name>` is checked in `trainer.callback_metrics` (a metric key, not
    log text).
12. **C12 data**: the spec does not say whether A appears in the 20 A/px tomograms. The 20 A/px phantoms contain B
    only and mark A unavailable (what the band rule gives a 150 A class resampled from 10 to 20 A/px).
13. **C13 sizes**: the spec fixes only "two of the same diameter". With 120/200/200/300 A or 150/220/220/300 A the
    tiny model confuses the small and the 200 A solid class (both solid spheres, size ratio < 2) for well over
    1800 steps, which makes the size distinction, not the shape distinction, the bottleneck. The test uses
    100/200/200/300 A (size ratios >= 1.5 and 2 to the neighbours), SNR 2.

## CLAUDE.md Section 1 vs code

No disagreement found in the parts Gate 3 touches (1.4 geometry and state-dict names, 1.5 loss, 1.6 groups, 1.9
backfill). Note: the Gate 1 `out`-layer fix makes fine-tuning groups `[swinViT 0.01, encoder 0.1, decoder 1.0,
rest 1.0]`, as Section 1.6 says.
