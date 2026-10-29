# Gate 3d tests (softmax class head with implicit background)

Written by the Test Author from `.claude/agents/03d_AGENT_SOFTMAX_HEAD.md` before the implementer started.
Fast tier: `pytest tests/gate3d -m "not slow"` (CPU, about 25 s). Slow tier: `pytest tests/gate3d -m slow`
(one test, CUDA, two tiny-model trainings).

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (copy of Gate 3b) |
| `test_g3d_probs.py` | C2, C3 (`class_probabilities`) |
| `test_g3d_model.py` | C4, C5, C6 (`training_step`, checkpoints, `predict_step`) |
| `test_g3d_learning.py` | C7 (slow) |

Reused without copying: `gate3.g3_helpers` (configs, v2 items, CPU one-step fit, checkpoint builder) and
`gate3.test_g3_learning` (C13 data, `train_picker`, `_c13_eval`). C1 = the existing suite (no new test).

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| C2 | `test_G3d_C2_one_class_equals_sigmoid[none/mask/indices]`, `_sigmoid_head_is_sigmoid`, `_multi_class_ranges_and_formula`, `_index_sequence_equals_mask`, `_finite_for_extreme_logits`, `_bf16_input` |
| C3 | `test_G3d_C3_unavailable_logit_has_no_effect[80/-80/noise]` |
| C4 | `test_G3d_C4_one_class_softmax_loss_equals_sigmoid_loss`, `test_G3d_C4_targets_above_one_are_renormalised` |
| C5 | `test_G3d_C5_softmax_training_step_gradients` |
| C6 | `test_G3d_C6_class_head_stored_in_checkpoint`, `_default_is_sigmoid`, `_softmax_predict_step`, `_sigmoid_predict_step_ignores_classes`, `_checkpoint_without_field_predicts_sigmoid[missing/softmax]`, `_invalid_class_head_raises` |
| C7 | `test_G3d_C7_softmax_fixes_c13_without_k` (slow) |

## Interface choices (least constraining)

1. **Strict ranges.** `Σ p < 1` and `p ∈ (0, 1)` are checked on moderate logits (`3·randn`). At logits of ±80
   float32 rounds `exp(80)/(1+exp(80))` to 1, so there only finite, `[0, 1]` and `Σ p ≤ 1 + 1e-6` are asserted,
   plus agreement with a float64 reference (atol 1e-6).
2. **bf16.** Output must be float32 and equal the float32 result on the same (bf16-rounded) values, atol 1e-6.
3. **Sigmoid head.** Only `class_probabilities(logits, "sigmoid")` with `available=None` is tested (= sigmoid).
   What the sigmoid head does with `available` is not specified and not tested.
4. **Loss in softmax mode** is tested through the first `training_step` (CPU, fp32): it must equal
   `multiclass_picking_loss(class_probabilities(logits, "softmax", class_available), y / max(1, Σ_c y_c), ...,
   checkpoint=False)`, rel 1e-5 (Gate 3b convention). Where the renormalisation sum runs (all channels or only
   available ones) is open: the C4 overlap test has every class available, and the C5 test accepts either.
   The C4 overlap scene changes the loss by about 4.7e-4 relative, so a missing renormalisation is caught.
5. **`predict_step(..., classes=...)`** may return all C channels or only the `classes` channels; the channels of
   `classes` must equal `class_probabilities(logits, "softmax", classes)` (sigmoid model: `sigmoid`). The test
   also checks the subset result differs from the all-class one. The `_predict_level` → `classes` wiring in
   `predict/multiclass.py` is private and not tested here.
6. **Old checkpoints.** The softmax checkpoint with `class_head` popped from the pickled `ParticlesConfig`
   instance must predict with sigmoid; the control (field kept) with softmax.
7. **Invalid value**: `ValueError` from `BasePickingModel(...)` construction with `class_head = "softmaxx"`.
8. **C7** wraps `gate3.g3_helpers.v2_config` (as Gate 3b C5) to set `class_head = "softmax"` and k = 0, and spies
   on `tomocpt.networks.pickingModel.multiclass_picking_loss` (the name Gate 3 imports): every training call must
   get k = 0 and probabilities whose per-voxel channel sum is ≤ 1 + 1e-3 (a sigmoid head fails this). Cross-pick
   rate = mean over seeds 0, 1 and both directions; F1 ≥ 0.9 for all four classes on each seed.

## Results on HEAD (1c3f2ae)

Fast tier: 21 failed (missing `class_probabilities`, `ParticlesConfig.class_head`, `predict_step(classes=)`;
the sigmoid `predict_step` test fails on the missing `classes` keyword). Slow test: fails (no `class_head`).

## Scratch reference implementation

A throwaway implementation (session scratchpad, `pytest -o pythonpath=<ref>`) passes all 21 fast tests.
Mutation "no target renormalisation" fails `test_G3d_C4_targets_above_one_are_renormalised`.
The reference also passes C7 (RTX A6000, 9 min 2 s for both seeds): seeds 0 and 1 give F1 1.0 for all four
classes and cross-pick 0.0 both ways (thresholds 0.9 / 0.05). Only two seeds were run, so the 9-of-10 rule is
not demonstrated; the margin is wide.
