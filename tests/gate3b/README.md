# Gate 3b tests (cross-class negative weighting)

The Test Author wrote these tests from `.claude/agents/03b_AGENT_CROSS_CLASS_NEG.md` before the implementer started.
Fast tier: `pytest tests/gate3b -m "not slow"` (CPU, about 35 s). Slow tier: `pytest tests/gate3b -m slow`
(one learning test, CUDA GPU, about 27 min: six tiny-model trainings of about 4.5 min each).

The new keyword `cross_class_neg_weight` and the config field are used inside the test functions, so the
modules import on code without them.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse) |
| `g3b_reference.py` | Independent reference of the Gate 3b loss written from the spec (Section 5.4 per available pair plus the cross-class term); no tests |
| `test_g3b_loss.py` | C1, C2, C3 |
| `test_g3b_model.py` | C4 (and a v1 lock) |
| `test_g3b_learning.py` | C5 (slow) |
| `../golden/gate3b_common.py` | Shared by the generator and the C1 test: three deterministic loss cases |
| `../golden/make_goldens_gate3b.py` | C1 golden generator (refuses to run on code that already has the keyword) |
| `../golden/gate3b_c1_loss.npz` | C1 golden (1.7 MB) |

Reused without copying: `gate3.g3_helpers` (configs, v2 items, CPU one-step fit, checkpoint builder),
`gate3.test_g3_loss._c4_inputs` (random C = 4 batch), `gate3.test_g3_learning` (the whole C13 scenario:
data, `train_picker`, `_c13_eval`).

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| G3b-C1 k = 0 equals Gate 3 | `test_G3b_C1_k0_equals_gate3_loss_golden[default/k0 × c1/c4/c4w × checkpoint True/False]` (12) |
| G3b-C2 Weights | `test_G3b_C2_voxel_weights_follow_the_formula[k=0, 0.5, 2 × checkpoint]`, `test_G3b_C2_extra_weight_only_on_negatives_inside_other_available_classes`, `test_G3b_C2_unavailable_class_target_is_ignored[checkpoint]`, `test_G3b_C2_loss_and_gradient_equal_reference[k=0.5, 2]` |
| G3b-C3 Checkpointed = vectorised | `test_G3b_C3_checkpointed_equals_vectorised[k=0.25, 3]` |
| G3b-C4 `training_step` / old checkpoints | `test_G3b_C4_config_field_default`, `test_G3b_C4_training_step_uses_config_k[None, 0.0, 1.5]`, `test_G3b_C4_checkpoint_without_field_loads_with_k0[missing, 1.5]`, `test_G3b_C4_v1_training_step_ignores_k` |
| G3b-C5 C13 with k (slow) | `test_G3b_C5_cross_class_negatives_fix_c13` |

32 fast test cases, 1 slow.

## Choices

1. **C1 golden.** Written by `make_goldens_gate3b.py` from HEAD 2b62832 (Gate 3 loss = `v2-gate-4-passed` code;
   the script exits if the keyword already exists). Three cases on a 32^3 cube with overlapping particles of
   different classes (`render_targets`), `y_pred = sigmoid(seeded normal)`: C = 1 with B = 3 (one sample
   unavailable), C = 4 with partial availability, C = 4 with partial availability and class weights. The golden
   holds the loss for `checkpoint` True and False and the full gradient w.r.t. `y_pred` (identical for both on
   HEAD, so stored once). The test compares the loss with rtol 1e-6 and the gradient with rtol 1e-6,
   atol 1e-6·max|grad| (an equal-but-reordered computation may round differently), both with the keyword
   omitted and with `cross_class_neg_weight=0.0`. The test also requires the keyword with default 0.0. Sums of
   the rebuilt inputs are checked against the golden first.
2. **C2 exact weights.** A fixed C = 3 scene (class 0 overlaps class 1, class 1 overlaps class 2, class 2
   unavailable in sample 0 and available in sample 1; same targets in both samples), float64,
   `y_pred = y_true + 0.01`: a constant shift gives the Hessian term no gradient and keeps Huber quadratic, so
   the gradient at a voxel is exactly `w · δ / (N · n_pairs)`. 16 named voxels (positive only, positive inside
   another class, another class's centre, a soft edge of another class with y ≈ 0.5, inside two other classes
   with different values (checks the max), the unavailable class's centre in sample 0 and in sample 1,
   background) are compared with the spec formula at rel 1e-6. A mask-level test checks that every voxel with
   `y_c > 0` or `o = 0` keeps the k = 0 gradient and every voxel with `y_c = 0, o > 0` gets a larger one. The
   full loss and gradient are also compared with `g3b_reference.reference_loss` on a random 64^3 C = 4 batch
   and the weighted golden case (rtol 1e-5).
3. **"Unavailable d does not raise c"** is tested twice: by the named voxel at class 2's centre in sample 0
   (weight `2 w_neg`), and by replacing the unavailable target with noise (loss and gradients bit-identical).
4. **C4.** `training_step` is checked through the loss of a one-batch CPU fit (as Gate 3 C4). The old-checkpoint
   test writes a v2 checkpoint with the code under test, removes `cross_class_neg_weight` from the pickled
   `ParticlesConfig` instance and loads it with `BasePickingModel.load_from_checkpoint`; the control sets the
   field to 1.5 and must train with 1.5. Validation loss with k is not specified and not tested.
5. **C5** reuses the Gate 3 C13 code unchanged; k is injected by wrapping `gate3.g3_helpers.v2_config`
   (`monkeypatch`), which `train_picker` calls. It trains seeds 0-2 with the recommended k and with k = 0 (six
   runs). "Cross-pick rate" = mean over seeds and both directions (`solid_in_hollow`, `hollow_in_solid`).
   `TOMOCPT_G3B_SKIP_K0=1` uses the Gate 3 k = 0 numbers (recorded in the test) instead of retraining;
   `TOMOCPT_G3B_K` overrides k (calibration). The test also checks that the training-step loss calls get
   `cross_class_neg_weight=k` (a spy on `tomocpt.networks.pickingModel.multiclass_picking_loss`, the name Gate 3
   imports). So `BasePickingModel` must pass k by that keyword.

## Calibration of k (C5)

Scratch reference implementation, RTX A6000 (idle), the Gate 3 C13 setup unchanged (1800 steps, bf16, batch 4,
lr 1e-3). Each cell: F1 small / solid / hollow / large; cross solid->hollow / hollow->solid.

| k | seed 0 | seed 1 | seed 2 | C13 passes |
|---|---|---|---|---|
| 0 (Gate 3 README) | 0.847 / 1.0 / 1.0 / 1.0; 0 / 0 | 0.857 / **0.571** / 1.0 / 1.0; 0 / **1.0** | 1.0 / 0.842 / 1.0 / 1.0; 0 / 0 | 2 of 3 |
| 0 (re-run here, seed 1 only) | - | 0.878 / **0.593** / 1.0 / 1.0; 0 / **0.875** | - | fails |
| **0.25** | 1.0 / 1.0 / 1.0 / 1.0; 0 / 0 | 1.0 / 1.0 / 1.0 / 1.0; 0 / 0 | 1.0 / 1.0 / 1.0 / 1.0; 0 / 0 | 3 of 3 |
| 1.0 | 0.973 / 1.0 / 1.0 / 1.0; 0 / 0 | 1.0 / 1.0 / 1.0 / 1.0; 0 / 0 | 1.0 / 1.0 / 1.0 / 1.0; 0 / 0 | 3 of 3 |

Mean cross-pick over seeds and directions: k = 0: 0.167 (recorded), k = 0.25: 0.0, k = 1.0: 0.0.
The term also removes the small class's false positives (at k = 0 they sit on the larger solid particles).

**Recommended k = 0.25** (`K_RECOMMENDED` in the test): the smallest value tried that fixes seed 1. It changes
the loss least, and k = 1.0 brings nothing more here. The sweep ran 8 trainings: k = 0.25 and k = 1.0 on seeds
0-2, k = 0 on seed 1, and one k = 1.0 run lost to a bug in the test (fixed: the k check now spies on the loss
calls during training, because `model.hparams.config` is the global `mainConfig`, which the test restores
after training). The sweep used `TOMOCPT_G3B_SKIP_K0=1`. The default test also retrains k = 0 on seeds 0-2, and
that full six-run form was not run end to end (about 27 min). Expected outcome: it passes unless k = 0 seed 1
also passes by chance, in which case the "lower on average" assertion fails because both means are 0.
Only 3 seeds per k were run, so the 9-of-10 rule is not demonstrated. The margin is wide: cross-pick 0.0
against a 0.15 threshold, minimum F1 0.973 against 0.8.

**Suggested k for CZII: start at 0.25.** CZII has several similar-looking species at similar sizes
(beta-amylase, beta-galactosidase, thyroglobulin; ribosome, VLP). Raise k to 0.5-1.0 if channels still fire on each
other's particles. Larger k pushes harder against a class wherever any other annotated class sits. Labels in
CZII are not perfectly exhaustive, so a high k also punishes correct picks near mislabelled or overlapping
particles. Watch per-class recall when you raise it.

## Results on HEAD (2b62832, `v2-gate-4-passed` + plan)

Fast tier: **1 passed, 31 failed**. The pass is `test_G3b_C4_v1_training_step_ignores_k` (a lock: v1 must stay
untouched). The failures are all for the missing feature: `multiclass_picking_loss() got an unexpected keyword
argument 'cross_class_neg_weight'`, the missing keyword in the signature (C1), or the missing
`ParticlesConfig.cross_class_neg_weight` (C4).
Full fast suite on HEAD: **493 passed, 27 skipped, 31 failed** (all 31 in `tests/gate3b`).

## Results on the scratch reference implementation

A throwaway implementation of the spec (session scratchpad, run with `pytest -o pythonpath=<ref>`) passes all
32 fast tests; full fast suite **525 passed, 27 skipped**.

Mutations of the reference, each caught:

| Mutation | Failing tests |
|---|---|
| Extra term also applied to positives (`w = where(y>0, w_pos, 2 w_neg) + k w_pos o`) | C2 formula (k > 0, 4), C2 mask test, C2 reference (2), C3 (2) |
| Unavailable classes counted in `o` | C2 formula (4), C2 mask test, C2 unavailable target (2), C2 reference (2), C3 (2) |
| Soft target ignored (`o` binary) | C2 formula (4), C2 reference (2), C3 (2) |
| `training_step` does not pass k | C4 `training_step[1.5]`, C4 checkpoint `[1.5]` |

## Spec ambiguities

1. "rtol 1e-6" for gradients: element-wise rtol alone is ill-posed for near-zero gradient entries; an absolute
   floor of 1e-6·max|grad| is added.
2. C5 "cross-pick rate lower on average": read as the mean over seeds 0-2 of both cross directions.
3. Whether `val_loss` should use k is not specified; not tested.
