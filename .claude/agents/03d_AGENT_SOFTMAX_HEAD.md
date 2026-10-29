# 03d_AGENT_SOFTMAX_HEAD.md — Gate 3d: softmax class head with implicit background

Added 29 Oct 2026 by human decision ("Implement softmax"), so that classes compete per voxel
without tuning `train.particles.cross_class_neg_weight` (Gate 3b). Evidence:
`experiments/czii_v2_results.md` (k = 0.25 removed double picks but made the small classes hedge).

## Design
- The network output is unchanged: `C` logits per voxel, channel `c` = class-table row `c`
  (invariant 7). No new parameters, so every checkpoint loads and DDP is unaffected.
- **Implicit background.** A background logit fixed at 0 is appended, and a softmax is taken
  over the background and the classes that are *available*:
  `p_c = exp(l_c) / (1 + Σ_{d available} exp(l_d))` for available `c`, `p_c = 0` otherwise.
  With one class this is exactly `sigmoid(l_0)`. Compute in float32 with a log-sum-exp
  (append a zero channel, `-inf` for unavailable channels, softmax, drop the background).
- **Training.** In softmax mode the pipeline-v2 loss (Section 5.4, `multiclass_picking_loss`)
  is applied unchanged to these probabilities, with `class_available` as the availability.
  Where the soft targets of a voxel sum to more than 1 (touching particles), they are divided
  by their sum for the loss (`y / max(1, Σ_c y_c)`), because softmax probabilities cannot
  exceed it. `cross_class_neg_weight` still works but is not needed (default 0).
- **Inference.** In softmax mode the probabilities at a level are normalised over the classes
  predicted at that level (`classes` of `_predict_level`). Sigmoid mode is unchanged.
- Pipeline v1 (one class) is unaffected: softmax with one class equals sigmoid.
- Distillation keeps sigmoid (teacher BCE on logits); softmax mode is not supported there.

## Config — `defaultConfigs/particles_config.py`
`ParticlesConfig.class_head: str = "sigmoid"` (`sigmoid` | `softmax`). Read from the
checkpoint config at inference (invariant 3); a checkpoint without the field is `sigmoid`
(`safe_getattr`). Any other value raises `ValueError` at model construction.

## Allowed write paths
- `tomocpt/training/multiclass_loss.py`
- `tomocpt/networks/pickingModel.py`
- `tomocpt/defaultConfigs/particles_config.py`
- `tomocpt/predict/multiclass.py` (passing the level's classes to the model only)
- `.claude/reports/gate_03d_report.json`

## Public names
`tomocpt.training.multiclass_loss.class_probabilities(logits, class_head, available=None)`:
`logits [B, C, D, H, W]`; `available` `[B, C]` bool, or a sequence of channel indices for all
samples, or `None` (all channels). Returns float32 `[B, C, D, H, W]`.
`BasePickingModel.predict_step(batch, batch_idx=0, dataloader_idx=0, apix=None, classes=None)`.

## Pass criteria
- **G3d-C1** Default `sigmoid`: every earlier test passes unchanged (legacy equivalence and
  Gate 3/3b/4 goldens).
- **G3d-C2** `class_probabilities(..., "softmax")` with C = 1 equals `sigmoid` (atol 1e-6);
  with C > 1, per voxel `Σ_c p_c < 1`, each `p_c` in (0, 1), and unavailable channels are 0;
  finite for logits of ±80 and under bf16 inputs.
- **G3d-C3** Changing the logit of an unavailable channel does not change any available
  channel's probability.
- **G3d-C4** Softmax loss with C = 1, all available, equals the sigmoid loss (rtol 1e-6);
  targets that sum to more than 1 give a finite loss equal to the loss on renormalised targets.
- **G3d-C5** One softmax training step (v2 batch, some classes unavailable in the batch):
  finite loss, every trainable parameter gets a gradient.
- **G3d-C6** `class_head` is stored in the checkpoint; `predict_step` of a softmax model returns
  `class_probabilities` (over `classes` when given); a checkpoint without the field predicts
  with sigmoid; an invalid value raises `ValueError`.
- **G3d-C7** (slow) The Gate 3 C13 synthetic scenario (solid vs hollow spheres of the same
  size), softmax with `cross_class_neg_weight = 0`, seeds 0 and 1 (seed 1 fails with sigmoid
  at k = 0): per-class F1 ≥ 0.9 and mean cross-pick rate ≤ 0.05.
