# 03b_AGENT_CROSS_CLASS_NEG.md — Gate 3b: cross-class negative weighting

Added 27 Oct 2026 by human decision, after CZII v2 training showed channels of similar
species (beta-amylase / beta-galactosidase / thyroglobulin; ribosome / VLP) firing at the
same positions.

## Problem
In the Section 5.4 loss each channel is weighted as in v1: positives get
`w_pos = N/(S+1)` (hundreds), every other voxel `2·w_neg ≈ 2`. A voxel inside a particle of
another class is an ordinary negative for channel c, so firing on a similar-looking particle
of a different species costs almost nothing. Gate 3 C13 (solid vs hollow) fails on some seeds
for this reason.

## Allowed write paths
- `tomocpt/training/multiclass_loss.py`
- `tomocpt/networks/pickingModel.py` (passing the new value to the loss only)
- `tomocpt/defaultConfigs/particles_config.py`
- `.claude/reports/gate_03b_report.json`

## Config (this gate adds)
`ParticlesConfig.cross_class_neg_weight: float = 0.0` (`train.particles.cross_class_neg_weight`).
`0.0` = off: the loss is bit-identical to Gate 3.

## Task
`multiclass_picking_loss(..., cross_class_neg_weight: float = 0.0)`. For sample b and an
available class c, let
`o[b,c] = max over classes d != c with class_available[b,d] of y_true[b,d]` (0 if none).
The per-voxel weight of the Huber term becomes
```
w = where(y_true[b,c] > 0, w_pos[b,c], 2·w_neg[b,c] + k · w_pos[b,c] · o[b,c])
```
with `k = cross_class_neg_weight`. So a voxel at the centre of another class's particle costs
`k` times as much as a missed voxel of class c; the extra weight falls off with the other
class's soft target. Voxels where `y_true[b,c] > 0` and pure background are unchanged.
Unavailable classes never contribute to `o` (their targets are unknown). Unavailable pairs
stay masked exactly as in Gate 3 (zero gradient). The Hessian term is unchanged.
`w_pos`, `w_neg`, class weights and the mean over available pairs are as in Section 5.4.
The checkpointed per-class form and the vectorised form must agree.
`BasePickingModel` passes `config.train.particles.cross_class_neg_weight`
(via `safe_getattr`, default 0.0) on v2 batches. v1 is untouched.

## Pass criteria
- **G3b-C1** `k = 0` (and the default): loss and gradients equal the Gate 3 loss
  (`v2-gate-4-passed` code) on random C = 1 and C = 4 batches with partial availability (rtol 1e-6).
- **G3b-C2** Weights: with `k > 0`, a voxel inside class d's sphere and outside class c's gets
  the formula weight in channel c (checked through gradients or a reference implementation);
  voxels with `y_true[c] > 0` and background voxels are unchanged; an unavailable class d does
  not raise channel c's weight; unavailable pairs still get exactly zero gradient.
- **G3b-C3** Checkpointed per-class form equals the vectorised form with `k > 0` (rtol 1e-5).
- **G3b-C4** `training_step` of a v2 model uses `train.particles.cross_class_neg_weight`;
  a checkpoint config without the field loads and gives `k = 0`.
- **G3b-C5** (slow) The Gate 3 C13 scenario (solid vs hollow, same size), trained with the
  recommended `k` and with `k = 0` on the same seeds: the cross-pick rate with `k > 0` is lower
  than with `k = 0` on average, and the C13 thresholds hold on the seed where `k = 0` failed
  (seed 1) and on at least 2 of 3 seeds. The TA calibrates the recommended `k` and records it.

## Notes for the user
Recommend a starting value of `k` for CZII in the report.
