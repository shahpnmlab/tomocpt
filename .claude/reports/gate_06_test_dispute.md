# Gate 6 test dispute: G6-C8 / G6-C9 class head (1 Nov 2026)

## What
`tests/gate6/test_g6_learning.py::train_picker` builds its config with
`g6_helpers.v2_config_with_filaments(tmp, particles, filaments)`, i.e. the default
`class_head = "sigmoid"` with `cross_class_neg_weight = 0`. C8/C9 fail with this setup
(attempt 1: joint particle F1 0.667, 32/16/16; particle-only 1.00).

## Evidence
- Diagnosis (31 Oct, seeds 0 and 1): the loss is class confusion between the two sphere classes,
  not filament interference. 16/16 200 A spheres are picked as 120 A; the 120 A channel reads
  0.87/0.95 at 200 A centres vs 0.83/0.87 for the 200 A channel. Misses lie a median 27-29 voxels
  from any tube.
- Ablation (1 Nov, scratch script reusing the frozen slow-test functions, joint model, 2000 steps):

  | class head | k | seed | particle F1 (tp/fp/fn) | mt F1 | actin F1 |
  |---|---|---|---|---|---|
  | sigmoid | 0 | 0 | 0.667 (32/16/16) | 1.00 | 0.91 |
  | softmax | 0 | 0 | 1.00 (48/0/0) | 1.00 | 0.92 |
  | softmax | 0 | 1 | 1.00 (48/0/0) | 1.00 | 0.98 |
  | softmax | 0.25 | 0 | 1.00 (48/0/0) | 1.00 | 0.90 |
  | softmax | 0.25 | 1 | 1.00 (48/0/0) | 1.00 | 0.97 |

  Extra cross-entropy terms (core CE, object-centre CE) classify perfectly but flatten the
  heatmaps (particle F1 0.98 / 0.61, actin precision down to 0.58): not adopted.

## Request
C8/C9 train both models (joint and particle-only) with `class_head = "softmax"`
(`cross_class_neg_weight = 0`): a multi-class model of mutually exclusive particle classes is
meant to use the softmax head (Gate 3d), and softmax with k = 0 is sufficient (CZII Gate 3b note:
k = 0.25 gave no gain). Thresholds unchanged (F1 >= 0.8, interference <= 0.03).

## Ruling
Human decision (1 Nov 2026): "we need to use softmax and cross_class_neg because we are doing a
multiclass prediction. To use a sigmoid is an amateur mistake." Then "close gate 6".
Orchestrator: softmax with k = 0 (the k = 0.25 arm adds nothing here; k stays a CZII-tuned option).
`train_picker` passes `class_head="softmax"`; spec G6-C8 amended; test re-frozen (new hash in
`gate_status.json`).

## Addendum (1 Nov 2026): v2 picking defaults
Human decision: make the CZII-measured settings the v2 defaults (`infer.peak_spacing_fraction = 0.5`,
`infer.cross_class_nms = False`). `tests/golden/gate6_common.py::predict_arrays` called `predict` with
the default spacing, so G6-C1 (`pred:net:big:coords` 159 vs 15 rows) failed although the golden locks
Gate 4 behaviour. The helper now passes `peak_spacing_fraction=1.0`, which is exactly the Gate 4
spacing (`1.0 x d == d`); `cross_class_nms` was already explicit. Golden data unchanged; re-frozen.
