# Gate 7 tests (discriminative training: contrastive + focal)

Written by the orchestrator acting as Test Author (handoff `HANDOFF_gate7.md`; usage-limited, no
agent fan-out) from `.claude/agents/07_AGENT_DISCRIMINATIVE.md`, before the implementation.

- Fast tier: `pytest tests/gate7 -m "not slow"` (CPU, fp32, seeded). On the development machine:
  about 20 s on Gate 6 code (most tests fail fast there).
- Slow tier: `pytest tests/gate7 -m slow` (G7-C9, G7-C10; CUDA; six tiny-model trainings:
  three arms x seeds 0, 1).

New public names are imported inside the test functions, so collection works before the
implementation exists.

## Rulings and decisions recorded here

1. **Human decision (1 Nov 2026), anchor threshold.** The spec default `anchor_min_target = 0.99`
   gives almost no anchors: an off-grid centre's peak voxel is 0.54 / 0.80 / 0.92 for 7 / 12 / 20
   voxel particles, and a 7-voxel tube off the voxel grid peaks at 0.65. Anchors (and the focal
   loss's positives) are the **peak voxels** (local maxima) of the target with value
   `>= anchor_min_target`, default **0.5**. `test_G7_C1_config_defaults` checks 0.5.
2. **Orchestrator ruling, call signatures** (the spec names the functions only; full text in
   `g7_helpers.py`):
   - `sample_anchors(target, available, d_vox, is_filament, min_target=0.5, max_per_class=32, generator=None) -> (index [N,4] (b,z,y,x) int64, label [N] int64)`
   - `sample_hard_negatives(prob, target, d_vox, n, min_score=0.3, max_score=0.9, exclusion_radius=1.0, negative_mask=None, n_mask=0, generator=None) -> (index [M,4], score [M], from_mask [M] bool)`; score = max class probability; hard negatives first, ranked descending
   - `ContrastiveMemoryBank(n_classes, size, dim)`: `push(embeddings, labels)` (label -1 = negative queue), `get() -> (embeddings, labels)`
   - `supcon_loss(embeddings, labels, bank_embeddings=None, bank_labels=None, temperature=0.1)`; label -1 never an anchor; L2-normalised inside; Khosla L_out (mean of log-probabilities over positives; denominator = all other batch and bank entries); anchors without a positive are skipped
   - `focal_heatmap_loss(pred, target, available, positive, num_instances, alpha=2.0, beta=4.0)`; positives given as a bool mask (the peak voxels); per pair divided by `max(num_instances, 1)`; mean over available pairs
   - the head is `BasePickingModel.contrastive_head` (a `DenseProjectionHead`), `None` when off;
     `forward_features(x, apix=None) -> (logits, hidden4, decoder2)` on `MySwinUNETR` and
     `ScaleConditionedSwinUNETR`.
3. **Orchestrator ruling, tolerances.** Filament anchors within 0.5 voxel of the centreline when it
   runs through voxel centres (spec); 0.75 for an oblique off-grid centreline (the nearest voxel
   centre can be sqrt(2)/2 away). Hard negatives: distance to every labelled centre
   `>= (1 + exclusion_radius) R - 1` voxel (the target support has radius R; one voxel slack for a
   discrete dilation).
4. **Orchestrator ruling, data path of negative masks.** `prepData.negative_masks` must reach the
   batch as `negative_mask [B, 1, 64, 64, 64]` bool, aligned with `input_data`; no key without
   masks (Gate 6 batches unchanged). The spec allows the write paths but has no criterion for it.
5. **Orchestrator ruling, G7-C10 peak shape** (`HANDOFF_gate7.md` 2.2): the focal value term must
   not lower particle F1 at threshold 0.3 without cross-class suppression by more than 0.03 vs the
   Huber term (cross-entropy with a background logit flattened the peaks in the Gate 6 ablation).
6. **Handoff ruling: softmax head in the slow tests** (k = 0); v2 picking defaults (peak spacing
   0.5 x d, no cross-class suppression).
7. G7-C7 "bit-identical to Gate 6": logits (SHA-256 of bytes) and first-step loss (float hex) from
   a golden made with the code at `v2-gate-6-passed`; gradients compared by their absolute sum,
   rtol 1e-5, because CPU backward is not bit-reproducible between runs of the same code.

8. **Human decision (1 Nov 2026), reconciled focal value term.** The spec's CenterNet focal loss
   (normalised by instance count) has minimiser 0 at every non-peak voxel and collapsed to a flat
   output in the slow tests. `value_loss = "focal"` is now Huber's weighted term (Section 1.5
   weights, voxel mean) with the per-voxel penalty `|y - p|^focal_alpha * BCE(p, y)` (minimiser
   `p = y`) on **particle** channels; filament channels keep the Huber penalty. `focal_beta` is
   unused. `focal_heatmap_loss` (CenterNet) stays in the module, tested by C6, unused by training.
   Huber stays the config default (invariant 1); recommending focal as the default is a doc change.
   C6b tests the new term; C10 adds `test_G7_C10_focal_picks_particles_as_well_as_huber`.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse), `slow` marker |
| `g7_helpers.py` | Signature rulings, renderers, configs, sheet phantom helper |
| `test_g7_c1_lock.py` | C1, C7 (both off vs Gate 6 golden), `forward_features` |
| `test_g7_sampling.py` | C2, C3 (functions) |
| `test_g7_negative_masks.py` | C3 (data path, ruling 4) |
| `test_g7_losses.py` | C4, C5, C6 |
| `test_g7_model.py` | C7 (options on), C8 |
| `test_g7_learning.py` | C9, C10 (slow) |
| `../golden/gate7_common.py`, `make_goldens_gate7.py`, `gate7_lock.json` | Lock golden (Gate 6 code) |

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| G7-C1 Defaults, no head | `test_G7_C1_config_defaults`, `test_G7_C1_defaults_create_no_head_and_no_new_state[*]` (state-dict keys = Gate 6) |
| G7-C2 `sample_anchors` | `test_G7_C2_particle_anchors_are_centres`, `..._follow_a_flip`, `test_G7_C2_min_target_is_applied`, `test_G7_C2_unavailable_pairs_give_no_anchors`, `test_G7_C2_per_class_cap_per_batch`, `test_G7_C2_filament_anchors_on_centreline_spaced[axis,diagonal,offgrid]` |
| G7-C3 `sample_hard_negatives` | `test_G7_C3_hard_negatives_exclusion_range_ranking[0.5,1.0]`, `test_G7_C3_scores_outside_the_range_are_never_selected`, `test_G7_C3_mask_negatives_come_only_from_the_mask`, `test_G7_C3_negative_mask_reaches_the_batch_aligned`, `test_G7_C3_no_negative_masks_no_batch_key` |
| G7-C4 `ContrastiveMemoryBank` | `test_G7_C4_bank_fifo_detached_capped_buffer`, `test_G7_C4_negative_queue_is_capped` |
| G7-C5 `supcon_loss` | `test_G7_C5_reference_toy_case`, `test_G7_C5_separated_classes_give_lower_loss_than_shuffled_labels`, `test_G7_C5_finite_under_bf16`, `test_G7_C5_no_anchor_gives_zero_with_gradient_path` |
| G7-C6 `focal_heatmap_loss`, Huber = Gate 6 | `test_G7_C6_focal_equals_centernet_formula[*]`, `test_G7_C6_huber_value_term_equals_gate6` |
| G7-C7 One step, options on / off | `test_G7_C7_options_on_every_parameter_gets_a_gradient[*]`, `test_G7_C7_batch_without_anchors_still_gives_the_head_a_gradient`, `test_G7_C7_both_off_is_bit_identical_to_gate6[*]`, `test_G7_C7_explicit_off_values_equal_defaults[*]` |
| G7-C8 SSL v2 transfer | `test_G7_C8_ssl_dense_head_initialises_the_contrastive_head[True,False]` |
| G7-C9 (slow) look-alikes | `test_G7_C9_contrastive_separates_lookalikes`, `test_G7_C9_contrastive_reduces_tube_picks_on_sheets` |
| G7-C6b reconciled focal (ruling 8) | `test_G7_C6b_soft_focal_minimiser_is_the_target`, `test_G7_C6b_focal_value_term_is_huber_weights_times_soft_focal[*]`, `test_G7_C6b_model_uses_soft_focal_on_particles_and_huber_on_filaments` |
| G7-C10 (slow, rulings 5, 8) peak shape | `test_G7_C10_focal_keeps_peak_shape`, `test_G7_C10_focal_picks_particles_as_well_as_huber` |
| invariant 7 | `test_G7_forward_features_matches_forward[False,True]` |

## Self-checks before hand-over

- On Gate 6 code (HEAD before implementation): 5 passed (C1 no-head x2, C7 bit-identical x2, no
  `negative_mask` key), 36 failed (missing module / config / `forward_features` / batch key).
- Scratch reference of the five functions (not committed): C2-C6 function tests 20/20 pass.
  Mutations caught: no exclusion zone (3 fail), unranked negatives (2), threshold 0.99 (5),
  filament spacing ignored (3), bank left out of the SupCon denominator (1), focal not normalised
  by the instance count (2).
- C9/C10 calibration of the `off` arm: see the section below.

## Calibration of the slow tier (1 Nov 2026)

Settings in `test_g7_learning.py`: SNR 0.25, hollow wall 0.4, 400 steps (env overrides
`TOMOCPT_G7_SNR`, `TOMOCPT_G7_WALL`, `TOMOCPT_G7_STEPS`).

Off arm, seed 0 (SNR 0.25, wall 0.4) at three step counts:

| Steps | Look-alike F1 | On-sheet FP | Particle F1 @0.3, no NMS |
|---|---|---|---|
| 1500 | 1.00 | 677 | 0.79 |
| 800 | 0.99 | 332 | 0.28 |
| 400 | 0.88 | 195 | 0.02 |

At 1500 steps the baseline already solves the look-alike pair, so C9's +0.05 cannot show. 400 steps
is the working point. Final slow run (400 steps, seeds 0 and 1): 3 passed, 421 s.

| Arm | Seed 0 look-alike F1 | Seed 1 look-alike F1 | Seed 0/1 on-sheet FP |
|---|---|---|---|
| off | 0.88 | 0.86 | 195 / 1044 |
| contrastive | 1.00 | 1.00 | 224 / 616 |
| focal | 0.00 | 0.00 | 0 / 207 |

**Open issues (not resolved here):**
- **C10 is vacuous at this budget.** The off (Huber) arm's particle F1 @0.3 without NMS is 0.02,
  so `focal >= off - 0.03` passes whenever focal is ≥ about 0. The focal arm itself detects nothing
  (0 TP, 0 FP, 20 FN per class on seed 0; the same on seed 1 for particles). The check does not
  guard the peak shape it was meant to guard.
- The focal arm's collapse is unexplained; the formula matches CenterNet (checked in C6). The
  likely cause is the background penalty pushing all probabilities below threshold before the
  peaks rise at this step count, but this is not verified.
- The C9 sheet criterion passes on the summed count only because seed 1's contrastive arm is
  less than 0.7 x off (616 vs 1044); seed 0 goes the other way (224 vs 195). Treat it as weak.

## Focal arm investigation (1 Nov 2026, scratch runs, no repo change)

- The focal formula is correct, and it is not a scale bug: dividing the value term by the voxel
  count (scratch patch) still gives 0 peaks at 400 steps.
- Cause: the focal positives get weight 1 per peak voxel (divided by the instance count), while
  the background is about 262k voxels; Huber gives each positive voxel about N/S (about 800 here).
  At 400 steps the focal model is still in its near-zero state: probabilities at true centres
  0.08 to 0.14, channel maxima 0.16 to 0.29.
- At 1500 steps the focal arm learns peaks (channel maxima 0.72 to 0.92; centre probabilities
  0.24 to 0.84). The look-alike baseline at 1500 steps is solved perfectly (F1 1.00), so C9 and
  C10 need different step budgets with this phantom.
- Decision needed: a longer step budget for the focal arm (with the same budget for all arms, C9
  becomes void), or a loss-balance change for the focal positives (outside the spec's formula).

## Focal positive rebalancing (1 Nov 2026, scratch runs, no repo change)

Positive term multiplied by a per-pair weight, value term divided by N (voxel mean):

| Positive weight | Seed 0 look-alike F1 / F1 @0.3 no NMS | Seed 1 look-alike F1 / F1 @0.3 no NMS | Actin FP on sheets (seed 0 / 1) |
|---|---|---|---|
| 1 (CenterNet, spec) | 0.00 / 0.00 | 0.00 / 0.00 | 0 / 207 (not rebalanced) |
| Huber `N/(S+1)` | 0.00 / 0.38 | 0.75 / 0.25 | 77k / 125k (overshoot, fires everywhere) |
| `sqrt(N/(S+1))` | 0.00 / 0.00 | 0.00 / 0.10 | 0 / 97k |

Neither weighting gives a usable focal arm at 400 steps: the full weight over-activates the whole
volume, the square root stays in the flat state. The focal value term is not validated as a
replacement for Huber under this budget. A working setting needs either a longer budget (1500 steps
learned peaks with the spec formula) or a different balance, and each choice changes C9/C10.

## Reconciled focal term (1 Nov 2026, scratch runs, no repo change)

Soft-target ("quality") focal penalty `q(p, y) = |y - p|^2 * BCE(p, y)` (Li et al. 2020): it is
>= 0 and 0 only at `p = y` (checked numerically for y in {0, 0.1, 0.37, 0.5, 0.8, 1}), so its
minimiser is Huber's. Alone (instance normalisation) it still collapses at 400 steps: class balance,
not the fixed point, caused the collapse. With Huber's weights and normalisation,
`mean_voxels(w * q)`, `w = where(y > 0, N/(S+1), 2 N/(N-S+1))`, it trains:

| Arm (400 steps) | Look-alike F1 s0 / s1 | F1 @0.3 no NMS s0 / s1 | Actin F1 s0 / s1 | Actin FP on sheets s0 / s1 |
|---|---|---|---|---|
| off (Huber) | 0.88 / 0.86 | 0.025 / 0.020 | 0.74 / 0.68 | 195 / 1044 |
| weighted soft focal | 0.82 / 1.00 | 0.041 / 0.028 | 0.61 / 0.26 | 1475 / 9811 |

Particles: all centres found on both seeds, look-alike mean F1 0.91 vs 0.87. Actin: worse
(over-activation on the actin channel, more tube picks on sheets).

Final slow run with the reconciled term (400 steps): 4 passed, 414 s. Focal arm vs off, seeds 0 / 1:
look-alike F1 0.87 / 0.92 vs 0.88 / 0.86; actin F1 0.80 / 0.89 vs 0.74 / 0.68; actin on-sheet FP
96 / 390 vs 195 / 1044.
