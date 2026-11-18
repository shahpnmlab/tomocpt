# 09b_AGENT_EMBED_HEAD.md — Gate 9b: make the class head use the contrastive embedding (prototype)

Added 18 Nov 2026 by human request (prototype on `v2-feat/embed-head`, based on `v2-feat/boss-recipe`).

## Why
Diagnostics on the DS-10490 `all` checkpoint (C = 134, softmax head with implicit background,
SupCon head on the `decoder2` features, `embed_dim` 64): at true particle centres a linear
classifier on the contrastive embeddings reaches 0.42 balanced accuracy across tomograms, the
softmax argmax 0.35. The embedding encodes identity that the class head does not use. Two opt-in
options tie the two together; both are off by default.

## Fields (`train.discriminative`, `defaultConfigs/discriminative_config.py`; not CLI flags)

| Path | Type / default | Meaning |
|---|---|---|
| `train.discriminative.embed_ce_weight` | float = `0.0` (off) | Option 3: weight of the cosine-softmax CE on the embedding at the anchor voxels |
| `train.discriminative.prototype_head` | bool = `False` | Option 2: particle logits get the residual `α · s · (cos_k − cos_bg)` from the embedding |
| `train.discriminative.prototype_scale` | float = `10.0` | `s` of both options (fixed; 10 = 1 / the default SupCon temperature) |

## Modules
- The projection head (`contrastive_head`, `DenseProjectionHead(fs, embed_dim)` on `decoder2`) is
  created when `contrastive_weight > 0` **or** `embed_ce_weight > 0` **or** `prototype_head`. The
  memory bank exists only when `contrastive_weight > 0` (as Gate 7). All three need pipeline v2.
- `tomocpt.training.discriminative.PrototypeClassifier(n_particles, dim, scale=10.0, residual=False)`
  (attribute `BasePickingModel.prototype_classifier`, `None` when both options are off):
  `prototypes [P + 1, dim]` (row `k < P` = the k-th particle channel of the class table, row `P` =
  background), random normal init; `alpha [1]` zero-initialised, only with `residual=True`
  (`prototype_head`). One prototype set is shared when both options are on.
  `cosine(emb [N, E]) -> [N, P + 1]`, `cosine_map(emb_map [B, E, ...]) -> [B, P + 1, ...]`
  (normalised embedding through a 1×1×1 convolution with the normalised prototypes).
- Filament channels have no prototype in either option.

## Option 3 — embedding CE (`embed_ce_weight > 0`)
- Anchors: the same `sample_anchors` / `sample_hard_negatives` call as SupCon (one call when both
  are on, so the RNG stream and therefore the SupCon term are unchanged). Particle anchors keep
  their class (row of their channel); hard and mask negatives are background (row `P`); filament
  anchors are dropped.
- Logits `s · cos(e, p_k)` on the L2-normalised projection embedding at the anchor voxel. Masking
  as the softmax head: a particle row whose channel is not available in the anchor's sample
  (`class_available[b, c] = False`; this encodes the level's class subset in training) is `-inf`;
  background is always allowed. `softmax_scope` does not matter (filaments are never in it).
- Loss `prototype_ce_loss(cos, rows, allowed, scale)`: mean CE over the anchors, float32.
  No anchor: `0 · Σ prototypes` (DDP: every prototype has a gradient). The embedding map is in
  the graph every step (`0 · Σ emb_map`).
- Weight schedule: `embed_ce_weight_at(step) = embed_ce_weight · min(1, step / contrastive_warmup_steps)`
  (constant when `contrastive_warmup_steps = 0`). It follows the contrastive ramp because it acts
  on the same, still random, embedding; one knob keeps the two auxiliary terms in proportion.
- Total loss = picking loss (+ SupCon) + `embed_ce_weight_at(step) · CE`; logged as `loss_embed_ce`.

## Option 2 — prototype residual (`prototype_head = True`)
- In every model forward (`forward`, `_forward_features`, `_forward_all`, so `predict_step`,
  validation and `predict_step_with_offsets` use it): with `e` the projection of `decoder2`
  (32³ for a 64³ crop), particle channel logits get
  `r_k = α · s · (ê · p̂_k − ê · p̂_bg)`, computed at 32³ (float32, autocast off), scattered to the
  particle channels (0 on filament channels) and trilinearly upsampled to the logit grid
  (`align_corners=False`). Because the 1×1×1 convolution and trilinear interpolation are both
  linear, this equals `s · up(ê) · p̂_k` (the normalised embedding upsampled, not renormalised),
  at the memory of `P` channels at 32³ instead of `E` channels at 64³.
- Background: the implicit background logit stays 0. Subtracting `cos_bg` from every particle
  logit is the same, for the particle softmax, as adding `α s cos_bg` to the background logit.
- Output contract unchanged: `forward` returns logits `[B, C, D, H, W]`; the network's
  `(logits, hidden4)` tuple is untouched (the residual lives in `BasePickingModel`). The logits are
  float32 when the residual is on.
- `α = 0`: the residual is exactly 0, the logits equal the unconditioned head's (invariant 6).

## Checkpoints and fine-tuning
- Defaults: no new module, `state_dict` keys unchanged; old checkpoints load with `strict=True`.
- `fine_tune_from` a v2 picker (`training/helpers._create_model`, flagged as outside Gate 7's
  SSL-only scope): the source's `contrastive_head.*` is copied when the shapes match, and
  `contrastive_bank.*` / `prototype_classifier.*` when, in addition, the class tables are equal
  (same class numbers in the same order). Anything missing starts fresh (prototypes random, α 0).
  Before this gate a fine-tune from a contrastive picker restarted the projection head.
- New parameters are not in `self.model`, so `_get_model_parameters_for_optimizer` puts them in
  the ×1.0 group when fine-tuning (one group from scratch).

## Allowed write paths
`tomocpt/defaultConfigs/discriminative_config.py` (own fields), `tomocpt/networks/pickingModel.py`,
`tomocpt/training/discriminative.py`, `tomocpt/training/helpers.py` (`_create_model`: head
transfer from a picker source), `tests/gate9b/**`, `experiments/run_embed_head_ablation.sh`,
`CLAUDE.md` (Sections 6, 7), `README.md`, `TODO_v2.md`.

## Pass criteria (`tests/gate9b/`, CPU)
- **G9b-C1** Defaults: the three fields; no prototype module; contrastive-only model has no
  `prototype_classifier` and identical logits and first-step loss to a model built without the
  new code path (Gate 7 lock stays green).
- **G9b-C2** `prototype_ce_loss` equals a hand-computed float64 reference on a tiny case, with a
  masked class; no anchors gives 0 with a gradient path.
- **G9b-C3** One training step with option 3, option 2 and both (softmax, particles + filament,
  `softmax_scope` all and particles): loss finite and different from contrastive-only; every
  trainable parameter has a gradient; prototypes and backbone get non-zero gradients (option 3);
  every trainable parameter is in the optimizer. A batch without targets still gives every
  parameter a gradient.
- **G9b-C4** Option 2 with `α = 0` gives logits equal to the same weights without the residual;
  with `α ≠ 0` filament channels are unchanged and particle channels change; `softmax_scope =
  particles` keeps filament probabilities equal to their sigmoids.
- **G9b-C5** Level masking: an unavailable class gets probability 0 in the embedding CE, and
  changing its prototype does not change the loss.
- **G9b-C6** Old checkpoint (contrastive only, written by this code with defaults): loads for
  `InferenceModel.load_from_checkpoint(strict=True)`; `fine_tune_from` with both options builds
  the model, copies the projection head and bank, initialises the prototypes and α = 0, puts the
  new parameters in the ×1.0 group, and gives the source's logits.
- **G9b-C7** Config backfill (Gate 1 sweep).
