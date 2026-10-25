# 06_AGENT_SCALE_MOE.md — Gate 6: scale-expert soft MoE (experiment with a decision)

## Mission
Test whether routing tokens between experts with different receptive fields, using
the target size as a router input, improves scale-free picking over the Gate 3 model.
Build it correctly, compare it fairly with two controls, and report. It ships
(`moe.enabled` default true for v2) only if it wins.

## Why this design (and not shape-named experts)
- After pipeline v2 the useful question is "which receptive field fits this target
  size here?". Depthwise convolutions with dilation 1, 2, 3 give receptive fields of 3, 5
  and 7 tokens. A plain MLP expert gives 1.
- Experts are not named after shapes (membrane, filament). Nothing supervises such
  names, and the earlier review found no basis for them.
- Depthwise convolutions need every spatial neighbour, so each expert runs on the
  whole token grid. This is a **soft (dense) MoE**: compute does not drop. The
  benefit, if any, is conditional capacity, not speed.

## Allowed write paths
- `tomocpt/networks/moe.py` (new)
- `tomocpt/networks/distillationModel.py` (auxiliary loss line only)
- `tomocpt/defaultConfigs/moe_config.py`
- Path exception: `tomocpt/defaultConfigs/network_config.py`, one call
  `maybe_inject_moe(model, self.moe)` at the end of `NetworkConfig.build_model`.
- Path exception: `tomocpt/networks/pickingModel.py`, add the MoE auxiliary loss in
  `training_step` (and in `DistillationPickingModel.training_step`) only.
- `benchmarks/moe_ablation.py` (new)
- `.claude/reports/gate_06_report.json`

## Design
Stage geometry (CLAUDE.md Section 1.4): default stages `layers2` (16³, 2fs) and
`layers3` (8³, 4fs). `layers4` (4³) is too small for dilated kernels.

`ScaleExpertMoE(dim, mlp_dim, experts, cond_dim, temperature)` replaces
`SwinTransformerBlock.mlp` in the chosen stages:
- Each expert: `linear1 → (+ branch_scale · depthwise_conv(h)) → GELU → drop →
  linear2 → drop`, with the same shapes as MONAI `MLPBlock`. Expert `mlp` has no
  conv branch. `dw_dK`: `Conv3d(mlp_dim, mlp_dim, 3, padding=K, dilation=K,
  groups=mlp_dim)`. `branch_scale` is a scalar parameter, initialised to 0.
- Router: `logits = W_x · x + W_c · e(d_vox)`, where `x` is the block input
  (already `norm2`-ed by MONAI) and `e` is the Gate 3 size embedding. `W_c` is
  zero-initialised. `W_x` has std 1e-3. Softmax over experts **in fp32**, divided by
  `temperature`. Output `y = Σ_e p_e · expert_e(x)`. No top-k, no epsilon.
- Conditioning delivery: the model sets `moe.cond = e` on each MoE module before
  `swinViT` runs, and clears it after. The module only reads it.
- Statistics: each forward **overwrites** `moe.last_stats` (mean probability per
  expert, router entropy, z-loss term). `collect_router_stats(model)` reads them right
  after `swinViT` returns. Never append to a list (recompute safety, CLAUDE.md
  invariant 6).
- Auxiliary loss: `balance_weight · KL(mean_p ‖ uniform) + z_loss_weight · mean(logsumexp(logits)²)`
  per MoE block, added in `BasePickingModel.training_step` when MoE is enabled.
  Keep `balance_weight` small: experts are meant to depend on size, and an equal-use
  constraint works against that.
- Upcycling: `upcycle_to_moe(model)` copies each replaced `mlp.linear1/linear2` into
  every expert. With `branch_scale = 0`, identical experts, and weights that sum to
  one, the output equals the dense model.
- Extra noise for symmetry breaking is not needed: the conv branches differ, so their
  `branch_scale` gradients differ.
- DDP: all experts and router parameters are used in every step.

## Tasks
- **T6.1** `tomocpt.networks.moe`: `ScaleExpertMoE`, `maybe_inject_moe(model, moe_cfg)`,
  `upcycle_to_moe(model)`, `collect_router_stats(model) -> dict`, `moe_aux_loss(stats, cfg)`.
- **T6.2** Integration through `maybe_inject_moe` and the training-step auxiliary loss
  (the two path exceptions above).
- **T6.3** `benchmarks/moe_ablation.py`: three arms with equal steps, data, and
  3 seeds each:
  - **A**: Gate 3 conditioned dense model.
  - **B**: A plus all conv branches in every chosen MLP, summed with learned
    `branch_scale`, **no router** (same parameters as C, no routing).
  - **C**: A plus `ScaleExpertMoE`.
  Report per size bin (held-out sizes included) F-beta and F1 (mean ± sd), training
  time per step, and peak memory, using the Gate 4 harness.

## Public names
`tomocpt.networks.moe.{ScaleExpertMoE, maybe_inject_moe, upcycle_to_moe, collect_router_stats, moe_aux_loss}`.

## Pass criteria (implementation)
- **G6-C1** Upcycling equivalence: tiny dense conditioned model vs its upcycled MoE,
  eval, CPU fp32, random input and `d_vox ∈ {10, 20, 32}`: max |Δlogits| < 1e-5.
- **G6-C2** Router conditioning starts inert: at initialisation, changing `d_vox`
  changes no router probability (atol 1e-7), because `W_c = 0`.
- **G6-C3** bf16 autocast forward and backward on CPU: no NaN or Inf.
- **G6-C4** Recompute safety: with `use_checkpoint=True`, after one training step
  `collect_router_stats` returns exactly one entry per MoE block, with values equal to
  a run with `use_checkpoint=False` (atol 1e-6).
- **G6-C5** DDP rule: after one backward pass every parameter has `grad is not None`.
- **G6-C6** Disabled MoE creates no MoE modules, and the model equals Gate 3 (state-dict keys).
- **G6-C7** Legacy and Gate 3 checkpoints upcycle and load for fine-tuning.
- **G6-C8** (slow) Size-dependent routing is learned: after training on mixed sizes,
  the router mass on dilated experts rises with `log d_vox` (Spearman ρ > 0.5 over
  the size bins) in `layers2`.

## Decision (human sign-off)
Status `PASSED` with `moe.enabled` default true for v2 only if **C** beats both **A**
and **B** by ≥ 0.02 mean F-beta over size bins, with the seed ranges not overlapping,
on synthetic data and on CZII. Otherwise status `PASSED_NO_SHIP`: keep the code,
keep `enabled = False`, and write the result into the report and the README.
