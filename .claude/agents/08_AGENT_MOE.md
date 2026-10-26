# 08_AGENT_MOE.md — Gate 8: structure- and scale-expert soft MoE (experiment with a decision)

## Mission
Test whether a mixture of experts improves the **Gate 7 multi-structure model**:
particles of many sizes, membranes, filaments, organelles and void in one pass.
Build it correctly, compare it fairly with two controls, and report. It ships
(`moe.enabled` default true for multi-structure models) only if it wins.

## Why MoE can help here
- The tasks need different local operators: isotropic blobs at several sizes, thin
  sheets, thin tubes, large smooth regions. One dense MLP per Swin block must serve all
  of them. Experts with different spatial kernels give the network separate capacity per
  kind of structure, and a router chooses per token.
- Gate 7 supplies labels for each structure class. Those labels can also supervise the
  router (optional, below), so names like "sheet expert" now mean something.
- Depthwise convolutions need every spatial neighbour, so each expert runs on the whole
  token grid. This is a **soft (dense) MoE**: compute does not drop. The benefit, if any,
  is conditional capacity, not speed.
- Kernels are fixed in the array frame, and the missing wedge makes Z different from
  X and Y. The directional experts below are a set of axis-aligned operators that the
  network can combine; they are not orientation detectors.

## Allowed write paths
- `tomocpt/networks/moe.py` (new)
- `tomocpt/defaultConfigs/moe_config.py`
- Path exception: `tomocpt/defaultConfigs/network_config.py`, one call
  `maybe_inject_moe(model, self.moe)` at the end of `NetworkConfig.build_model`.
- Path exception: `tomocpt/networks/pickingModel.py` and
  `tomocpt/networks/distillationModel.py`, the MoE auxiliary loss lines in the training
  steps only.
- `benchmarks/moe_ablation.py` (new)
- `.claude/reports/gate_08_report.json`

## Design
Stage geometry (CLAUDE.md Section 1.4): default stages `layers2` (16³, 2fs) and
`layers3` (8³, 4fs). `layers4` (4³) is too small for dilated or 5-long kernels.

### Experts (`moe.experts`, by name)
Each expert: `linear1 → (+ branch_scale · Σ branches(h) / n_branches) → GELU → drop →
linear2 → drop`, with the same shapes as MONAI `MLPBlock`. All convolutions are depthwise
(`groups = mlp_dim`). `branch_scale` is a scalar parameter initialised to 0.

| Name | Branches | Intended use |
|---|---|---|
| `mlp` | none | plain MLP; points, textures |
| `dw_d1` | 3×3×3, dilation 1 | small blobs |
| `dw_d2` | 3×3×3, dilation 2 | medium blobs |
| `dw_d3` | 3×3×3, dilation 3 | large blobs |
| `line` | 1×1×5, 1×5×1, 5×1×1 | filaments |
| `sheet` | 1×5×5, 5×5×1, 5×1×5 | membranes |
| `smooth` | 3×3×3 average pool (fixed) + 1×1×1 | organelle interiors, void |

Default `moe.experts = ("mlp", "dw_d1", "dw_d2", "dw_d3", "line", "sheet", "smooth")`.

### Router
`logits = W_x · x + W_c · c`, where `x` is the block input (already `norm2`-ed by MONAI)
and `c` is the Gate 7 `ConditionEmbedding(d_vox, apix)`. `W_c` is zero-initialised;
`W_x` has std 1e-3. Softmax over experts **in fp32**, divided by `temperature`.
Output `y = Σ_e p_e · expert_e(x)`. No top-k, no epsilon.
- Conditioning delivery: the model sets `moe.cond = c` on each MoE module before `swinViT`
  runs, and clears it after. The module only reads it.
- Statistics: each forward **overwrites** `moe.last_stats` (mean probability per expert,
  router entropy, z-loss term, and the router probability map for the optional
  supervision). `collect_router_stats(model)` reads them right after `swinViT` returns.
  Never append to a list (recompute safety, CLAUDE.md invariant 6).

### Losses
- `balance_weight · KL(mean_p ‖ uniform) + z_loss_weight · mean(logsumexp(logits)²)`
  per MoE block. Keep `balance_weight` small (default 0.01): tomograms differ in content,
  and equal use of every expert in every batch is not wanted.
- Optional router supervision (`moe.route_supervision_weight`, default 0): for samples
  where a structure class is available (Gate 7 `seg_available`), pool its target to the
  token grid and add cross-entropy that pushes those tokens towards the mapped experts
  (`moe.class_to_experts`, default membrane → `sheet`, filament → `line`,
  organelle and void → `smooth`; particles → `dw_d*` by `d_vox`). Only tokens with pooled
  label > 0.5 count. This is an ablation arm, not the default.

### Upcycling
`upcycle_to_moe(model)` copies each replaced `mlp.linear1/linear2` into every expert.
With `branch_scale = 0`, identical linear weights, `smooth` branch scaled by 0 too, and
weights that sum to one, the output equals the dense model. The branches differ, so
their `branch_scale` gradients differ and no extra noise is needed for symmetry breaking.

DDP: every expert and router parameter is used in every step.

## Tasks
- **T8.1** `tomocpt.networks.moe`: `ScaleExpertMoE`, `EXPERT_REGISTRY`,
  `maybe_inject_moe(model, moe_cfg)`, `upcycle_to_moe(model)`,
  `collect_router_stats(model) -> dict`, `moe_aux_loss(stats, cfg, seg_batch=None)`.
- **T8.2** Integration through `maybe_inject_moe` and the training-step auxiliary loss.
- **T8.3** `benchmarks/moe_ablation.py`: equal steps, data and 3 seeds per arm:
  - **A**: Gate 7 multi-structure dense model.
  - **B**: A plus every conv branch of every expert in each chosen MLP, summed with
    learned `branch_scale`, **no router** (same operators, no routing).
  - **C**: A plus `ScaleExpertMoE`.
  - **C+S**: C with router supervision on.
  Report per class (Dice / clDice) and per particle size bin (F-beta, F1), mean ± sd,
  training time per step, peak memory, and router usage maps per class.

## Public names
`tomocpt.networks.moe.{ScaleExpertMoE, EXPERT_REGISTRY, maybe_inject_moe, upcycle_to_moe, collect_router_stats, moe_aux_loss}`.

## Pass criteria (implementation)
- **G8-C1** Upcycling equivalence: tiny dense Gate 7 model vs its upcycled MoE, eval, CPU
  fp32, random input, `d_vox ∈ {10, 20, 32}`, `apix ∈ {None, 10, 40}`:
  max |Δlogits| < 1e-5 on every channel.
- **G8-C2** Router conditioning starts inert: at initialisation, changing `d_vox` or
  `apix` changes no router probability (atol 1e-7), because `W_c = 0`.
- **G8-C3** bf16 autocast forward and backward on CPU: no NaN or Inf.
- **G8-C4** Recompute safety: with `use_checkpoint=True`, after one training step
  `collect_router_stats` returns exactly one entry per MoE block, with values equal to a
  run with `use_checkpoint=False` (atol 1e-6).
- **G8-C5** DDP rule: after one backward pass every parameter has `grad is not None`,
  with and without router supervision, and for a batch that has no structure labels.
- **G8-C6** Disabled MoE creates no MoE modules, and the model equals Gate 7
  (state-dict keys).
- **G8-C7** Legacy, Gate 3 and Gate 7 checkpoints upcycle and load for fine-tuning.
- **G8-C8** Each registered expert has the stated kernel shapes and is depthwise.
- **G8-C9** (slow) Routing follows structure: on the Gate 7 phantom, after training arm C
  (no router supervision), the mean router mass on `sheet` is higher on membrane tokens
  than on solvent tokens, and the mass on `line` is higher on filament tokens (report the
  ratios; pass if both > 1.2).

## Decision (human sign-off)
Status `PASSED` with `moe.enabled` default true for multi-structure models only if
**C** (or **C+S**) beats both **A** and **B** by ≥ 0.02 on the mean of per-class Dice/clDice
and picking F-beta, with seed ranges not overlapping, on synthetic data and on portal data.
Otherwise status `PASSED_NO_SHIP`: keep the code, keep `enabled = False`, and write the
result into the report and the README.
