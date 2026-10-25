# 03_AGENT_SIZE_CONDITIONED_MODEL.md — Gate 3: size-conditioned picking model

## Mission
Give the network the target diameter `d_vox` as an input, so that one model picks
particles of any size inside the band, and one backbone pass can serve several
sizes. Add the v2 loss. Add an optional same-species positive-unlabeled (PU) mode.
Keep v1 and legacy checkpoints unchanged.

## Design
- **Conditioning by FiLM in the decoder.** The encoder (`swinViT`, `encoder1..4`)
  stays unconditioned, so `encode(x)` runs once and `decode(features, d_vox)` runs
  once per requested size.
- FiLM points (6): after `encoder10` (16fs, 2³), `decoder5` (8fs), `decoder4` (4fs),
  `decoder3` (2fs), `decoder2` (fs), `decoder1` (fs). `f ← f · (1 + γ) + β`, with
  `γ, β = Linear(cond_dim → 2C)(e)`, weights and bias **zero-initialised**.
- Size embedding `e = SizeEmbedding(d_vox)`: `t = log(d_vox / 20)`;
  features `[t, sin(2^k t), cos(2^k t)]` for `k = 0…7`; then
  `Linear(17 → 64) → SiLU → Linear(64 → 64)`.
- Same-species PU (optional, `train.pu.mode = ema_teacher`): annotators miss
  particles, and v1 drops border particles. An EMA copy of the model marks voxels that
  it believes are particles; the loss lowers the negative weight there.

## Allowed write paths
- `tomocpt/networks/{swinunetr,conditioning,pickingModel,distillationModel,inference,baseModel}.py`
- `tomocpt/training/**` (new: `pu.py`)
- `tomocpt/defaultConfigs/{network_config,pu_config}.py`
- `.claude/reports/gate_03_report.json`

## Tasks

### T3.1 Modules (`networks/conditioning.py`, `networks/swinunetr.py`)
- `SizeEmbedding(dim=64)`, `FiLM(channels, cond_dim)` as in Design.
- `EncoderFeatures = NamedTuple(hidden: list[Tensor], enc0, enc1, enc2, enc3)`.
- `SizeConditionedSwinUNETR(MySwinUNETR)`:
  - `__init__(..., cond_dim: int = 64)`; new submodules live under `self.cond`
    (`self.cond.size_embed`, `self.cond.film` as `ModuleList`).
  - `forward(x, d_vox=None) -> (logits, hidden[4])`. With `d_vox=None` the
    computation is exactly `MySwinUNETR.forward`.
  - `encode(x) -> EncoderFeatures`; `decode(feats, d_vox) -> logits`.
- `load_legacy_state_dict(model, state_dict) -> (missing, unexpected)`: loads with
  `strict=False`. Raises if `unexpected` is not empty or if a missing key does not
  start with `cond.`.
- `NetworkConfig.build_model`: when `size_conditioning` is true and the model type is
  SwinUNETR with labels, build `SizeConditionedSwinUNETR`. For UNet, raise
  `NotImplementedError("size conditioning needs SwinUNETR")`.

### T3.2 LightningModules (`networks/pickingModel.py`, `baseModel.py`, `inference.py`)
- `BasePickingModel.forward(x, d_vox=None)`. `training_step` and `validation_step`
  read `d_vox` from the batch when conditioning is on. If conditioning is on and the
  batch has no `d_vox`, raise `ValueError`.
- `BasePickingModel.loss(y_pred, y_true, d_vox=None, neg_factor=None)`:
  - v1 or `d_vox is None`: Section 1.5 exactly.
  - v2: Hessian of both `ŷ` and `y` multiplied per sample by `(d_vox / 20)²`
    before the Huber term (CLAUDE.md Section 5.3).
  - `neg_factor` (same shape as `y`, PU only) multiplies the weight where `y == 0`.
- `predict_step(batch, batch_idx=0, dataloader_idx=0, d_vox=None)` returns
  `sigmoid(logits)`. Gate 4 passes `d_vox`.
- Fine-tune parameter groups: for conditioned models or pipeline v2, every trainable
  parameter is in exactly one group: `swinViT` ×0.01, `encoder*` ×0.1, all others
  (`decoder*`, `out`, `cond`) ×1.0. Keep the v1 grouping for v1 unconditioned models
  and record the frozen-`out` defect in the report for a human decision.
- `InferenceModel.on_load_checkpoint` also removes keys that start with `pu_teacher.`.

### T3.3 Model creation (`training/helpers._create_model`)
- Fine-tune from a legacy checkpoint into a conditioned model (CLI
  `network.size_conditioning=True`): build the conditioned model, load the legacy
  weights with `load_legacy_state_dict`, and start a new run (no optimizer state).
- SSL checkpoint → conditioned picking model: same rule.
- Resume (`resume_from`) never changes the architecture; raise if the requested
  config differs in `size_conditioning` or `moe.enabled`.

### T3.4 Distillation (`networks/distillationModel.py`)
- `distill_sample_mask(d_vox, teacher_px, tol=1.0) -> BoolTensor[B]`.
  A legacy (fixed-particle) teacher only distils samples with `|d_vox - teacher_px| < tol`.
  `teacher_px` comes from the teacher's checkpoint config.
- A conditioned teacher receives `d_vox` and distils all samples.
- If no sample qualifies, the distillation terms are `0 · (student output sum)` so the
  graph stays complete (DDP).

### T3.5 PU mode (`training/pu.py`)
- `EMATeacher(model, decay)`: deep copy with `requires_grad_(False)`, stored as
  `self.pu_teacher`; `update(model)` after each optimizer step.
- `pu_negative_weight_factor(p_teacher, y, prob_threshold, max_neg_reduction) -> Tensor`:
  `1 - max_neg_reduction · p_teacher` where `y == 0` and `p_teacher > prob_threshold`;
  1 elsewhere.
- Active only when `epoch >= start_epoch`. `mode = none` creates no teacher.

## Public names
`tomocpt.networks.conditioning.{SizeEmbedding, FiLM}`,
`tomocpt.networks.swinunetr.{SizeConditionedSwinUNETR, EncoderFeatures, load_legacy_state_dict}`,
`tomocpt.networks.pickingModel.BasePickingModel` (`forward`, `loss`, `predict_step` signatures above),
`tomocpt.networks.distillationModel.distill_sample_mask`,
`tomocpt.training.pu.{EMATeacher, pu_negative_weight_factor}`.

## Pass criteria
- **G3-C1** v1 lock: tiny unconditioned model, pipeline v1, dropout 0: `training_step`
  loss on a fixed batch equals the `gate-0-base` golden (rtol 1e-6).
- **G3-C2** Zero-init equivalence: `SizeConditionedSwinUNETR` with legacy weights
  gives the same logits as `MySwinUNETR` (atol 1e-6) for `d_vox ∈ {8, 20, 40}` and `None`.
- **G3-C3** `decode(encode(x), d)` equals `forward(x, d)[0]` (atol 1e-6).
- **G3-C4** After one backward pass with conditioning, every parameter has
  `grad is not None`; at least one FiLM weight gradient is non-zero.
- **G3-C5** v2 loss at `d_vox = 20` equals v1 loss (rtol 1e-6). The normalised
  Hessian at the centre of a target with `d = 12` and `d = 24` differs by < 10 %.
- **G3-C6** Legacy checkpoint → conditioned fine-tune through `_create_model`:
  only `cond.*` keys missing; initial logits equal legacy (atol 1e-6).
- **G3-C7** Parameter groups (conditioned/v2): each trainable parameter appears in
  exactly one group; `out.*` and `cond.*` are at ×1.0. v1 grouping unchanged.
- **G3-C8** Conditioning on, batch without `d_vox` → `ValueError`.
- **G3-C9** `distill_sample_mask` values; legacy teacher + conditioned student runs a
  training step; with an all-false mask the loss still back-propagates to all
  student parameters.
- **G3-C10** PU: factor values on a hand-made tensor; `mode = none` loss equals v1;
  `InferenceModel` strips `pu_teacher.` keys and loads with `strict=True`.
- **G3-C11** (slow) Size selectivity: chunks with spheres of 12 and 28 voxels; each
  sample's condition picks one size and its target shows only that size. After
  training, on held-out chunks: F1 ≥ 0.8 for the conditioned size, and mean
  prediction at centres of the other size < 0.2.
- **G3-C12** (slow) Size interpolation: train on sizes {10, 14, 24, 32}; test at 18:
  F1 ≥ 0.7.

F1 in slow tests: peaks from `predict.helpers.peak_local_max_torch` with threshold
0.3 and min distance `d/2`; a hit is within `d/4` voxels of a true centre.

## Notes for the next gate
Give the exact call pattern for multi-size inference (`encode` once, `decode` per
size) and the checkpoint keys added.
