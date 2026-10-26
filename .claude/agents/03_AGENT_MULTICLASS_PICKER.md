# 03_AGENT_MULTICLASS_PICKER.md — Gate 3: multi-class picker with voxel-size conditioning

## Mission
One network outputs one centroid heatmap channel per class (CLAUDE.md invariant 7),
trained on pipeline-v2 batches with partial labels, and knows the working voxel size
through FiLM conditioning. Keep v1 models and legacy checkpoints unchanged.

## Why one channel per class
Coding the class in the peak height (class 1 peaks at 1, class 2 at 2, …) was rejected:
peak height already means confidence; a network unsure between class 1 and class 3
outputs about 2, a confident wrong class; it imposes an order on species; tails of one
class create false peaks of another; and the loss grows with the class number. One sigmoid
channel per class keeps the current target, loss and peak finding per class, and partial
labels become a per-class mask.

## Why condition on the voxel size
With levels and scale jitter (Gate 2) a class is trained at several voxel sizes. The same
voxel diameter then means different physical sizes: at 10 Å/px a 150 Å ribosome is 15
voxels, and so is a 300 Å particle at 20 Å/px. Without `apix_work` the 300 Å channel is
taught to fire on 15-voxel objects at one level and to stay silent on them at another.
FiLM on `t = log(apix_work / 10)` removes the contradiction. When every class fits one
level (for example CZII at 10 Å/px) and no jitter is used, conditioning is not needed;
the benchmark (Gate 4) measures both.

## Design
- FiLM points (6) in the decoder path: after `encoder10` (16fs, 2³), `decoder5` (8fs),
  `decoder4` (4fs), `decoder3` (2fs), `decoder2` (fs), `decoder1` (fs).
  `f ← f · (1 + γ) + β`, with `γ, β = Linear(cond_dim → 2C)(e)`, weights and bias
  **zero-initialised** (invariant 6). The Swin encoder stays unconditioned, so the Gate 5
  relative-scale head (which reads encoder features) cannot read the answer from the input.
- Scale embedding `e = ScaleEmbedding(apix)`: `t = log(apix / 10)`; features
  `[t, sin(2^k t), cos(2^k t)]` for `k = 0…7`; then `Linear(17 → 64) → SiLU → Linear(64 → 64)`.
- `out_channels = C` = length of the class table
  (`labels.multiclass.class_table_from_config(prepData)`).

## Allowed write paths
- `tomocpt/networks/{swinunetr,conditioning,pickingModel,distillationModel,inference,baseModel}.py`
- `tomocpt/training/**` (except `helpers._prepare_data_if_needed`; new: `multiclass_loss.py`)
- `tomocpt/defaultConfigs/{network_config,particles_config}.py`
- `.claude/reports/gate_03_report.json`

## Config (this gate adds)
- `NetworkConfig.scale_conditioning: bool = False`.
- `defaultConfigs/particles_config.py`, `TrainConfig.particles: ParticlesConfig`:
```python
@dataclass
class ParticlesConfig:
    class_weights: Optional[str] = None   # "1:1.0,2:2.0" by class number
    loss_checkpoint: bool = True          # Hessian term one class at a time, checkpointed
```

## Tasks

### T3.1 Modules (`networks/conditioning.py`, `networks/swinunetr.py`)
- `ScaleEmbedding(dim=64)`, `FiLM(channels, cond_dim)` as in Design.
- `ScaleConditionedSwinUNETR(MySwinUNETR)`: `__init__(..., cond_dim: int = 64)`; new
  submodules live under `self.cond` (`self.cond.scale_embed`, `self.cond.film` as
  `ModuleList`). `forward(x, apix=None) -> (logits, hidden[4])`; with `apix=None` the
  computation is exactly `MySwinUNETR.forward`.
- `load_legacy_state_dict(model, state_dict) -> (missing, unexpected)`: `strict=False`;
  raises if `unexpected` is not empty or a missing key does not start with `cond.`.
- `load_single_class_into_multiclass(model, state_dict) -> None`: copies a one-channel
  `out` into every channel of a C-channel `out` (each class starts as the old picker),
  loads everything else as `load_legacy_state_dict` does, and logs it.
- `NetworkConfig.build_model`: `out_channels` from the class table; `scale_conditioning`
  with SwinUNETR builds `ScaleConditionedSwinUNETR`; with UNet raises
  `NotImplementedError("scale conditioning needs SwinUNETR")`.

### T3.2 Loss (`training/multiclass_loss.py`)
`multiclass_picking_loss(y_pred, y_true, class_available, class_d_vox, class_weights=None,
checkpoint=True) -> Tensor`, CLAUDE.md Section 5.4:
- per `(b, c)`: Section 1.5 weights computed per sample and per class; Hessian of both
  `ŷ` and `y` multiplied by `(class_d_vox[b, c] / 20)²` before the Huber term;
- unavailable pairs contribute exactly 0; the total is the mean over available pairs;
- `class_weights` multiply each class;
- the Hessian term runs one class at a time inside `torch.utils.checkpoint` when
  `checkpoint=True`;
- DDP rule: if a class is unavailable in the whole batch, add `0 · y_pred[:, c].sum()`.
With `C = 1`, all available and `class_d_vox = 20`, it equals `BasePickingModel.loss`.

### T3.3 LightningModules (`pickingModel.py`, `baseModel.py`, `inference.py`)
- `BasePickingModel.forward(x, apix=None)`; `training_step` and `validation_step` use
  `multiclass_picking_loss` for pipeline v2 batches and the legacy loss for v1 batches;
  they pass `batch["apix_work"]` when conditioning is on and raise `ValueError` if it is
  missing.
- `predict_step(batch, batch_idx=0, dataloader_idx=0, apix=None)` returns `sigmoid(logits)`.
- Per-class validation metrics: `val_loss_<class name>` for each class.
- TensorBoard images: one panel per class (central slice of target and prediction).
- Fine-tune parameter groups: `cond.*` falls into the ×1.0 group that
  `BaseModel._get_model_parameters_for_optimizer` builds for the rest; check that every
  trainable parameter is in exactly one group.

### T3.4 Model creation (`training/helpers._create_model`)
- Fine-tune from a legacy single-class checkpoint into a v2 model (multi-class and/or
  conditioned): build the new model, load with `load_single_class_into_multiclass` (or
  `load_legacy_state_dict` when C = 1), start a new run (no optimizer state).
- Fine-tune from a v2 checkpoint with a different class table: load everything except
  `out`; channels of classes present in both tables (matched by class number) are copied,
  new channels start from the mean of the old ones. Log the mapping.
- SSL checkpoint → v2 picking model: same rules.
- Resume (`resume_from`) never changes the architecture; raise if the requested config
  differs in `scale_conditioning` or the class table.

### T3.5 Distillation (`networks/distillationModel.py`)
Distillation stays a v1 feature. A pipeline-v2 student, a multi-class student or a
conditioned student raises `NotImplementedError("distillation supports single-class v1
models")` at construction.

## Public names
`tomocpt.networks.conditioning.{ScaleEmbedding, FiLM}`,
`tomocpt.networks.swinunetr.{ScaleConditionedSwinUNETR, load_legacy_state_dict, load_single_class_into_multiclass}`,
`tomocpt.training.multiclass_loss.multiclass_picking_loss`,
`tomocpt.networks.pickingModel.BasePickingModel` (`forward`, `predict_step` signatures above),
`tomocpt.defaultConfigs.particles_config.ParticlesConfig`.

## Pass criteria
- **G3-C1** v1 lock: tiny unconditioned single-class model, pipeline v1, dropout 0:
  `training_step` loss on a fixed batch equals the `v2-gate-0-base` golden (rtol 1e-6).
- **G3-C2** Zero-init: `ScaleConditionedSwinUNETR` with legacy weights gives the same
  logits as `MySwinUNETR` (atol 1e-6) for `apix ∈ {5, 10, 20}` and `None`.
- **G3-C3** After one backward pass with conditioning, every parameter has `grad is not None`
  and at least one FiLM weight gradient is non-zero.
- **G3-C4** Loss equivalence: C = 1, all available, `class_d_vox = 20` equals the v1 loss
  (rtol 1e-6); the checkpointed per-class form equals the vectorised form on a C = 4 batch
  (rtol 1e-5). The normalised Hessian at the centre of targets with `d = 12` and `d = 24`
  differs by < 10 %.
- **G3-C5** Partial labels: with `class_available[:, c] = False`, the gradient of output
  channel c is exactly 0 (not `None`), and changing that channel's target changes nothing.
- **G3-C6** Per-class weights: a class with 10× fewer positive voxels gets 10× larger
  positive weights than a common class in the same sample (rtol 1e-3).
- **G3-C7** Warm start: after `load_single_class_into_multiclass`, every channel's logits
  equal the single-class logits (atol 1e-6). Class-table change copies matching channels.
- **G3-C8** Legacy checkpoint → v2 fine-tune through `_create_model`: only `cond.*` keys
  missing (and `out` reshaped); initial logits equal legacy per channel (atol 1e-6).
- **G3-C9** Parameter groups: every trainable parameter in exactly one group; `out.*` and
  `cond.*` at ×1.0; v1 grouping unchanged.
- **G3-C10** Conditioning on, batch without `apix_work` → `ValueError`. Resume with a
  different class table or `scale_conditioning` → `ValueError`. Distillation with a v2
  student → `NotImplementedError`.
- **G3-C11** `InferenceModel.load_from_checkpoint(strict=True)` loads a v2 checkpoint.
- **G3-C12** (slow) Scale disambiguation. Phantoms at 10 Å/px with class A (150 Å) and
  class B (300 Å); class B is also trained at 20 Å/px (15 voxels, the same voxel size as A
  at 10 Å/px). After training, on held-out 10 Å/px chunks: F1 ≥ 0.8 for both classes, and
  the conditioned model's mean B prediction at A centres < 0.2. Report the same number
  for `scale_conditioning = False` (expected higher; report only).
- **G3-C13** (slow) Same size, different shape: four classes, two of them the same
  diameter (solid vs hollow spheres). Held-out F1 ≥ 0.8 for every class, and fewer than
  15 % of the true particles of one same-size class are picked in the other's channel.

F1 in slow tests: peaks from `predict.helpers.peak_local_max_torch` with threshold 0.3
and min distance `d/2` per channel; a hit is within `d/4` voxels of a true centre
(`tomocpt.metrics`). Thresholds must pass in 9 of 10 seeds.

## Notes for the next gate
Give the exact call pattern for inference (`forward(x, apix=level.apix_work)`), the
checkpoint keys added, and how the class table is recovered from a checkpoint.
