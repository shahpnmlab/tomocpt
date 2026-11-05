# 04f_AGENT_AVERAGING_SUBVOXEL.md — Gate 4f: weight averaging, centroid weight, sub-voxel picks, offset head (TODO 3b.3a, 3b.3b)

Added 4–5 Nov 2026 by human decisions (TODO 3b.3a, 3b.3b; "build 3b.3a and 3b.3b").

## Why
- **Legacy SWA never reaches a checkpoint.** Lightning 2.4 `StochasticWeightAveraging` copies the
  average into the model only in `on_train_end` when `current_epoch - 1 == swa_end`
  (= `max_epochs - 1`), i.e. only if training runs to `max_epochs`, and then after the last
  validation, so `ModelCheckpoint` (best `val_loss`) only ever saves raw weights. With early
  stopping (all Gate 7 folds stopped at epoch 25–28 of 60) no average is ever used.
- **Narrow core not learned** (TODO 3a.3 on TS_69_2): predicted peaks follow only the wide Gaussian
  (FWHM 0.63–0.83 R vs 0.53 R, heights 0.3–0.9); the core voxels are a tiny share of the weighted loss.
- **Picks** are already sub-voxel: `predict.helpers._refine_peaks_subpixel_gpu` fits a 3D quadratic to the
  log of the map around each local maximum (exact for a Gaussian peak; finding of 5 Nov 2026, which corrects
  TODO 3b.3b's "integer voxel indices"). With flattened, widened peaks (3a.3) that fit is the reference the
  offset head has to beat.

## Design

### D1 Weight averaging (3b.3a) — `tomocpt/training/averaging.py`, callback `WeightAveraging`
`train.weight_averaging`: `"swa"` (default, the legacy Lightning callback, unchanged) | `"ema"` |
`"swa_plateau"` | `"none"`.
- `"ema"`: after every optimiser step `ema = d_t * ema + (1 - d_t) * w` with
  `d_t = min(train.ema_decay, (1 + t) / (10 + t))`, `t` = number of updates so far (0-based), over
  every floating-point parameter and buffer of the LightningModule (others copied).
- `"swa_plateau"`: averaging starts at the first validation where `val_loss_smooth` has not
  improved on its best by more than `train.swa_plateau_rel_delta * |best|` (relative, scale-free)
  for `train.swa_plateau_patience` validations; from then on the average is the equal-weight mean
  of the weights at the end of each training epoch (`n`-th update: `avg += (w - avg) / n`).
- Both: while an average exists, every validation (and so `val_loss`, `val_loss_smooth`, early
  stopping and `ModelCheckpoint`) runs on the **averaged** weights: they are swapped into the
  model at `on_validation_start` and the raw weights are swapped back at the next
  `on_train_batch_start` (after checkpointing). At the end of fit the model holds the average.
  The callback's `state_dict` holds the average, the raw weights, `t` / `n` and the trigger state,
  so `resume_from` continues exactly.
- Replaces Lightning SWA (not added) for `"ema"`, `"swa_plateau"`, `"none"`. With
  `train.schedule_free` only `"swa"` (ignored: no SWA under schedule-free, Gate 4e) or `"none"` are
  allowed; `"ema"` / `"swa_plateau"` raise ValueError (the optimiser already averages).

### D2 Centroid weight (3b.3b a) — `training/multiclass_loss.py`
`train.particles.core_weight: float = 0.0`, `train.particles.core_power: float = 4.0`. On particle
channels the Section 1.5 value weight becomes `w * (1 + core_weight * y^core_power)` (Huber and focal
value terms; the Hessian term unchanged). `core_weight = 0` leaves the loss bit-identical.

### D3 Sub-voxel picks (3b.3b b, c) — `infer.subvoxel_refine`: `"fit"` (default) | `"offset"`
- `"fit"`: the legacy log-quadratic fit (unchanged; amended 5 Nov 2026: the planned per-axis parabola
  was dropped, the legacy fit already does this better).
- `"offset"`: each pick becomes `v + offset(v)`, `v` = the fitted pick rounded to a voxel, `offset`
  the offset head's prediction (D4) there (the head predicts the vector from any core voxel to the
  centre). ValueError if the model (or an ensemble member) has no offset head, or with
  `skip_inference` (offsets are not saved).
- Scores unchanged; picks stay `[N, 4]` rows `z, y, x, score`. Applied before cross-class
  suppression. The saved-picks fingerprint (Gate 4c) includes the field.

### D4 Offset head (3b.3b c)
- `train.particles.offset_weight: float = 0.0`. When > 0 (pipeline v2 only) the picking model gets
  `offset_head`: `Conv3d(fs, 16, 3, padding 1) -> GELU -> Conv3d(16, 3, 1)` on the `decoder1`
  output (full resolution, after FiLM for conditioned models), last layer zero-initialised
  (invariant 6). Class-agnostic: 3 channels `(dz, dy, dx)` in working voxels. Not created when off
  (invariant 5). `forward` still returns logits only (invariant 7); the network gains
  `forward_features(x, apix=None, with_decoder1=False)` (`True` appends `decoder1`).
- **Target** (`training/offsets.py`, `offset_targets(y, class_d_vox, available, particle_channels)`),
  computed from the (augmented) dense target so flips and rotations need no extra handling: for
  class `c` with radius `R = d_vox / 2`, the centre of the blob around voxel `v` is the
  `y`-weighted centroid in the box `|u| <= ceil(1.3 R) + 1` around `v` (covers the blob from any core voxel) (box sums by cumulative sums);
  target `= centroid - v` on voxels with `y_c > 0.5` (the core); where cores of two classes
  overlap the larger `y` wins. Returns `(target [B, 3, D, H, W], mask [B, 1, D, H, W])`.
- **Loss**: `offset_weight * mean(|pred - target|)` over masked voxels and the 3 components (0 with
  the head still in the graph when the mask is empty: DDP). Added to `loss` and `val_loss`; logged as
  `loss_offset` / `val_loss_offset`.
- **Inference**: `BasePickingModel.predict_step_with_offsets(x, apix=None, classes=None) ->
  (probs, offsets)`; `EnsembleModel` averages both; `_predict_level` accumulates the offsets
  Hann-weighted like the maps (flip TTA negates the flipped component); `predict_class_maps(...,
  offsets_out=None)` fills `{row: [3, D, H, W]}` in **native** voxels (resampled like the maps and
  multiplied by `apix_work / apix_native`); `peaks_from_class_maps(..., offsets=None)` applies them.

## Config
| Path | Default |
|---|---|
| `train.weight_averaging` | `"swa"` |
| `train.ema_decay` | `0.999` |
| `train.swa_plateau_patience`, `train.swa_plateau_rel_delta` | `5`, `1e-3` |
| `train.particles.core_weight`, `train.particles.core_power` | `0.0`, `4.0` |
| `train.particles.offset_weight` | `0.0` |
| `infer.subvoxel_refine` | `"fit"` |

## Allowed write paths
`tomocpt/training/{averaging,offsets,multiclass_loss,helpers}.py` (`helpers`: `_setup_callbacks`),
`tomocpt/networks/{pickingModel,swinunetr,baseModel}.py`, `tomocpt/defaultConfigs/{train_config,
particles_config,infer_config}.py`, `tomocpt/predict/{multiclass,ensemble,calibration}.py`,
`README.md`, `.claude/reports/gate_04f_report.json`.

## Pass criteria
- **G4f-C1** Defaults: callbacks, loss, model state-dict keys, `forward`, and predictions unchanged
  (full suite; `core_weight = 0` bit-identical loss).
- **G4f-C2** EMA: after a short CPU fit the callback's average equals the hand recursion over the
  recorded post-step weights (atol 1e-6); validation sees the average; training steps see raw
  weights; the saved checkpoint and the end-of-fit model hold the average; `state_dict` round trip.
- **G4f-C3** Plateau trigger on a given `val_loss_smooth` sequence starts at the expected epoch
  (relative delta, patience); after the start the average is the mean of the epoch-end weights.
- **G4f-C4** `schedule_free` + `ema` / `swa_plateau` raises ValueError; `none` adds no averaging.
- **G4f-C5** Core weight: equals `w * (1 + k y^p)` on particle channels by hand on small tensors;
  filament channels unchanged.
- **G4f-C6** The default `"fit"` recovers a sub-voxel Gaussian peak centre within 0.1 voxel through
  `peaks_from_class_maps` (legacy refinement, documents the reference).
- **G4f-C7** `offset_targets` on targets rendered at known float centres gives `centre - v` on the
  core (atol 0.05 voxel) and an empty mask where there is no particle.
- **G4f-C8** Offset head: absent when off; zero-initialised and zero output when on; every head
  parameter gets a gradient in a training step (also with an empty target); loss includes the term;
  checkpoint round trip `strict=True`; `forward` unchanged.
- **G4f-C9** Predict with `"offset"`: with a constant predicted offset every pick is the rounded
  `"fit"` pick plus that offset times `apix_work / apix_native` (atol 6e-3: STAR files hold 2 decimals); `"offset"` without a head raises
  ValueError; an ensemble averages the members' offsets.
