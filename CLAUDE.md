# CLAUDE.md — TomoCPT v2 (scale-aware multi-class picking): engineering contract

This file is the shared contract for every agent that works on TomoCPT v2.
Read all of it before you write code.

Precedence when two sources disagree:
1. A human instruction in the conversation.
2. The legacy code at tag `v2-gate-0-base` for **legacy behaviour**. `v2-gate-0-base` is
   the first commit of branch `v2-dev` (released `main` + `fix/v1-patches` + this plan);
   the orchestrator records the commit in `.claude/state/gate_status.json`.
   Section 1 records that behaviour. If Section 1 and the code disagree, the code
   is correct. Stop and write an escalation report (see Section 8).
3. This file.
4. Your agent spec in `.claude/agents/`.

---

## 1. Repository facts (verified at `v2-gate-0-base`)

### 1.1 CLI and config
- Entry point: `tomocpt/main.py`. Commands: `tomocpt init`, `tomocpt prepare_data`,
  `tomocpt train`, `tomocpt predict`. There is **no** `tomocpt infer` command.
  `predict` reads the config section `infer`.
- Config is a tree of Python dataclasses in `tomocpt/defaultConfigs/`, rooted at
  `MainConfig` in `tomocpt/mainConfig.py`:
  - `prepData: PrepdataConfig`
  - `train: TrainConfig`, which contains `train.optimizer: OptimizerConfig`,
    `train.network: NetworkConfig` (with `UNET` and `SwinUNETR` sub-configs) and
    `train.augmentation: AugmentationConfig`
  - `infer: InferConfig`
  - `shared: GlobalPropertyConfig` (copied into sections in `MainConfig.__post_init__`)
- `configManager/configManager.py` builds the Typer CLI from these dataclasses.
  Hydra-style dot overrides work (`optimizer.lr=1e-4`). CLI options are spelled with
  underscores, as the field names (`--config_file`, `--output_path`, `--resume_from`);
  hyphenated spellings are rejected. Booleans also get a `--no-<field>` form.
- A global singleton `mainConfig` exists. Legacy code reads it from many places
  (for example `dataManager/transforms.py` reads `mainConfig.prepData` and
  `mainConfig.train`).

### 1.2 Data pipeline (legacy = "pipeline v1")
1. `prepare_data` (`labels/run.py`, `labels/helpers.py`):
   - Reads STAR or IMOD coordinates (both give points as `(x, y, z)`). Writes one
     full-size label MRC per tomogram at the **native** voxel size, plus a tracking CSV
     (`tomogram_path`, `label_path`, `particle_diameter_angst`, `num_particles`).
     One dataset = one species; several species are several datasets.
   - It does **not** chunk.
2. `train` chunks the data (`training/helpers._prepare_data_if_needed` →
   `dataManager/chunking.do_chunking`) if `chunks_dir/done_labels_{True|False}.txt`
   is absent. If it exists, chunking is skipped. It refuses (RuntimeError) chunks that
   have no `chunk_layout.json` when tomogram stems repeat (chunks built before the
   collision fix).
3. Per tomogram (`dataManager/transforms.PreprocessTomogramd` →
   `dataManager/preprocessing.preprocess_tomogram`):
   - Robust normalisation: `clip((x - median) / (p95 - p5), -3, 3)`.
   - **Fixed-particle rescale**: trilinear resize so that the particle *diameter*
     becomes `prepData.desired_particle_pixel_size` (default 20) voxels. The resize
     is skipped if the native diameter is within 1 voxel of the target. No anti-aliasing.
   - Pad to at least 64 per axis. The label is resized to the same shape (trilinear).
4. TorchIO `GridSampler`: chunk 64³, overlap 32 (stride 32). Empty chunks are dropped
   with probability `clip(1 - ALPHA * f1/f0, 0, 1)`. Chunks are `.nii.gz` files in
   `chunks_dir/{train,val}/<dir>/volumes/` and `.../labels_supervised/`.
   `<dir>` is the tomogram stem when all stems are unique, otherwise
   `<dataset dir>__<stem>` (`chunking.assign_chunk_dir_names`).
   `chunks_dir/chunk_layout.json` = `{"layout": 2, "dir_names": ..., "split": ...}`.
5. Train/val split (`chunking.split_rows_by_tomogram`): `train_test_split(test_size=0.2,
   random_state=42)` when stems are unique, otherwise `GroupShuffleSplit` grouped by
   tomogram stem. Fewer than 2 distinct tomograms raises `ValueError`.
6. `dataManager/dataloading.VolumeDatsetIO` returns
   `{"input_data": [1,64,64,64], "target_data": [1,64,64,64]}`. `Data` (the
   LightningDataModule) is constructed in `training/run.py`.
7. Training augmentation is built by
   `dataManager/dataloading.build_training_transforms(cfg: AugmentationConfig)`, in this order:
   1. `RandomAffine(degrees=45, p=0.8)` and `OneOf(RandomElasticDeformation, RandomBlur)`;
   2. `RandomFourierDegradation` (`dataManager/fourier_augmentations.py`): missing wedge,
      amplitude-spectrum randomisation, local blend. Off by default
      (`use_mw_aug`, `use_fourier_aug`). It must run after every spatial transform.
      It orients the wedge for `(z, y, x)` arrays and keeps the input mean and std;
   3. `RandomContrastInversion` at `constants.CONTRAST_INVERSION_PROB = 0.5`
      (**always on, deliberately not configurable** — a human decision), then optional
      intensity transforms (`dataManager/intensity_augmentations.py`), all off by default.
   Validation uses no transform. The target is a `tio.LabelMap`: spatial transforms
   resample it with **nearest-neighbour** interpolation, and every
   `tio.IntensityTransform` skips it (see invariant 10).

### 1.3 Target
`labels/helpers.generate_gaussian_sphere(radius_px, grid)`:
`0.5·exp(-r²/2σs²) + 0.5·exp(-r²/2σl²)`, with `σs = 0.33·R/2` and `σl = 0.66·R/2`,
normalised to max 1, set to 0 for `r > R`. `R = round(radius_ang / apix_native)`.
Particles whose box touches or crosses the tomogram border are **skipped**
(`zmin <= 0 or zmax >= dim`), so they become negatives.

### 1.4 Model and stage geometry
- MONAI 1.4.0 `SwinUNETR`, subclass `networks/swinunetr.MySwinUNETR`.
  Defaults (`defaultConfigs/models/swinunetr_config.py`): `feature_size=36`,
  `use_v2=True`, `use_checkpoint=True`, `drop_rate=0.3`, `attn_drop_rate=0.3`,
  `dropout_path_rate=0.2`, `out_channels=1`. MONAI defaults: `depths=(2,2,2,2)`,
  `num_heads=(3,6,12,24)`, window 7, patch 2.
- `MySwinUNETR.forward(x)` returns a **tuple** `(logits, hidden_states_out[4])`.
  `BasePickingModel.forward` keeps `[0]`. Distillation and SSL need the tuple.
- Geometry for a 64³ input and `fs = feature_size` (numbers for `fs=36`):

| Module | Swin blocks run at | Channels | Output after downsample |
|---|---|---|---|
| `patch_embed` | — | — | 32³, fs (36) = `hidden[0]` |
| `layers1` (+`layers1c` if v2) | 32³ | fs (36) | 16³, 2fs (72) = `hidden[1]` |
| `layers2` | 16³ | 2fs (72) | 8³, 4fs (144) = `hidden[2]` |
| `layers3` | 8³ | 4fs (144) | 4³, 8fs (288) = `hidden[3]` |
| `layers4` | 4³ | 8fs (288) | 2³, 16fs (576) = `hidden[4]` |

- Decoder output channels: `encoder10` 16fs at 2³, `decoder5` 8fs at 4³, `decoder4` 4fs
  at 8³, `decoder3` 2fs at 16³, `decoder2` fs at 32³, `decoder1` fs at 64³, then `out`
  (1×1×1 conv, fs → `out_channels`).
- MONAI checkpoints the Swin blocks with `use_reentrant=False`; in backward they run a
  second time. Nothing new runs inside a Swin block in this plan.

### 1.5 Loss (`networks/pickingModel.BasePickingModel.loss`)
Computed on `ŷ = sigmoid(logits)`. Per sample, `N` = voxels per sample,
`S = Σ y` (sum of **soft** labels, not a count):
```
w_pos = N / (S + 1)
w_neg = N / (N - S + 1)
w     = where(y > 0, w_pos, 2 · w_neg)
H(v)  = kornia.filters.spatial_gradient3d(v, mode="diff", order=2)   # 6 channels: xx,yy,zz,xy,xz,yz
L     = mean( w · huber(ŷ, y)  +  10 · Σ_6 huber(H(ŷ), H(y)) )       # second term is NOT weighted by w
```
Huber delta is 1.0. The second term is the full Hessian, not a Laplacian.

### 1.6 Training runtime (`training/helpers.py`, `networks/baseModel.py`)
- RAdam, lr 4e-4, `ReduceLROnPlateau` on `val_loss_smooth`, `EarlyStopping` on
  `val_loss_smooth`, `StochasticWeightAveraging`, gradient clip 1.0.
  `BaseModel.on_validation_epoch_end` logs `val_loss_smooth` (EMA of `val_loss`) for
  every model type.
- Default precision `bf16-mixed`.
- GPU strategy `ddp_find_unused_parameters_false`. **Any parameter without a
  gradient in a step crashes multi-GPU training.**
- Fine-tuning (`fine_tune_from`, which sets `train_continue`) uses differential LR:
  `swinViT` ×0.01, `encoder*` ×0.1, `decoder*` ×1.0, and every other trainable
  parameter (`out`, SSL heads) ×1.0 (`BaseModel._get_model_parameters_for_optimizer`).
  Training from scratch puts every trainable parameter in one group.
- Resume (`resume_from`) keeps the checkpoint's `train_continue`. A run whose saved
  optimizer predates the `out`-layer fix resumes with its original groups.
- Distillation (`networks/distillationModel.py`): frozen teacher; temperature BCE
  on logits plus MSE on `hidden_states_out[4]`.

### 1.7 Self-supervised mode (`networks/selfSupervisedModel.py`)
- `train.mode = selfSupervised`. Chunks still need a particle diameter (the
  fixed-particle rescale runs).
- Two views per cube: `rot_rand` (rot90 by k·90° in the (H, W) = (Y, X) plane,
  that is, about Z) then `aug_rand` (random blocks filled with min-max-scaled noise
  in [0, 1], plus pasted blocks from another sample).
- Heads on `hidden_states_out[4]` (2³ × 16fs): 4-way rotation (CE), contrastive
  (NT-Xent, temperature 0.1, plus L1 = `1e-4 · Σ|z|`). Both heads are optimised.
- Reconstruction: the decoder output, L1 over the **whole** volume, weight 10.
  TensorBoard `recons_*` shows the reconstruction of view 1.
- Picking can start from an SSL checkpoint (`training/helpers._create_model`, via the
  `RuntimeError` fallback when the checkpoint does not load as `BasePickingModel`).
- SSL training uses the same `Data` module, so the Section 1.2 augmentation (including
  contrast inversion) also runs on SSL inputs.

### 1.8 Inference (`predict/run.py`, `predict/helpers.py`)
- Dask cluster, one worker per GPU. `InferenceModel.load_from_checkpoint(strict=True)`.
- Target size comes from the **checkpoint** config
  (`model.hparams.config.prepData.desired_particle_pixel_size`).
- The tomogram is rescaled with `infer.length`, then predicted with `GridSampler`
  64³, overlap `64 // PATCH_OVERLAP_FACTOR` = 16, `GridAggregator(overlap_mode="hann")`,
  optional flip TTA. The map is resized back to the native shape. Peaks are found
  at the native voxel size with radius `length / (2·apix_native)`
  (`_extract_peaks_from_mask`, `peak_local_max_torch`).
- Output formats: `warp`, `relion_31`, `relion_50`, `imod`.

### 1.9 Checkpoints
- Lightning saves the full `MainConfig` dataclass (pickled) in `hyper_parameters`.
- `BaseModel.__init__` first calls `configManager.backfill_missing_fields(config,
  _mainConfigNoChanges)`, which adds current defaults for fields that an older
  checkpoint lacks, then merges with `update_config(...)`. Because of this, **each gate
  adds its own config fields**; there is no up-front schema.
- An unpickled checkpoint config lacks newer fields in its instance `__dict__`, while
  plain `getattr` still finds the class default. `backfill_missing_fields` checks the
  instance (fixed at Gate 1), so after `BaseModel.__init__` every field is present.
  Code that reads a checkpoint config **before** that point (raw `torch.load`) reads a
  gate's new fields with `safe_getattr(cfg, name, <current default>)`, never with
  `hasattr` or bare `getattr`. `tests/gate1/test_g1_backfill_instance_fields.py` sweeps
  every `MainConfig` field, so a new field is covered automatically.

### 1.10 Known legacy defects (do not "fix" them outside the named gate)
| Defect | Fixed in |
|---|---|
| Border particles are skipped and become negatives | Gate 2 (`label_border_policy`, v2) |
| Soft targets resampled with nearest-neighbour in augmentation | Gate 2 (pipeline v2 only) |
| Downsampling uses trilinear interpolation without anti-aliasing | Gate 2 (v2 `resample=fourier`) |
| Several species of one tomogram are separate datasets, each rescaled differently, and each species is a negative for the others | Gate 2 (one multi-class dataset per tomogram set) |
| SSL 4-way rotation task is ill-posed with a missing wedge (see Gate 5 spec) | Gate 5 (SSL v2) |
| SSL chunks need a particle size and see one scale | Gate 5 (SSL v2) |
| Chunking workers read a fresh default `mainConfig`, ignoring non-default `desired_particle_pixel_size`, `ALPHA_FOR_DROPPING_EMPTY_CUBES`, `CHUNK_SIZE`, `CHUNK_STRIDE` | Gate 2 (T2.6, v1 and v2) |
| v1 label is resized straight to the padded tomogram shape, so it is misaligned on padded axes (only when an axis is < 64 after rescale) | not fixed (v2 points path does not have it) |
| v1 empty-chunk dropping is unseeded (`random.random()`) | not fixed |
| v1 inference un-pads with the padding tuple in `F.pad` order (x, y, z) read as (z, y, x): thin tomograms that need Z padding are cropped on the wrong axis and coordinates shift | Gate 4 (v1 and v2; human decision 27 Oct 2026, C1 golden made with the fix) |
| `imodmodel==0.0.13` has no `write`, so IMOD output wrote nothing | Gate 4 (pin `imodmodel==0.1.0`, human-approved) |
| `pyproject.toml` pins `numpy==1.26.4`, `scipy==1.14.1`; the Fourier code was developed on newer versions | Gate 1 (run the suite on the pins) |

---

## 2. Goal and non-goals

**Goal.** One model and one `tomocpt predict` run pick several particle species of
different sizes and microtubules, from tomograms of different voxel sizes.
- **Data (Gate 2).** No fixed-particle rescale. Each tomogram is resampled only as far as
  needed to keep every class inside a band of voxel diameters (default 10–32), at one or
  a few working voxel sizes. All species of a tomogram are labelled together.
- **Model (Gate 3).** One output channel per class. FiLM conditioning on the working
  voxel size, so a class trained at several voxel sizes knows its physical scale.
- **Inference (Gate 4).** The plan of working voxel sizes comes from the checkpoint's
  class table and the tomogram's voxel size; each class is read at the size that fits it.
- **Self-supervised pretraining (Gate 5).** Wedge-aware, scale-aware SSL with dense
  embeddings gives the picker a better starting point, especially with few labels.
- **Filaments (Gate 6).** Microtubules as centreline channels, traced to polylines.

Legacy behaviour and legacy checkpoints keep working unchanged throughout.

**Non-goals.** Segmentation of membranes, organelles or void. Mixture-of-experts models.
Selecting a species at inference from its diameter alone (a class is a trained channel).
Objects outside the model's class table: the model picks only the classes it was trained on.
Handling the open-ended set of objects in a cell needs new ideas and is an open research
question, not part of this plan.

---

## 3. Definitions

| Term | Meaning |
|---|---|
| `apix_native` | Voxel size in Å from the MRC header (`mrc.voxel_size.x`), or `apix_override`. |
| `apix_work` | Voxel size in Å of the array the network sees. |
| `d_vox` | Diameter in working voxels: `diameter_ang / apix_work`. |
| class table | Ordered list of `ClassSpec(number, name, diameter_ang, kind)`; `kind` is `particle` or `filament`. Channel `c` = row `c`. Particles first, then filaments. Stored in `chunk_layout.json` and in the checkpoint config. |
| band | `[band_min_vox, band_max_vox]`, default `[10, 32]`. |
| level | One working voxel size used for a tomogram, with the classes assigned to it. |
| pipeline v1 | Legacy data pipeline (Section 1.2). `train.data_pipeline_version = 1`. |
| pipeline v2 | Band scale policy, multi-class points. `train.data_pipeline_version = 2`. |

Band rule (v2), for the class diameters `L` of one tomogram. The working voxel size is
**never finer than native**: upsampling adds no information and costs compute, so a class
below the band at native size is used at native size with `d_vox < band_min`.
```
L_eff     = max(L, apix_native · band_min)     # per class; a class under the band counts as at its lower edge
apix_work = max(apix_native, max(L) / band_max)  # valid when max(L_eff) / min(L_eff) <= band_max / band_min
```
If `apix_native` already puts every class inside the band (or below it), nothing is resampled.
If `max(L_eff)/min(L_eff) > band_max/band_min`, no single value fits:
`dataManager.scaling.group_lengths` splits the classes into groups, one level each. A class is
**assigned** to exactly one level (its group) and is read there at inference.
Per class, the lower limit is `lo_c = min(band_min, diameter / apix_native)`. During training a
class is available at every level where `lo_c <= d_vox <= band_max`.

---

## 4. Invariants (never break)

1. **Legacy equivalence.** With all new config fields at their defaults, training
   loss, chunk files, inference maps, and coordinates are identical to `v2-gate-0-base`
   (tolerance in each gate spec). Every new behaviour is opt-in.
2. **Checkpoint compatibility.** Every checkpoint written by `v2-gate-0-base` code
   (picking and SSL) loads for `train --fine_tune_from`, `train --resume_from`,
   distillation teacher, and `predict`, and behaves as v1.
3. **Config isolation (Issue #14).** New library code takes values as arguments.
   It does not read `mainConfig` globals. Inference reads training-time settings
   from the checkpoint (`model.hparams.config.train...`), never from `mainConfig.train`.
   Never access one section through another (no `cfg.train.prepData`).
4. **Chunk format is versioned.** Pipeline v2 adds `"pipeline_version": 2` and the class
   table to `chunks_dir/chunk_layout.json` (one manifest, no second file). Training
   refuses chunks whose `pipeline_version` (missing = 1) does not match the request.
5. **DDP safety.** When a feature is enabled, all its parameters receive a gradient
   (zero is allowed) in every training step. When a feature is disabled, do not
   create its modules.
6. **Zero-init for new paths.** FiLM layers start at zero, so a conditioned model loaded
   from an unconditioned checkpoint gives the unconditioned output.
7. **Output contract.** Model `forward` returns `(logits, hidden_states_out[4])`.
   `logits` are pre-sigmoid, `[B, C, D, H, W]`, channel `c` = class-table row `c`
   (particles, then filaments). With one class, channel 0 is the legacy heatmap.
   Never encode the class in the peak height.
8. **Dependencies.** Use only the packages pinned in `pyproject.toml`
   (torch 2.5.1, MONAI 1.4.0, Kornia 0.7.4, TorchIO 0.20.2, Lightning 2.4.0,
   scikit-image 0.24.0, scikit-learn 1.6.0, scipy 1.14.1, imodmodel 0.1.0 (from Gate 4), and the others; the `portal`
   extra for portal access). Any new dependency needs human approval.
9. **Tests are frozen.** You never edit a test file that the orchestrator froze.
   The tests that exist at `v2-gate-0-base` (`tests/test_*.py`) are frozen from the start
   and run in every gate.
10. **Targets stay `tio.LabelMap`.** Intensity and Fourier augmentations rely on TorchIO
    skipping label maps. For smooth target resampling in pipeline v2, set
    `label_interpolation="linear"` on the spatial transforms; never turn the target
    into a `tio.ScalarImage`.

---

## 5. Shared tensor contracts

### 5.1 Training batch (pipeline v2; v1 keeps only the first two keys)
| Key | Shape | dtype | Notes |
|---|---|---|---|
| `input_data` | `[B,1,64,64,64]` | float32 | normalised volume |
| `target_data` | `[B,C,64,64,64]` | float32 | soft target per class, values in [0,1] |
| `class_available` | `[B,C]` | bool | class annotated exhaustively in this sample **and** inside the band at this level |
| `class_d_vox` | `[B,C]` | float32 | class diameter in working voxels |
| `apix_work` | `[B,1]` | float32 | Å per working voxel (after zoom augmentation) |
| `dataset_id` | `[B]` | int64 | index of the `prepData` dataset |

Do not rename `input_data` or `target_data`.

### 5.2 Target storage (pipeline v2)
Chunks store targets as geometry, not dense volumes:
- `points/<chunk>.npy`, float32 `[N, 5]`: `z, y, x, class_index, d_vox` in chunk voxels;
- `filaments/<chunk>.npy` (Gate 6), float32 `[M, 8]`: segment `z0, y0, x0, z1, y1, x1,
  class_index, d_vox` in chunk voxels.
The dataloader draws dense targets with `labels.helpers.render_targets` before the
augmentation. The GPU always sees dense `[B, C, D, H, W]`; do not use `torch.sparse`.

### 5.3 Model signatures
- Legacy: `MySwinUNETR.forward(x) -> (logits, hidden4)`.
- Gate 3: `ScaleConditionedSwinUNETR.forward(x, apix=None) -> (logits, hidden4)`.
  `apix=None` gives exactly the unconditioned computation.
- Scale embedding input: `t = log(apix / 10.0)`. At 10 Å/px, `t = 0`.

### 5.4 Picking loss in pipeline v2
Section 1.5 per class `c`, with per-sample **and per-class** weights, computed only where
`class_available[b, c]`, Hessian scale-normalised per sample and class:
`H_n(v) = (class_d_vox[b, c] / 20)² · H(v)`. At `d_vox = 20` and one class this equals v1.
The total is the mean over available `(b, c)` pairs. The Hessian term runs one class at a
time (gradient checkpointing). Pipeline v1 uses Section 1.5 unchanged.

---

## 6. Config additions (each gate adds its own; all defaults reproduce v1)

| Path | Type / default | Gate |
|---|---|---|
| `prepData.apix_override` | Optional[float] = `None` | 2 |
| `prepData.label_border_policy` | `skip` \| `clip` = `skip` | 2 |
| `prepData.multiclass` | bool = `False` | 2 |
| `prepData.class_diameters` | Optional[str] = `None` (`"1:120,2:130"`, Å) | 2 |
| `prepData.class_names` | Optional[str] = `None` (`"1:apo-ferritin,..."`) | 2 |
| `prepData.class_annotated` | Optional[str] = `None` (classes annotated exhaustively; default all present) | 2 |
| `train.data_pipeline_version` | int = `1` | 2 |
| `train.scale` | `ScaleConfig` (`defaultConfigs/scale_config.py`) | 2 |
| `train.augmentation.label_interpolation` | `nearest` \| `linear` = `nearest` | 2 |
| `train.augmentation.rotation_mode` | `affine` \| `flips` = `affine` | 2 |
| `train.augmentation.flip_p` | float = `0.0` | 2 |
| `train.chunk_sampling` | `uniform` \| `repeat_factor` = `uniform` | 2 |
| `train.repeat_factor_threshold` | float = `0.1` | 2 |
| `train.network.scale_conditioning` | bool = `False` | 3 |
| `train.particles` | `ParticlesConfig` (`defaultConfigs/particles_config.py`) | 3 |
| `train.particles.cross_class_neg_weight` | float = `0.0` (off) | 3b |
| `infer.apix_override` | Optional[float] = `None` | 4 |
| `infer.classes` | Optional[str] = `None` (subset of class names) | 4 |
| `infer.cross_class_nms` | bool = `True` | 4 |
| `infer.accumulate_dtype`, `infer.accumulate_device` | `float32`, `auto` | 4 |
| `infer.class_thresholds` | Optional[str] = `None` (`"ribosome:0.4,..."`; overrides calibration) | 4 |
| `infer.calibration_beta`, `infer.calibration_min_particles` | float = `4.0`, int = `20` | 4 |
| `train.ssl` | `SSLConfig` (`defaultConfigs/ssl_config.py`) | 5 |
| `prepData.filament_masks` | Optional[str] = `None` (`"<class number>:<mask dir>"`) | 6 |
| `infer.filament_spacing_ang`, `infer.filament_min_length_ang` | float = `40.0`, `200.0` | 6 |

Field details are in each gate spec. A gate may add fields only to the files it owns.

---

## 7. Gate map and file ownership

| Gate | Agent spec | Purpose | May write (tests are written by TA) |
|---|---|---|---|
| 1 | `01_AGENT_BASELINE.md` | Lock legacy numbers and checkpoint loading; metrics | `tomocpt/configManager/configManager.py` (`safe_getattr` only), `tomocpt/metrics.py`, `benchmarks/__init__.py`, `pyproject.toml` (pytest section), `.gitignore` |
| 2 | `02_AGENT_SCALE_DATA.md` | Pipeline v2: band rule, levels, multi-class points, augmentation | `tomocpt/dataManager/**`, `tomocpt/labels/**`, `tomocpt/defaultConfigs/{scale_config,prepdata_config}.py`, `tomocpt/defaultConfigs/train_config.py` (`data_pipeline_version`, `scale`, `chunk_sampling`, `repeat_factor_threshold`, `AugmentationConfig` fields), `tomocpt/training/helpers.py` (`_prepare_data_if_needed` only), `tomocpt/training/run.py` (`Data(...)` arguments only) |
| 3 | `03_AGENT_MULTICLASS_PICKER.md` | Multi-class heads, FiLM on voxel size, v2 loss | `tomocpt/networks/{swinunetr,conditioning,pickingModel,distillationModel,inference,baseModel}.py`, `tomocpt/training/**` (except `_prepare_data_if_needed`), `tomocpt/defaultConfigs/{network_config,particles_config}.py` |
| 3b | `03b_AGENT_CROSS_CLASS_NEG.md` | Cross-class negative weighting in the v2 loss (human decision, 27 Oct 2026) | `tomocpt/training/multiclass_loss.py`, `tomocpt/networks/pickingModel.py` (passing the value only), `tomocpt/defaultConfigs/particles_config.py` |
| 4 | `04_AGENT_INFER_BENCHMARK.md` | v2 inference with levels, per-class outputs, benchmarks | `tomocpt/predict/**`, `tomocpt/defaultConfigs/infer_config.py`, `tomocpt/training/run.py` (post-fit calibration call only), `benchmarks/**`, `README.md` |
| 5 | `05_AGENT_SSL.md` | Wedge-aware, scale-aware SSL v2 with dense embeddings | `tomocpt/networks/selfSupervisedModel.py`, `tomocpt/networks/ssl/**`, `tomocpt/dataManager/ssl_views.py`, `tomocpt/defaultConfigs/ssl_config.py`, `tomocpt/training/helpers.py` (self-supervised branches only), `benchmarks/ssl_*.py` |
| 6 | `06_AGENT_FILAMENTS.md` | Microtubules: mask → centrelines, tube targets, tracing, output | `tomocpt/labels/**`, `tomocpt/dataManager/**`, `tomocpt/predict/**`, `tomocpt/defaultConfigs/{prepdata_config,infer_config}.py`, `scripts/download_portal_data.py`, `benchmarks/filaments.py`, `README.md` |

Every gate also writes `.claude/reports/gate_0N_report.json`.
Tests live in `tests/gateN/` and are written by the Test Author
(`TA_TEST_AUTHOR.md`) **before** the implementer starts. Shared test helpers live
in `tests/utils/` (written at Gate 1, frozen after Gate 1).

---

## 8. Rules for every agent

0. **Branches.** All v2 work (features, fixes, plans) lives on branches whose names start
   with `v2-`. The integration branch is `v2-dev`; side branches are `v2-feat/<name>`,
   `v2-fix/<name>`, `v2-plan/<name>` and merge into `v2-dev`. Tags start with `v2-`
   (`v2-gate-0-base`, `v2-gate-N-passed`). Never commit to, merge into or push `main`:
   it is the production branch. Moving v2 to `main` is a human decision.

1. Read this file, `00_ORCHESTRATOR.md`, your own spec, and all earlier gate reports.
2. Write only inside your paths in Section 7. If you need a change elsewhere, stop
   and write `.claude/reports/gate_0N_escalation.md` (what, where, why).
3. Do not edit frozen tests. If you think a test is wrong, write
   `.claude/reports/gate_0N_test_dispute.md` with evidence. The orchestrator decides.
4. Run `pytest tests -m "not slow"` before you report. Run `pytest tests/gateN -m slow`
   (a CUDA GPU is available on the development machine); report if it is not.
5. If legacy code disagrees with Section 1, trust the code and escalate.
6. Report with the schema in `00_ORCHESTRATOR.md`. Report failures honestly. Do not
   weaken a check to make it pass.
7. Keep public names exactly as the specs give them. Tests import them.
