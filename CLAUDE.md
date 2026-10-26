# CLAUDE.md — TomoCPT scale-free upgrade: engineering contract

This file is the shared contract for every agent that works on the TomoCPT
scale-free upgrade. Read all of it before you write code.

Precedence when two sources disagree:
1. A human instruction in the conversation.
2. The legacy code at tag `gate-0-base` for **legacy behaviour**. `gate-0-base` is
   `main` after `feat/fourier-augmentations` is merged; the orchestrator records the
   commit in `.claude/state/gate_status.json`.
   Section 1 records that behaviour. If Section 1 and the code disagree, the code
   is correct. Stop and write an escalation report (see Section 8).
3. This file.
4. Your agent spec in `.claude/agents/`.

---

## 1. Repository facts (verified at `gate-0-base`)

### 1.1 CLI and config
- Entry point: `tomocpt/main.py`. Commands: `tomocpt init`, `tomocpt prepare_data`,
  `tomocpt train`, `tomocpt predict`. There is **no** `tomocpt infer` command.
  `predict` reads the config section `infer`.
- Config is a tree of Python dataclasses in `tomocpt/defaultConfigs/`, rooted at
  `MainConfig` in `tomocpt/mainConfig.py`:
  - `prepData: PrepdataConfig`
  - `train: TrainConfig`, which contains `train.optimizer: OptimizerConfig` and
    `train.network: NetworkConfig` (which contains `UNET` and `SwinUNETR` sub-configs)
  - `infer: InferConfig`
  - `shared: GlobalPropertyConfig` (copied into sections in `MainConfig.__post_init__`)
- `configManager/configManager.py` builds the Typer CLI from these dataclasses.
  Hydra-style dot overrides work (`optimizer.lr=1e-4`).
- A global singleton `mainConfig` exists. Legacy code reads it from many places
  (for example `dataManager/transforms.py` reads `mainConfig.prepData` and
  `mainConfig.train`).

### 1.2 Data pipeline (legacy = "pipeline v1")
1. `prepare_data` (`labels/run.py`, `labels/helpers.py`):
   - Reads STAR or IMOD coordinates. Writes one full-size label MRC per tomogram at
     the **native** voxel size, plus a tracking CSV (`tomogram_path`, `label_path`,
     `particle_diameter_angst`, `num_particles`).
   - It does **not** chunk.
2. `train` chunks the data (`training/helpers._prepare_data_if_needed` →
   `dataManager/chunking.do_chunking`) if `chunks_dir/done_labels_{True|False}.txt`
   is absent. If that file exists, chunking is skipped without any check.
3. Per tomogram (`dataManager/transforms.PreprocessTomogramd` →
   `dataManager/preprocessing.preprocess_tomogram`):
   - Robust normalisation: `clip((x - median) / (p95 - p5), -3, 3)`.
   - **Fixed-particle rescale**: trilinear resize so that the particle *diameter*
     becomes `prepData.desired_particle_pixel_size` (default 20) voxels. The resize
     is skipped if the native diameter is within 1 voxel of the target.
   - Pad to at least 64 per axis. The label is resized to the same shape (trilinear).
4. TorchIO `GridSampler`: chunk 64³, overlap 32 (stride 32). Empty chunks are dropped
   with probability `clip(1 - ALPHA * f1/f0, 0, 1)`. Chunks are `.nii.gz` files in
   `chunks_dir/{train,val}/<tomo>/volumes/` and `.../labels_supervised/`. No metadata
   is stored.
5. Train/val split: `train_test_split(test_size=0.2, random_state=42)` over
   tomograms (default). **One tomogram gives an empty train set and an error.**
6. `dataManager/dataloading.VolumeDatsetIO` returns
   `{"input_data": [1,64,64,64], "target_data": [1,64,64,64]}`.
7. Training augmentation is built by
   `dataManager/dataloading.build_training_transforms(cfg: AugmentationConfig)`
   (config `train.augmentation`, in `defaultConfigs/train_config.py`), in this order:
   1. `RandomAffine(degrees=45, p=0.8)` and `OneOf(RandomElasticDeformation, RandomBlur)`
      (the historical pipeline);
   2. `RandomFourierDegradation` (`dataManager/fourier_augmentations.py`): missing wedge,
      amplitude-spectrum randomisation, local blend. Off by default
      (`use_mw_aug`, `use_fourier_aug`). It must run after every spatial transform.
      It orients the wedge for `(z, y, x)` arrays and keeps the input mean and std;
   3. `RandomContrastInversion` at a fixed `p = 0.8` (`CONTRAST_INVERSION_PROB`,
      **always on, not configurable**), then optional intensity transforms
      (`dataManager/intensity_augmentations.py`), all off by default.
   Validation uses no transform. The target is a `tio.LabelMap`: spatial transforms
   resample it with **nearest-neighbour** interpolation, and every
   `tio.IntensityTransform` skips it. That second property is what keeps the
   Fourier and intensity augmentations off the target (see invariant 11).

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
  `dropout_path_rate=0.2`. MONAI defaults: `depths=(2,2,2,2)`,
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

  No Swin block runs at 2³. The MLP inside each block is MONAI `MLPBlock`
  (`linear1`, `fn`=GELU, `drop1`, `linear2`, `drop2`; in "swin" mode
  `drop2 is drop1`). It receives tokens shaped `[B, D, H, W, C]` after `norm2`.
- MONAI checkpoints `forward_part1` and `forward_part2` (which holds the MLP) with
  `use_reentrant=False`. In backward, the MLP **runs a second time**.
- Decoder output channels: `decoder5` 8fs at 4³, `decoder4` 4fs at 8³,
  `decoder3` 2fs at 16³, `decoder2` fs at 32³, `decoder1` fs at 64³, then `out`.

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

### 1.6 Training runtime (`training/helpers.py`)
- RAdam, lr 4e-4, `ReduceLROnPlateau` on `val_loss_smooth`, `EarlyStopping`,
  `StochasticWeightAveraging`, gradient clip 1.0.
- Default precision `bf16-mixed`.
- GPU strategy `ddp_find_unused_parameters_false`. **Any parameter without a
  gradient in a step crashes multi-GPU training.**
- Fine-tuning (`fine_tune_from`) uses differential LR: `swinViT` ×0.01,
  `encoder*` ×0.1, `decoder*` ×1.0.
- Distillation (`networks/distillationModel.py`): frozen teacher; temperature BCE
  on logits plus MSE on `hidden_states_out[4]`.

### 1.7 Self-supervised mode (`networks/selfSupervisedModel.py`)
- `train.mode = selfSupervised`. Chunks still need a particle diameter (the
  fixed-particle rescale runs).
- Two views per cube: `rot_rand` (rot90 by k·90° in the (H, W) = (Y, X) plane,
  that is, about Z) then `aug_rand` (random blocks filled with min-max-scaled noise
  in [0, 1], plus pasted blocks from another sample).
- Heads on `hidden_states_out[4]` (2³ × 16fs): 4-way rotation (CE), contrastive
  (NT-Xent, temperature 0.1, plus L1 = `1e-4 · Σ|z|`).
- Reconstruction: the decoder output, L1 over the **whole** volume, weight 10.
- Picking can start from an SSL checkpoint (`training/helpers._create_model`).
- SSL training uses the same `Data` module, so the training augmentation in Section 1.2
  (including contrast inversion at `p = 0.8`) also runs on SSL inputs.

### 1.8 Inference (`predict/run.py`, `predict/helpers.py`)
- Dask cluster, one worker per GPU. `InferenceModel.load_from_checkpoint(strict=True)`.
- Target size comes from the **checkpoint** config
  (`model.hparams.config.prepData.desired_particle_pixel_size`).
- The tomogram is rescaled with `infer.length`, then predicted with `GridSampler`
  64³, overlap `64 // PATCH_OVERLAP_FACTOR` = 16, `GridAggregator(overlap_mode="hann")`,
  optional flip TTA. The map is resized back to the native shape. Peaks are found
  at the native voxel size with radius `length / (2·apix_native)`.
- Output formats: `warp`, `relion_31`, `relion_50`, `imod`.

### 1.9 Checkpoints
- Lightning saves the full `MainConfig` dataclass (pickled) in `hyper_parameters`.
- `BaseModel.__init__` first calls `configManager.backfill_missing_fields(config,
  _mainConfigNoChanges)`, which adds current defaults for fields that an older
  checkpoint lacks, then merges with `update_config(...)`, which skips fields missing
  on the source. Without this, a new nested field built with `default_factory` would
  make every older checkpoint raise `AttributeError` (dataclass unpickling does not
  run `__init__`). Gate 1 adds tests that load a real older checkpoint.

### 1.10 Known legacy defects (do not "fix" them outside the named gate)
| Defect | Fixed in |
|---|---|
| `prepare_picking_imod` uses undefined `class_id` (NameError) | Gate 1 |
| Old checkpoints failed to load once new nested config fields existed | Fixed on `feat/fourier-augmentations`; Gate 1 adds tests |
| Help text says `desired_particle_pixel_size` is a radius; code uses diameter | Gate 1 (text only) |
| Border particles are skipped and become negatives | Gate 2 (flag) |
| Soft targets resampled with nearest-neighbour in augmentation | Gate 2 (pipeline v2 only) |
| Downsampling uses trilinear interpolation without anti-aliasing | Gate 2 (pipeline v2 option) |
| Fine-tune parameter groups (`swinunetr._get_model_parameters_for_optimizer`) put only `swinViT`, `encoder*`, `decoder*` into the optimizer; `out.*` is in no group, so the output layer is frozen during fine-tuning | Gate 3 (v2 and conditioned models; v1 needs a human decision) |
| SSL TensorBoard "recons" panel logs the input, not the reconstruction | Gate 6 |
| SSL 4-way rotation task is ill-posed with a missing wedge (see Gate 6 spec) | Gate 6 (SSL v2) |
| `config.yaml` has `infer.predictions_coord_format: relion`, which is not a valid enum value, so the file does not load | Human decision (`relion_31` or `relion_50`) |
| `pyproject.toml` pins `numpy==1.26.4`, `scipy==1.14.1`; the Fourier branch was developed on newer versions | Gate 1 (run the suite on the pins) |
| Contrast inversion is always on at `p = 0.8`, so inverted polarity is the majority | Human decision; Gate 1 makes the probability a config field with default 0.8 |

---

## 2. Goal and non-goals

**Goal, stage 1 (Gates 1–6).** Remove the requirement that every particle is rescaled
to one size. The network must pick particles whose diameter varies within a band of
voxel sizes, given the target diameter as an input. One backbone pass must serve several
target sizes. Several labelled species are picked by one model, one heatmap channel per
species (Gate 5). Wedge-aware self-supervised pretraining supports this (Gate 6).

**Goal, stage 2 (Gates 7–8).** One model, one inference run: particle centroids plus
probability maps for membranes, filaments, organelles and void. Trained from partial
labels (Gate 7). A mixture of experts with structure- and scale-specific operators
(Gate 8) is an experiment; it ships only if it beats two controls.

Legacy behaviour and legacy checkpoints must keep working unchanged throughout.

**Main hypothesis.** Size conditioning plus multi-scale data gives most of the picking
benefit. Labelled multi-task heads give the segmentation. MoE may add capacity where the
tasks compete.

**Non-goals.** Segmentation that "emerges" from router maps without labels.
"Discover all species" picking.

---

## 3. Definitions

| Term | Meaning |
|---|---|
| `apix_native` | Voxel size in Å from the MRC header (`mrc.voxel_size.x`). |
| `apix_work` | Voxel size in Å of the array the network sees. |
| `d_vox` | Target diameter in working voxels: `particle_length_ang / apix_work`. |
| pipeline v1 | Legacy data pipeline (Section 1.2). `train.data_pipeline_version = 1`. |
| pipeline v2 | New pipeline with a scale policy. `train.data_pipeline_version = 2`. |
| `fixed_particle` | Scale policy of v1: `d_vox` is about `desired_particle_pixel_size`. |
| `band` | Scale policy of v2: keep `d_vox` inside `[band_min, band_max]` (default [10, 32]) with the least resampling. |

Band rule (v2), for one or more lengths `L` in one tomogram:
```
apix_work = clip(apix_native, max(L) / band_max, min(L) / band_min)
```
If `apix_native` already puts every length inside the band, nothing is resampled.
Otherwise the result is the voxel size closest to native that fits the band. If
`max(L)/min(L) > band_max/band_min`, no single value fits: split the lengths into
groups (`dataManager.scaling.group_lengths`) and process each group separately.

---

## 4. Invariants (never break)

1. **Legacy equivalence.** With all new config fields at their defaults, training
   loss, chunk files, inference maps, and coordinates are identical to `gate-0-base`
   (tolerance in each gate spec). Every new behaviour is opt-in.
2. **Checkpoint compatibility.** Every checkpoint written by `gate-0-base` code
   loads for `train --fine-tune-from`, `train --resume-from`, distillation teacher,
   and `predict`, and behaves as v1.
3. **Config isolation (Issue #14).** New library code takes values as arguments.
   It does not read `mainConfig` globals. Inference reads training-time settings
   from the checkpoint (`model.hparams.config.train...`), never from `mainConfig.train`.
   Never access one section through another (no `cfg.train.prepData`).
4. **Chunk format is versioned.** Pipeline v2 writes `chunks_dir/chunk_format.json`.
   Training refuses chunks whose format does not match the requested pipeline version.
5. **DDP safety.** When a feature is enabled, all its parameters receive a gradient
   (zero is allowed) in every training step. When a feature is disabled, do not
   create its modules.
6. **Recompute safety.** Code that runs inside a checkpointed Swin block must not
   append to Python lists or change state that depends on how many times it runs.
   Overwrite per-forward statistics. Collect them right after `swinViT` returns.
7. **Output contract.** Model `forward` returns `(logits, hidden_states_out[4])`.
   `logits` are pre-sigmoid. Channels `0..P-1` are particle centroid heatmaps, one per
   particle class (`P = 1` unless Gate 5 multi-class is on, so channel 0 is the legacy
   heatmap). Gate 7 adds structure channels after them, in `train.tasks.classes` order.
   Never encode the class in the peak height.
8. **Dependencies.** Use only the packages pinned in `pyproject.toml`
   (torch 2.5.1, MONAI 1.4.0, Kornia 0.7.4, TorchIO 0.20.2, Lightning 2.4.0,
   einops 0.8.0, and the others). Add nothing that needs compilation. Any new
   dependency needs human approval.
9. **Zero-init for new paths.** New conditioning, FiLM, expert branches, and router
   conditioning start at zero, so a new model loaded from a legacy checkpoint gives
   the legacy output.
10. **Tests are frozen.** You never edit a test file that the orchestrator froze.
    The tests that exist at `gate-0-base` (`tests/test_*.py`) are frozen from the start
    and run in every gate.
11. **Targets stay `tio.LabelMap`.** Intensity and Fourier augmentations rely on TorchIO
    skipping label maps. To get smooth target resampling in pipeline v2, set
    `label_interpolation="linear"` on the spatial transforms; never turn the target
    into a `tio.ScalarImage`.

---

## 5. Shared tensor contracts

### 5.1 Training batch (pipeline v2; v1 keeps only the first two keys)
| Key | Shape | dtype | Notes |
|---|---|---|---|
| `input_data` | `[B,1,64,64,64]` | float32 | normalised volume |
| `target_data` | `[B,P,64,64,64]` | float32 | soft Gaussian target per particle class, values in [0,1]; `P = 1` before Gate 5 |
| `class_available` | `[B,P]` | bool | Gate 5: class annotated exhaustively in this sample |
| `class_d_vox` | `[B,P]` | float32 | Gate 5: class diameter in working voxels |
| `d_vox` | `[B,1]` | float32 | target diameter in working voxels |
| `apix_work` | `[B,1]` | float32 | Å per working voxel |
| `dataset_id` | `[B]` | int64 | index of the `prepData` dataset |

Do not rename `input_data` or `target_data`.

### 5.2 Model signatures
- Legacy: `MySwinUNETR.forward(x) -> (logits, hidden4)`.
- Gate 3: `SizeConditionedSwinUNETR.forward(x, d_vox=None) -> (logits, hidden4)`,
  plus `encode(x) -> hidden_states` and `decode(x, hidden_states, d_vox) -> logits`.
  `d_vox=None` gives exactly the legacy computation.
- Size embedding input: `t = log(d_vox / 20.0)`. At `d_vox = 20`, `t = 0`.
- Gate 7: `MultiStructureSwinUNETR.forward(x, d_vox=None, apix=None) -> (logits, hidden4)`,
  `logits [B, P+K, D, H, W]`; conditioning `ConditionEmbedding(d_vox, apix)` with
  `t_apix = log(apix / 10.0)`. Multi-structure batches add `seg_target [B,K,D,H,W]` float32
  and `seg_available [B,K]` bool.

### 5.2b Target storage (pipeline v2)
Chunks store particles as points (`points/<chunk>.npy`, `[N, 5]`: `z, y, x, class_index,
d_vox` in chunk voxels), not as dense label volumes. The dataloader draws the dense target
with `labels.helpers.render_targets` before the augmentation. The GPU always sees dense
`[B, P, D, H, W]` tensors; do not use `torch.sparse`.

### 5.3 Picking loss in pipeline v2
Same as Section 1.5, except the Hessian is scale-normalised per sample before the
Huber term: `H_n(v) = (d_vox / 20)² · H(v)`. At `d_vox = 20` this equals v1.
Pipeline v1 uses Section 1.5 unchanged. With several particle classes (Gate 5) the loss
is computed per class with per-class weights and `class_d_vox`, only for available
classes, and the Hessian term one class at a time (gradient checkpointing).

---

## 6. Config additions (Gate 1 adds them all, inert; later gates add behaviour)

| Path | Type / default | Owner gate |
|---|---|---|
| `prepData.label_border_policy` | `skip` \| `clip` = `skip` | 2 |
| `prepData.write_coordinates` | bool = `True` (additive CSV per tomogram) | 2 |
| `prepData.apix_override` | Optional[float] = `None` | 2 |
| `train.data_pipeline_version` | int = `1` | 2 |
| `train.scale` | `ScaleConfig` (file `defaultConfigs/scale_config.py`) | 2 |
| `train.augmentation.label_interpolation` | `nearest` \| `linear` = `nearest` | 2 |
| `train.augmentation.rotation_mode` | `affine` \| `flips` = `affine` | 2 |
| `train.augmentation.contrast_inversion_p` | float = `0.8` (replaces the constant) | 1 |
| `train.pu` | `PUConfig` (file `defaultConfigs/pu_config.py`) | 3 |
| `train.scale.target_storage` | `points` \| `dense` = `points` (v2 only) | 2 |
| `prepData.multiclass`, `class_diameters`, `class_names`, `class_annotated` | off / `None` | 5 |
| `train.particles` | `ParticlesConfig` (file `defaultConfigs/particles_config.py`) | 5 |
| `infer.cross_class_nms`, `accumulate_dtype`, `accumulate_device`, `classes` | `True`, `float32`, `auto`, `None` | 5 |
| `train.ssl` | `SSLConfig` (file `defaultConfigs/ssl_config.py`) | 6 |
| `train.network.size_conditioning` | bool = `False` | 3 |
| `train.tasks` | `TasksConfig` (file `defaultConfigs/tasks_config.py`) | 7 |
| `train.network.moe` | `MoEConfig` (file `defaultConfigs/moe_config.py`) | 8 |
| `infer.export_structures` | bool = `False` | 7 |
| `infer.lengths` | Optional[str] = `None` (comma-separated Å) | 4 |
| `infer.apix_override` | Optional[float] = `None` | 4 |

Field details are in each gate spec. All defaults reproduce v1.

---

## 7. Gate map and file ownership

| Gate | Agent spec | Purpose | May write (plus its own tests are written by TA) |
|---|---|---|---|
| 1 | `01_AGENT_BASELINE_CONFIG.md` | Lock legacy, safe config loading, inert config schema | `tomocpt/configManager/configManager.py`, `tomocpt/defaultConfigs/**`, `tomocpt/labels/helpers.py` (IMOD fix only), `tomocpt/dataManager/dataloading.py` (contrast-inversion probability from config only), `pyproject.toml` (pytest section), `.gitignore` |
| 2 | `02_AGENT_SCALE_DATA.md` | Pipeline v2: band scale policy, metadata, multi-scale chunks, augmentation | `tomocpt/dataManager/**`, `tomocpt/labels/**`, `tomocpt/defaultConfigs/scale_config.py`, `tomocpt/defaultConfigs/prepdata_config.py` (own fields), `tomocpt/defaultConfigs/train_config.py` (`AugmentationConfig` fields only), `tomocpt/training/helpers.py` (`_prepare_data_if_needed` only) |
| 3 | `03_AGENT_SIZE_CONDITIONED_MODEL.md` | FiLM size conditioning, v2 loss, optional PU | `tomocpt/networks/{swinunetr,conditioning,pickingModel,distillationModel,inference,baseModel}.py`, `tomocpt/training/**`, `tomocpt/defaultConfigs/{network_config,pu_config}.py` |
| 4 | `04_AGENT_INFER_BENCHMARK.md` | v2 inference, multi-length picking, benchmark harness | `tomocpt/predict/**`, `tomocpt/defaultConfigs/infer_config.py`, `benchmarks/**`, `README.md` |
| 5 | `05_AGENT_MULTICLASS_PARTICLES.md` | Several particle classes in one model: one heatmap channel per class, partial labels, per-class inference | `tomocpt/labels/**`, `tomocpt/dataManager/**`, `tomocpt/networks/{swinunetr,pickingModel,inference,distillationModel}.py`, `tomocpt/training/**`, `tomocpt/predict/**`, `tomocpt/defaultConfigs/{prepdata_config,particles_config,infer_config}.py`, `benchmarks/multiclass_czii.py`, `README.md` |
| 6 | `06_AGENT_SSL_WEDGE.md` | Wedge-aware SSL v2 with relative-scale task | `tomocpt/networks/selfSupervisedModel.py`, `tomocpt/networks/ssl/**`, `tomocpt/dataManager/ssl_views.py`, `tomocpt/defaultConfigs/ssl_config.py`, `tomocpt/training/helpers.py` (self-supervised branches only) |
| 7 | `07_AGENT_MULTI_STRUCTURE.md` | Particles + membrane, filament, organelle, void in one model; partial labels; two-resolution inference | see the spec (heads, seg losses, segmentation labels, chunk targets, predict outputs, portal masks) |
| 8 | `08_AGENT_MOE.md` | Structure- and scale-expert soft MoE experiment and decision | `tomocpt/networks/moe.py`, `tomocpt/defaultConfigs/moe_config.py`, `benchmarks/moe_ablation.py`; one call in `NetworkConfig.build_model`; auxiliary-loss lines in `pickingModel.py` and `distillationModel.py` |

Every gate also writes `.claude/reports/gate_0N_report.json`.
Tests live in `tests/gateN/` and are written by the Test Author
(`TA_TEST_AUTHOR.md`) **before** the implementer starts. Shared test helpers live
in `tests/utils/` (written at Gate 1, frozen after Gate 1).

---

## 8. Rules for every agent

1. Read this file, `00_ORCHESTRATOR.md`, your own spec, and all earlier gate reports.
2. Write only inside your paths in Section 7. If you need a change elsewhere, stop
   and write `.claude/reports/gate_0N_escalation.md` (what, where, why).
3. Do not edit frozen tests. If you think a test is wrong, write
   `.claude/reports/gate_0N_test_dispute.md` with evidence. The orchestrator decides.
4. Run `pytest tests/gate1 ... tests/gateN -m "not slow"` before you report. Run
   `-m slow` for your gate if a GPU is available; report if it is not.
5. If legacy code disagrees with Section 1, trust the code and escalate.
6. Report with the schema in `00_ORCHESTRATOR.md`. Report failures honestly. Do not
   weaken a check to make it pass.
7. Keep public names exactly as the specs give them. Tests import them.
