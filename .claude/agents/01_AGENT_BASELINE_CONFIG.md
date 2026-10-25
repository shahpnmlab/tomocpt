# 01_AGENT_BASELINE_CONFIG.md — Gate 1: legacy lock, safe config loading, inert schema

## Mission
Make the code base safe to extend. After this gate, new config fields exist but do
nothing, old checkpoints load, and the legacy numbers are locked by tests.
No numerical behaviour changes in this gate.

## Allowed write paths
- `tomocpt/configManager/configManager.py`
- `tomocpt/defaultConfigs/**` (new files below; edits to existing files only to add fields)
- `tomocpt/labels/helpers.py` (IMOD `class_id` fix only)
- `pyproject.toml` (`[tool.pytest.ini_options]` only: register marker `slow`)
- `.gitignore` (add `tests/.cache/`)
- `.claude/reports/gate_01_report.json`

## Tasks

### T1.1 Safe config merge (mostly done at `gate-0-base`)
`backfill_missing_fields` and the skip in `update_config` already exist (from
`feat/fourier-augmentations`). Check `compare_dataclasses` and
`update_config_with_changed_values` for bare `getattr` on fields that can be missing,
and guard them the same way. Keep all other behaviour.

Export `tomocpt.configManager.configManager.safe_getattr(obj, name, default)`.

### T1.1b Contrast-inversion probability in config
Add `AugmentationConfig.contrast_inversion_p: float = 0.8` and read it in
`build_training_transforms` instead of the module constant. Keep the constant as the
default value. Behaviour does not change. (The value is a human decision; see
CLAUDE.md Section 1.10.)

### T1.1c Dependency pins
Run the fast suite on the pinned `numpy==1.26.4` and `scipy==1.14.1`. Report any
failure; do not change the pins without a human decision.

### T1.2 Inert config schema
Add these dataclasses. Every default must reproduce v1 behaviour. Use `str` Enums
(Typer needs them) and `field(default_factory=...)` for nested dataclasses.

`defaultConfigs/scale_config.py`
```python
class ScalePolicy(str, Enum): fixed_particle = "fixed_particle"; band = "band"
class ResampleMethod(str, Enum): trilinear = "trilinear"; fourier = "fourier"
@dataclass
class ScaleConfig:
    policy: ScalePolicy = ScalePolicy.fixed_particle
    band_min_vox: float = 10.0
    band_max_vox: float = 32.0
    scales_per_tomogram: int = 1
    resample: ResampleMethod = ResampleMethod.trilinear
    zoom_aug_max: float = 1.0          # 1.0 = off; v2 suggestion 1.25
    size_balanced_sampling: bool = False
```

`defaultConfigs/pu_config.py`
```python
class PUMode(str, Enum): none = "none"; ema_teacher = "ema_teacher"
@dataclass
class PUConfig:
    mode: PUMode = PUMode.none
    ema_decay: float = 0.999
    start_epoch: int = 2
    prob_threshold: float = 0.5
    max_neg_reduction: float = 0.8
```

`defaultConfigs/ssl_config.py`
```python
@dataclass
class SSLConfig:
    version: int = 1
    use_rotation: bool = True            # v1 behaviour; Gate 5 sets v2 default False
    rot_weight: float = 1.0
    contrast_weight: float = 1.0
    recon_weight: float = 10.0
    recon_visible_weight: float = 1.0    # v1: whole-volume loss
    mask_fill: str = "noise01"           # v1; v2 uses "zero"
    scale_head: bool = False
    scale_weight: float = 1.0
    scale_range_log2: float = 1.0
    even_odd_suffixes: Optional[Tuple[str, str]] = None
    wedge_task: bool = False
    tilt_range_deg: float = 60.0
    use_mw_aug: bool = True              # used only when version == 2
    use_fourier_aug: bool = True         # used only when version == 2
    degradation_p: float = 0.8
    keep_angle_min: float = 30.0
    keep_angle_max: float = 60.0
    inversion_p: float = 0.5
```

`defaultConfigs/moe_config.py`
```python
@dataclass
class MoEConfig:
    enabled: bool = False
    stages: Tuple[str, ...] = ("layers2", "layers3")
    experts: Tuple[str, ...] = ("mlp", "dw_d1", "dw_d2", "dw_d3", "line", "sheet", "smooth")
    cond_dim: int = 64
    balance_weight: float = 0.01
    z_loss_weight: float = 1e-3
    router_temperature: float = 1.0
    route_supervision_weight: float = 0.0
```

`defaultConfigs/tasks_config.py`
```python
@dataclass
class TasksConfig:
    classes: Tuple[str, ...] = ()        # subset of ("membrane", "filament", "organelle", "void")
    class_weights: Tuple[float, ...] = ()
    context_apix: float = 40.0
    context_classes: Tuple[str, ...] = ("organelle", "void")
    fine_apix: Optional[float] = None    # None: use the particle band rule
```

Add fields (CLAUDE.md Section 6):
- `PrepdataConfig`: `label_border_policy: str = "skip"`, `write_coordinates: bool = True`,
  `apix_override: Optional[float] = None`.
- `TrainConfig`: `data_pipeline_version: int = 1`, `scale: ScaleConfig`, `pu: PUConfig`,
  `ssl: SSLConfig`, `tasks: TasksConfig`.
- `NetworkConfig`: `size_conditioning: bool = False`, `moe: MoEConfig`.
- `AugmentationConfig` (existing, `train_config.py`): `label_interpolation: str = "nearest"`,
  `rotation_mode: str = "affine"`, `flip_p: float = 0.0`, `contrast_inversion_p: float = 0.8`.
- `InferConfig`: `lengths: Optional[str] = None`, `apix_override: Optional[float] = None`,
  `export_structures: bool = False`.

Expose the new top-level scalar fields (`data_pipeline_version`, `label_border_policy`,
`write_coordinates`, `apix_override`, `size_conditioning`, `lengths`) to the CLI with
Typer `Annotated` help text, like the existing fields. `tomocpt init` writes only
`Annotated` fields; nested sub-configs follow the existing rule for `optimizer` and
`network`. Do not change that rule.

### T1.3 Small legacy fixes
- `labels/helpers.prepare_picking_imod`: pass the class id that the caller already
  uses (`"all"`) to `create_tracking_dataframe`. Do not change STAR behaviour.
- `PrepdataConfig.desired_particle_pixel_size` help text: "diameter", not "radius".

## Public names (tests import these)
- `tomocpt.configManager.configManager.update_config`, `safe_getattr`
- `tomocpt.defaultConfigs.scale_config.{ScaleConfig, ScalePolicy, ResampleMethod}`
- `tomocpt.defaultConfigs.pu_config.{PUConfig, PUMode}`
- `tomocpt.defaultConfigs.ssl_config.SSLConfig`
- `tomocpt.defaultConfigs.moe_config.MoEConfig`
- `tomocpt.defaultConfigs.tasks_config.TasksConfig`

## Pass criteria
- **G1-C1** Gaussian target golden: `generate_gaussian_sphere(R, 2R+1)` for R in
  {5, 10, 17} equals the `gate-0-base` golden (atol 1e-7).
- **G1-C2** Loss golden: `BasePickingModel.loss(ŷ, y)` on fixed random tensors
  (soft labels from G1-C1 placed in a 64³ volume, B=2) equals the golden (rtol 1e-6).
- **G1-C3** Forward golden: tiny `MySwinUNETR` (`tests/utils/tiny.py`), seed 0, eval,
  CPU fp32: logits and `hidden[4]` equal the golden (atol 1e-5).
- **G1-C4** Config isolation: `PrepdataConfig()`, `TrainConfig()`, `InferConfig()`
  instantiate alone, and every field path in CLAUDE.md Section 6 exists with the
  stated default. `MainConfig()` has no `train.prepData`.
- **G1-C5** Legacy checkpoint loads: `BasePickingModel.load_from_checkpoint` on
  `legacy_checkpoint()` succeeds; `InferenceModel.load_from_checkpoint(strict=True)`
  succeeds; the loaded model's logits equal the legacy model's logits (atol 1e-6).
  Also for `legacy_ssl_checkpoint()` with `SelfSupervisedModel`.
- **G1-C6** Distillation with a legacy teacher constructs and runs one training
  step on CPU.
- **G1-C7** `update_config` keeps target defaults for fields missing on the source, and
  `backfill_missing_fields` adds every field that a legacy checkpoint lacks.
- **G1-C11** `build_training_transforms(AugmentationConfig())` is transform-wise
  identical to the `gate-0-base` pipeline, and `contrast_inversion_p=0.0` removes the
  inversion transform (or makes it a no-op).
- **G1-C8** IMOD path: `prepare_picking_imod` runs on a tiny `.mod` fixture (built
  with `imodmodel`) without NameError and writes the tracking CSV.
- **G1-C9** `tomocpt init --output-path x.yaml` writes the new `Annotated` scalar fields
  with their defaults, and the file loads back with `--config-file`.
- **G1-C10** (optional, skip without env var) the Zenodo checkpoint loads in `predict`
  model loading.

## Report notes for the next gate
List every new config path and its default. State that no numerical code changed.
