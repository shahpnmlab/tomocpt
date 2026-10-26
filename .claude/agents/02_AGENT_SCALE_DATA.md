# 02_AGENT_SCALE_DATA.md — Gate 2: data pipeline v2 (band rule, levels, multi-class points)

## Mission
Produce training chunks where every species of a tomogram is labelled together, each
class keeps a diameter inside a band of voxel sizes, and the working voxel size is
recorded per chunk. Keep pipeline v1 byte-identical. No model or loss change here.

## Why
- In v1 every species becomes about 20 voxels wide, each in its own dataset. A 70 Å
  particle at 10 Å/px is upsampled ×2.86; a 500 Å particle loses detail; and in a
  tomogram with six species, each species is a negative for the other five.
- Pipeline v2 resamples only when a class is outside the band, puts all classes of a
  tomogram into one target with one channel per class, and records `apix_work`, which the
  Gate 3 model receives as its conditioning input.

## Allowed write paths
- `tomocpt/dataManager/**` (new: `scaling.py`, `metadata.py`)
- `tomocpt/labels/**` (new: `multiclass.py`)
- `tomocpt/defaultConfigs/scale_config.py` (new), `tomocpt/defaultConfigs/prepdata_config.py`
- `tomocpt/defaultConfigs/train_config.py`: `data_pipeline_version`, `scale`, and
  `AugmentationConfig` fields only
- `tomocpt/training/helpers.py`: `_prepare_data_if_needed` only
- `tomocpt/training/run.py`: the `Data(...)` arguments only
- `.claude/reports/gate_02_report.json`

## Config (this gate adds)
`defaultConfigs/scale_config.py`
```python
class ResampleMethod(str, Enum): trilinear = "trilinear"; fourier = "fourier"
@dataclass
class ScaleConfig:
    band_min_vox: float = 10.0
    band_max_vox: float = 32.0
    scales_per_level: int = 1          # k > 0 adds jittered voxel sizes per level
    resample: ResampleMethod = ResampleMethod.fourier   # used by v2 only
    zoom_aug_max: float = 1.0          # 1.0 = off; suggestion 1.25
    target_storage: str = "points"     # "points" or "dense" (debugging, comparison)
```
`PrepdataConfig`: `apix_override`, `label_border_policy`, `multiclass`, `class_diameters`,
`class_names`, `class_annotated` (CLAUDE.md Section 6).
`TrainConfig`: `data_pipeline_version: int = 1`, `scale: ScaleConfig`.
`AugmentationConfig`: `label_interpolation`, `rotation_mode`, `flip_p`.
Use `str` Enums (Typer needs them) and `field(default_factory=...)` for nested dataclasses.
Scalar fields get Typer `Annotated` help text like the existing fields.

## Tasks

### T2.1 Voxel size and scale rule (`dataManager/scaling.py`)
- `read_apix(mrc_path, override: Optional[float] = None) -> float`. Use `override` if
  given. Raise `ValueError` that names `apix_override` if the header value is `<= 0`.
  Log a warning if it is exactly 1.0 (often an unset header).
- `choose_working_apix(apix_native, lengths_ang, band_min, band_max) -> float`: the band
  rule (CLAUDE.md Section 3). Raise `ValueError` if no value fits (use `group_lengths`).
- `group_lengths(apix_native, lengths_ang, band_min, band_max) -> list[tuple[float, list[float]]]`:
  sort lengths; greedily grow a group while `max/min <= band_max/band_min`; return
  `(apix_work, lengths)` per group with the band rule.
- `plan_levels(apix_native, class_table, band_min, band_max) -> list[Level]`, where
  `Level = NamedTuple(apix_work: float, assigned: list[int], available: list[int])`:
  one level per group; `assigned` = class indices of that group; `available` = every class
  whose `d_vox` at that level is inside the band. Gate 4 uses the same function.
- `working_shape(native_shape_zyx, apix_native, apix_work) -> tuple[int, int, int]`:
  `int(s * apix_native / apix_work)` per axis (the v1 rounding).
- `resample_volume(vol: Tensor[D,H,W], target_shape, method: ResampleMethod) -> Tensor`.
  `trilinear` calls the existing `preprocessing.resize_volume` (unchanged). `fourier`
  crops or zero-pads the centred spectrum (`torch.fft.fftn` / `ifftn`), keeps the mean,
  returns a real tensor, and handles odd and even sizes. Fourier cropping is the
  anti-aliasing that v1 lacks.

### T2.2 Multi-class labels (`labels/**`)
- `labels/multiclass.py`:
  - `ClassSpec = dataclass(number: int, name: str, diameter_ang: float, kind: str = "particle")`.
  - `parse_class_diameters(text) -> dict[int, float]`, `parse_class_names(text) -> dict[int, str]`:
    tolerate spaces; a duplicate class or a malformed entry raises `ValueError` naming it.
  - `build_class_table(diameters, names, kinds=None) -> list[ClassSpec]`: particles sorted
    by class number, then filaments sorted by class number. Channel `c` = row `c`.
  - `class_table_from_config(prep_cfg) -> list[ClassSpec]`: the **only** way to obtain the
    table. Multi-class: from `class_diameters` / `class_names`. Single-class: one row
    (number 1, name `class_id`, diameter `particle_length_ang`). Gate 3 builds the model
    and Gate 4 plans inference from it, using the `prepData` saved in the checkpoint.
    Training raises `ValueError` if it differs from the table in `chunk_layout.json`.
- `prepare_data` with `prepData.multiclass = True`: one dataset holds all species.
  `coordinate_files` are STAR files with `rlnClassNumber` (for example
  `coords/all_objects.star` from `scripts/download_portal_data.py`); `class_diameters` is
  required. Classes in the STAR file without a diameter raise `ValueError`.
  `class_annotated` (default: every class present) lists the exhaustively annotated
  classes; the others get `class_available = False` in every chunk of that dataset.
- For every tomogram (single-class and multi-class), when pipeline v2 data are prepared,
  also write `<dataset_dir>/coords/<tomo>.csv` with columns
  `x, y, z, class_number, diameter_ang` (native voxels, the values used to draw the v1
  labels) and add columns `coords_path`, `apix_native` to the tracking CSV. Existing
  columns and label MRCs stay identical. Single-class STAR data get the class number from
  `prepData.class_id` (`"all"` maps to 1) and the dataset diameter.
- `render_targets(points, out_shape_zyx, n_classes, border_policy="clip") -> np.ndarray[float32]`
  of shape `[n_classes, *out_shape_zyx]`. `points` is `[N, 5]`
  (`z, y, x, class_index, d_vox`) in voxels of the output array; float positions allowed.
  Draws the CLAUDE.md Section 1.3 profile with float radius `R = d_vox / 2` centred at the
  float position (not rounded), max-combined per channel. `"clip"` draws the in-bounds
  part; `"skip"` drops particles whose box touches the array border (the v1 rule).
- `points_to_working(coords_df, apix_native, apix_work, class_table) -> np.ndarray[N, 5]`.
- `prepData.label_border_policy` (`skip` default) also applies to v1 label MRCs.

### T2.3 Chunking v2 (`dataManager/chunking.py`, `transforms.py`, `metadata.py`)
- New keyword arguments on `do_chunking`: `pipeline_version: int = 1`,
  `scale_cfg: Optional[ScaleConfig] = None`, `class_table: Optional[list] = None`,
  `seed: int = 42`. Defaults run the v1 code path unchanged. Pass values explicitly;
  new code does not read `mainConfig`.
- Rows: in v2, rows of the tracking CSVs that refer to the same tomogram file are merged
  into one task (their points concatenated), so a tomogram is chunked once with all its
  classes. Reuse `assign_chunk_dir_names` and `split_rows_by_tomogram`.
- Per tomogram, `plan_levels(...)` gives the levels. Per level and scale index `k`
  (`k = 0 … scales_per_level-1`): `k = 0` uses the level's `apix_work`; `k > 0` draws a
  voxel size log-uniformly in the range that keeps the level's `assigned` classes inside
  the band (seeded by `seed`, tomogram name, level). Normalise as v1 at native size,
  resample with `scale_cfg.resample`, pad to ≥ 64 (as v1).
- Targets are stored as points (`target_storage = "points"`): convert coordinates with
  `points_to_working`, shift by the padding, apply `label_border_policy` once at tomogram
  level, then per chunk save `points/<chunk stem>.npy` (CLAUDE.md Section 5.2) for every
  particle whose sphere overlaps the chunk (centre within `[-R, 64 + R)` on each axis).
  `target_storage = "dense"` writes dense multi-channel label chunks instead.
- Grid sample as v1. Drop empty chunks with the v1 probability, "empty" meaning no point
  of an available class inside the chunk.
- Output directory `<split>/<dir name>__L{level}s{k}/` with `volumes/`, `points/` and
  `chunk_meta.json` (`ChunkMeta`).
- Manifest: v2 writes `chunk_layout.json` with the v1 keys plus `"pipeline_version": 2`,
  `"band"`, `"resample"`, `"scales_per_level"` and `"class_table"`. Training with
  `data_pipeline_version = 2` on a manifest without `pipeline_version: 2` raises
  `ValueError("chunk format ...")`; v1 training on a v2 manifest raises the same.
  The done file is still written last.

`metadata.py`:
```python
@dataclass
class ChunkMeta:
    tomo_name: str; dataset_id: int; level: int; scale_index: int
    apix_native: float; apix_work: float
    class_available: list[bool]; class_d_vox: list[float]
    pipeline_version: int; resample: str
def write_chunk_meta(dir, meta) / read_chunk_meta(dir) -> ChunkMeta
def write_chunk_layout(chunks_dir, info: dict) / read_chunk_layout(chunks_dir) -> Optional[dict]
```

### T2.4 Dataset and augmentation v2 (`dataManager/dataloading.py`)
- Extend `build_training_transforms(cfg: AugmentationConfig)`. Keep its order
  (spatial → Fourier → intensity) and its defaults (v1 unchanged).
  - `label_interpolation="linear"` passes `label_interpolation="linear"` to `RandomAffine`
    and `RandomElasticDeformation`. The target **stays a `tio.LabelMap`** (invariant 10).
  - `rotation_mode="flips"` replaces `RandomAffine` with
    `RandomFlip(axes=(0,1,2), flip_probability=flip_p)`. Axis flips keep the real missing
    wedge where it is; `RandomFourierDegradation` then adds its oriented wedge as before.
  - Recommended v2 settings (not defaults): `label_interpolation="linear"`,
    `rotation_mode="flips"`, `flip_p=0.5`.
- Zoom augmentation (`zoom_aug_max > 1`): draw `s ~ U(1, zoom_aug_max)`, zoom the volume
  by `s` (trilinear), crop a random 64³, scale the point positions and `d_vox` by `s`,
  divide `apix_work` by `s`, then draw the target. Mark a class unavailable if its zoomed
  `d_vox` leaves the band. Zoom runs **before** `build_training_transforms`.
- `VolumeDatsetIO` v2 reads the volume chunk, its points file and `chunk_meta.json`, draws
  the target with `render_targets(points, (64, 64, 64), C, "clip")` in the worker, wraps it
  as a `tio.LabelMap`, applies the transforms, and returns the CLAUDE.md Section 5.1 batch.
  v1 output unchanged.
- `Data` gets `pipeline_version` and `scale_cfg` arguments; `training/run.py` passes them
  from the train config.

## Public names
`tomocpt.dataManager.scaling.{read_apix, choose_working_apix, group_lengths, plan_levels, Level, working_shape, resample_volume}`,
`tomocpt.dataManager.metadata.{ChunkMeta, write_chunk_meta, read_chunk_meta, write_chunk_layout, read_chunk_layout}`,
`tomocpt.labels.multiclass.{ClassSpec, parse_class_diameters, parse_class_names, build_class_table, class_table_from_config}`,
`tomocpt.labels.helpers.{render_targets, points_to_working}`,
`tomocpt.dataManager.dataloading.{build_training_transforms, VolumeDatsetIO, Data}`,
`tomocpt.dataManager.chunking.do_chunking` (new kwargs),
`tomocpt.defaultConfigs.scale_config.{ScaleConfig, ResampleMethod}`.

## Pass criteria
- **G2-C1** v1 equivalence: `do_chunking` with defaults on `make_dataset(n_tomos=2)`
  gives the same file list and identical decoded arrays as the `gate-0-base` golden.
- **G2-C2** Band rule table: (apix 10, L [70]) → 7.0; (10, [250]) → 10.0;
  (5, [350]) → 10.9375; (14, [140]) → 14.0; group (10, [120, 300]) → one group at 10.0;
  group (10, [70, 500]) → two groups. CZII diameters at 10.012 Å/px → one level, no
  resampling.
- **G2-C3** `plan_levels`: every class is assigned to exactly one level; `available ⊇
  assigned`; every available class has `d_vox` inside the band at that level.
- **G2-C4** Fourier resampling: a cosine above the new Nyquist keeps < 1 % of its power;
  the centroid of a smooth blob moves < 0.1 voxel; the mean is kept (rtol 1e-4);
  round trip native → work → native moves a blob centroid < 0.25 native voxel.
- **G2-C5** Class table: parsing (spaces, missing diameter, duplicate class → `ValueError`
  naming it); particles before filaments; `class_table_from_config` gives the same table
  that `chunk_layout.json` records; a mismatch at training start raises `ValueError`.
- **G2-C6** Multi-class prepare_data: a STAR with three classes on two tomograms gives one
  dataset, coordinate CSVs with all classes, and `class_annotated` is respected.
- **G2-C7** `render_targets`: maximum within 1 voxel of each centre; target > 0 for
  `r < R - 0.5` and == 0 for `r > R + 1`; each particle appears only in its class channel;
  overlapping particles of different classes keep both peaks in their own channels.
- **G2-C8** v2 chunking: one task per tomogram even when several CSV rows share it;
  `chunk_layout.json` has `pipeline_version` 2 and the class table; one `chunk_meta.json`
  per directory; batches match CLAUDE.md Section 5.1 shapes and dtypes;
  `class_d_vox == diameter / apix_work`.
- **G2-C9** Points storage: for every chunk, `render_targets` from the stored points equals
  the crop of a tomogram-level `render_targets` at the same scale (atol 1e-6), including
  particles that cross the chunk border. A chunk with 50 particles stores < 4 kB;
  `"dense"` and `"points"` give the same batches without augmentation (atol 1e-6).
- **G2-C10** Two levels: a phantom with 80 Å and 400 Å species at 10 Å/px produces chunks at
  two `apix_work`; at each level only in-band classes are available.
- **G2-C11** `scales_per_level=3`: three directories per level, distinct `apix_work`, all
  assigned classes inside the band, same result for the same seed.
- **G2-C12** Border policy: a particle 3 voxels from the edge is absent with `skip`
  (matches v1) and partly drawn with `clip`.
- **G2-C13** Format guard errors in both directions.
- **G2-C14** Wedge preservation: on a wedged noise volume, `wedge_power_fraction` stays
  < 0.02 for 50 draws of the spatial part of the pipeline with `rotation_mode="flips"`,
  and also with `use_mw_aug=True` added.
- **G2-C15** The target is a `tio.LabelMap` in v1 and v2. With `label_interpolation="linear"`,
  a rotated target keeps its maximum within 1 % of the analytic peak after a 30° rotation,
  and the full pipeline with every intensity and Fourier augmentation at `p=1` leaves the
  target unchanged except for the spatial transforms.
- **G2-C16** Zoom: with a fixed `s`, `class_d_vox` = input × `s`, `apix_work` = input / `s`
  (rtol 1e-6), the target radius grows by `s` (± 1 voxel), and a class pushed outside the
  band becomes unavailable.
- **G2-C17** `read_apix`: header 0 → `ValueError` naming `apix_override`; override wins.

## Notes for the next gate
Document the exact batch keys, the meaning of `apix_work` and `class_d_vox` after zoom,
the class-table format in `chunk_layout.json`, and that `train.scale` and the class table
are saved in checkpoints (through `MainConfig`), because inference reads them from there.
