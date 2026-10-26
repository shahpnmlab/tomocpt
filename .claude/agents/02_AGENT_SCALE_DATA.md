# 02_AGENT_SCALE_DATA.md — Gate 2: data pipeline v2 (band scale policy)

## Mission
Let the data pipeline produce chunks where the target diameter varies inside a
band of voxel sizes, with the size recorded per chunk. Keep pipeline v1
byte-identical. No model or loss change in this gate.

## Why
In v1 every particle becomes about 20 voxels wide. A 70 Å particle at 10 Å/px is
upsampled ×2.86 (23× more voxels), and a 500 Å particle loses detail. Pipeline v2
resamples only when the particle is outside the band and tells the model the size.

## Allowed write paths
- `tomocpt/dataManager/**` (new: `scaling.py`, `metadata.py`)
- `tomocpt/labels/**`
- `tomocpt/defaultConfigs/scale_config.py`, `tomocpt/defaultConfigs/prepdata_config.py` (own fields)
- Path exception: `tomocpt/training/helpers.py`, function `_prepare_data_if_needed` only
  (pass the new arguments to `do_chunking`).
- `.claude/reports/gate_02_report.json`

## Tasks

### T2.1 Voxel size and scale rule (`dataManager/scaling.py`)
- `read_apix(mrc_path, override: Optional[float] = None) -> float`.
  Use `override` if given. Raise `ValueError` that names `apix_override` if the header
  value is `<= 0`. Log a warning if it is exactly 1.0 (often an unset header).
- `choose_working_apix(apix_native, lengths_ang, scale_cfg, desired_particle_px) -> float`
  - `fixed_particle` (v1, one length only): `d = L / apix_native`. If
    `abs(d - desired_particle_px) < 1`, return `apix_native`. Else return
    `apix_native * d / desired_particle_px`. (This matches
    `preprocessing.get_shape_for_resizing`.)
  - `band`: `apix_work = clip(apix_native, max(L)/band_max, min(L)/band_min)`.
    Raise `ValueError` if `max(L)/band_max > min(L)/band_min` (use `group_lengths`).
- `group_lengths(apix_native, lengths_ang, band_min, band_max) -> list[tuple[float, list[float]]]`.
  Sort lengths. Greedily grow a group while `max/min <= band_max/band_min`. Return
  `(apix_work, lengths)` per group with the band rule.
- `working_shape(native_shape_zyx, apix_native, apix_work) -> tuple[int, int, int]`:
  `int(s * apix_native / apix_work)` per axis (same rounding as v1).
- `resample_volume(vol: Tensor[D,H,W], target_shape, method: ResampleMethod) -> Tensor`.
  `trilinear` calls the existing `preprocessing.resize_volume` (unchanged).
  `fourier` crops or zero-pads the centred spectrum (`torch.fft.fftn` / `ifftn`),
  keeps the mean, and returns a real tensor. Handle odd and even sizes.

### T2.2 Coordinates and targets (`labels/**`)
- When `prepData.write_coordinates` is true, `prepare_data` also writes
  `<dataset_dir>/coords/<tomo>.csv` with columns `x,y,z,class_index,diameter_ang`
  (native voxels, the same values used to draw labels; `class_index = 0` and the
  dataset's diameter for single-class data; Gate 5 fills several classes) and adds
  columns `coords_path` and `apix_native` to the tracking CSV. Existing columns and
  label MRCs stay identical.
- `render_targets(points, out_shape_zyx, n_classes=1, border_policy="clip") -> np.ndarray[float32]`
  of shape `[n_classes, *out_shape_zyx]`. `points` is an `[N, 5]` array
  `z, y, x, class_index, d_vox` in the voxels of the output array (float positions allowed).
  Draws the Section 1.3 target with float radius `R = d_vox / 2`, grid `2*ceil(R)+1`,
  centred at the rounded position, max-combine per channel. `"clip"` draws the in-bounds
  part; `"skip"` drops particles whose box touches the array border (the v1 rule).
- `points_to_working(coords_csv_rows, apix_native, apix_work) -> np.ndarray[N, 5]`
  converts native `x,y,z` and `diameter_ang` to working `z,y,x,class_index,d_vox`.
- `prepData.label_border_policy` also applies to v1 label MRCs. Default `skip` keeps v1.

### T2.3 Chunking v2 (`dataManager/chunking.py`, `transforms.py`, `metadata.py`)
- New keyword arguments on `do_chunking`: `pipeline_version: int = 1`,
  `scale_cfg: Optional[ScaleConfig] = None`, `seed: int = 42`. Defaults run the v1 code path
  unchanged. Pass values explicitly; do not read `mainConfig` in new code.
- v2 per tomogram and per scale index `k` (`k = 0 … scales_per_tomogram-1`):
  - `k = 0` uses the band rule. `k > 0` draws `d_target` log-uniformly in the band
    (seeded by `seed` and the tomogram name) and sets `apix_work = L / d_target`.
  - Normalise as v1, resample with `scale_cfg.resample`, pad to ≥ 64 (as v1).
  - **Targets are stored as points, not as dense label chunks**
    (`scale.target_storage = "points"`, the v2 default). Convert the coordinates to
    working voxels (`points_to_working`), shift them by the padding, apply the
    tomogram-border policy (`prepData.label_border_policy`) once at this level, then for
    each chunk save `points/<chunk stem>.npy`: float32 `[N, 5]`, `z, y, x` relative to
    the chunk origin, `class_index`, `d_vox`, for every particle whose sphere overlaps the
    chunk (centre within `[-R, 64 + R)` on each axis). A chunk with 50 particles is
    about 1 kB, whatever the number of classes.
  - `scale.target_storage = "dense"` writes dense label chunks as in v1 (for debugging and
    for comparison tests).
  - Grid sample as v1. Drop empty chunks with the v1 probability, using "no point inside
    the chunk" as "empty".
  - Output directory `<split>/<dataset dir>__<tomo>__s{k}/` (unique per dataset,
    tomogram and scale; see `HANDOFF_chunk_collisions.md`) with `volumes/`, `points/`
    and `chunk_meta.json` (`ChunkMeta`).
  - Split by tomogram stem (all rows of one tomogram in one split).
- After all tomograms, write `chunks_dir/chunk_format.json`
  (`{"version": 2, "policy": ..., "band": [..], "resample": ..., "scales_per_tomogram": ...}`).
  In v2 this file is the "done" marker.
- Format guard: v2 training on a directory without a v2 format file raises
  `ValueError("chunk format ...")`. v1 training on a directory with a v2 format file
  raises the same kind of error.
- v2 split: if there are fewer than 2 tomograms, split by chunks and log a warning.
  v1 split is unchanged.

`metadata.py`:
```python
CHUNK_FORMAT_FILE = "chunk_format.json"
CHUNK_META_FILE = "chunk_meta.json"
@dataclass
class ChunkMeta:
    tomo_name: str; dataset_id: int; scale_index: int
    apix_native: float; apix_work: float; particle_length_ang: float; d_vox: float
    pipeline_version: int; resample: str
def write_chunk_meta(dir, meta) / read_chunk_meta(dir) -> ChunkMeta
def write_chunk_format(chunks_dir, info: dict) / read_chunk_format(chunks_dir) -> Optional[dict]
```

### T2.4 Dataset and augmentation v2 (`dataManager/dataloading.py`)
- Extend the existing `build_training_transforms(cfg: AugmentationConfig)`. Keep its
  order (spatial → Fourier → intensity) and its defaults (v1 pipeline unchanged).
  - `label_interpolation="linear"` passes `label_interpolation="linear"` to
    `RandomAffine` and `RandomElasticDeformation`. The target **stays a `tio.LabelMap`**
    (CLAUDE.md invariant 11): the Fourier and intensity transforms must keep skipping it.
  - `rotation_mode="flips"` replaces `RandomAffine` with
    `RandomFlip(axes=(0,1,2), flip_probability=flip_p)`. Axis flips keep the real missing
    wedge where it is, so the network never sees a wedge orientation that real data do
    not have. `RandomFourierDegradation` then adds its (oriented) wedge as before.
  - Recommended v2 settings (not defaults): `label_interpolation="linear"`,
    `rotation_mode="flips"`, `flip_p=0.5`.
- Zoom augmentation (v2, `zoom_aug_max > 1`): draw `s ~ U(1, zoom_aug_max)`, zoom both
  volume and target by `s` (trilinear), crop a random 64³, return `d_vox * s`. Zoom is a
  spatial step: apply it **before** `build_training_transforms` runs, so the Fourier
  degradation still comes after every spatial change.
- `VolumeDatsetIO` v2 reads the volume chunk and its points file, draws the target with
  `render_targets(points, (64, 64, 64), n_classes, "clip")` in the worker, and wraps it
  as a `tio.LabelMap` before the transforms (invariant 11). It returns the batch in
  CLAUDE.md Section 5.1. v1 output unchanged.
- With points, zoom augmentation scales the point positions and `d_vox` and draws the
  target after zooming, instead of interpolating a dense target. Rotations and elastic
  deformation still act on the drawn target (with `label_interpolation="linear"`).
- `size_balanced_sampling`: sample weights `∝ 1 / count(bin)` with 8 bins of
  `log(d_vox)` over the band. Lightning wraps the sampler for DDP.

## Public names
`tomocpt.dataManager.scaling.{read_apix, choose_working_apix, group_lengths, working_shape, resample_volume}`,
`tomocpt.dataManager.metadata.{ChunkMeta, write_chunk_meta, read_chunk_meta, write_chunk_format, read_chunk_format, CHUNK_FORMAT_FILE, CHUNK_META_FILE}`,
`tomocpt.labels.helpers.render_targets`,
`tomocpt.dataManager.dataloading.{build_training_transforms, VolumeDatsetIO}`,
`tomocpt.dataManager.chunking.do_chunking` (new kwargs).

## Pass criteria
- **G2-C1** v1 equivalence: `do_chunking` with defaults on `make_dataset(n_tomos=2)`
  gives the same file list and identical decoded arrays as the `gate-0-base` golden.
- **G2-C2** Band rule table: (apix 10, L 70) → 7.0; (10, 250) → 10.0; (5, 350) → 10.9375;
  (14, 140) → 14.0; group (10, [120, 300]) → one group at 10.0; group (10, [70, 500])
  → two groups.
- **G2-C3** `fixed_particle`: (10, 250, desired 20) → 12.5; (10, 205, 20) → 10.0.
- **G2-C4** Fourier resampling: a cosine above the new Nyquist keeps < 1 % of its power;
  the centroid of a smooth blob moves < 0.1 voxel; the mean is kept (rtol 1e-4).
  Round trip native → work → native moves a blob centroid < 0.25 native voxel.
- **G2-C5** v2 chunking writes `chunk_format.json` and one `chunk_meta.json` per
  `<tomo>__s{k}` dir; batches match Section 5.1 shapes and dtypes; `d_vox == L/apix_work`.
- **G2-C6** v2 targets: the maximum lies within 1 voxel of each particle centre;
  target > 0 for `r < R - 0.5` and == 0 for `r > R + 1`.
- **G2-C15** Points storage: for every chunk, `render_targets` from the stored points
  equals the crop of a tomogram-level `render_targets` at the same working scale
  (atol 1e-6), including particles that cross the chunk border.
- **G2-C16** A chunk with 50 particles stores a points file under 4 kB; `target_storage =
  "dense"` and `"points"` give the same training batches without augmentation (atol 1e-6).
- **G2-C17** Two datasets that share tomograms write separate chunk directories, and no
  tomogram stem is in both splits.
- **G2-C7** Border policy: a particle 3 voxels from the edge is absent with `skip`
  (matches v1) and partly drawn with `clip`.
- **G2-C8** Format guard errors in both directions.
- **G2-C9** Wedge preservation: on a wedged noise volume, `wedge_power_fraction`
  stays < 0.02 for 50 draws of the spatial part of the pipeline with
  `rotation_mode="flips"`, and also with `use_mw_aug=True` added (the oriented
  Fourier wedge must agree with the real one).
- **G2-C10** The target is a `tio.LabelMap` in v1 and v2. With
  `label_interpolation="linear"`, a rotated target has no nearest-neighbour plateaus
  (its maximum stays within 1 % of the analytic peak after a 30° rotation), and the
  full pipeline with every intensity and Fourier augmentation at `p=1` leaves the
  target unchanged except for the spatial transforms.
- **G2-C11** Zoom: with a fixed `s`, returned `d_vox` equals input × `s` (rtol 1e-6)
  and the target radius grows by `s` (± 1 voxel).
- **G2-C12** `scales_per_tomogram=3`: three directories, distinct `apix_work`,
  all `d_vox` inside the band, same result for the same seed.
- **G2-C13** v2 with one tomogram completes and logs a warning.
- **G2-C14** `read_apix`: header 0 → `ValueError` naming `apix_override`; override wins.

## Notes for the next gate
Document the exact batch keys, the meaning of `d_vox` after zoom, and where
`chunk_format.json` lives. State that `train.scale` must be saved in checkpoints
(it is, through `MainConfig`), because inference reads it from there.
