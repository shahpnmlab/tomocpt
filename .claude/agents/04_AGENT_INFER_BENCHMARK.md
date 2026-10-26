# 04_AGENT_INFER_BENCHMARK.md — Gate 4: v2 inference and benchmarks

## Mission
Make `tomocpt predict` run pipeline-v2 checkpoints: every class of the model is picked in
one run, at the working voxel size that fits it, with per-class outputs. Keep v1 output
identical. Build the benchmarks that decide whether v2 becomes the recommended pipeline.

## Allowed write paths
- `tomocpt/predict/**`
- `tomocpt/defaultConfigs/infer_config.py`
- `benchmarks/**` (except `benchmarks/metrics.py`, frozen API from Gate 1; you may add files)
- `README.md` (usage and changelog)
- `.claude/reports/gate_04_report.json`

## Config (this gate adds)
`InferConfig`: `apix_override: Optional[float] = None`, `classes: Optional[str] = None`
(comma-separated class names; default all), `cross_class_nms: bool = True`,
`accumulate_dtype: str = "float32"`, `accumulate_device: str = "auto"`.
For v2 checkpoints `infer.length` is not used (the class table gives the diameters); make
it optional without changing its v1 meaning.

## Tasks

### T4.1 Read the training settings from the checkpoint
With `safe_getattr` on `model.hparams.config`: `train.data_pipeline_version`, `train.scale`,
`train.network.scale_conditioning`, `prepData` (for `class_table_from_config`),
`prepData.desired_particle_pixel_size`. Never read these from `mainConfig.train`.

### T4.2 Plan the run
`plan_inference(apix_native, class_table, scale_cfg, desired_px, pipeline_version, length=None)
-> list[InferenceLevel]`, where
`InferenceLevel = NamedTuple(apix_work: float, classes: list[int], overlap: int)`.
- v1: one level from the fixed-particle rule with `infer.length`, overlap 16
  (`64 // PATCH_OVERLAP_FACTOR`), class `[0]`.
- v2: `dataManager.scaling.plan_levels` with the band from the checkpoint; each level reads
  only its `assigned` classes. `overlap = min(32, max(16, ceil(max d_vox of its classes)))`.
- `infer.classes` removes classes (and levels left empty).

### T4.3 Predict a level
- Preprocess as in training: normalise at native size, resample with the checkpoint's
  `scale.resample`, pad, grid-sample 64³ with the level overlap.
- Conditioned model: `forward(x, apix=level.apix_work)`. One `GridAggregator(overlap_mode="hann")`
  per class of the level. Accumulators use `accumulate_dtype` on `accumulate_device`
  (`auto` = the GPU if the maps fit in free memory, else CPU).
- TTA as in v1.
- Resize each class map back to the native shape, apply `masks_dir`.

### T4.4 Peaks and outputs
- `_extract_peaks_from_mask` takes `length` and `distance_threshold` arguments; defaults come
  from `infer_config`, so v1 calls are unchanged. v2 calls it per class with the class
  diameter.
- `cross_class_suppression(peaks_by_class, radii, factor=0.5) -> (peaks_by_class, removed_counts)`:
  when peaks of different classes are closer than `factor × min(r_a, r_b)`, keep the higher
  score. Log removed counts per class pair (the confusion signal).
- Outputs, v1: unchanged names and formats.
- Outputs, v2:
  - RELION 3.1 / 5.0: one STAR file with `rlnClassNumber` (original class numbers) and
    `rlnAutopickFigureOfMerit` (peak score);
  - Warp and IMOD: one file per class, `<name>_<class name>.<ext>`;
  - `save_prediction_confidence_map`: one map per class, `<stem>_<class name>.mrc`;
    `skip_inference` re-reads them.
- `infer.apix_override` passes to `read_apix`.

### T4.5 Benchmarks (`benchmarks/`, not part of the fast tests)
- `benchmarks/synthetic_multisize.py`: phantoms (`tests/utils/phantoms.py`), at least 6
  tomograms, species from 80 to 500 Å (so two levels are needed at 10 Å/px), including two
  of the same size and different shape; wedge ±60°; two SNR levels; 3 seeds.
- `benchmarks/czii_multiclass.py`: CZII 10440 (6 species), split by tomogram; paths given by
  the user (`scripts/download_portal_data.py`).
- Arms, same epochs and seeds: (a) one v1 model per species (today's practice);
  (b) one v2 multi-class model, conditioned; (c) (b) without conditioning;
  (d) (b) warm-started from the public TomoCPT model.
- `benchmarks/report.py`: `benchmarks/out/<run>/report.json` and `report.md` with per-class
  F-beta (β = 4), precision, recall (mean ± sd over seeds), cross-class suppression counts,
  wall time per tomogram, peak GPU memory.

## Public names
`tomocpt.predict.helpers.{plan_inference, InferenceLevel, cross_class_suppression, _extract_peaks_from_mask}`.

## Pass criteria
- **G4-C1** v1 lock: legacy tiny checkpoint on a synthetic tomogram through `infer_tomos`
  (no Dask): confidence map equals the `gate-0-base` golden (atol 1e-5); coordinates equal
  (same count, atol 1e-3 voxel).
- **G4-C2** `plan_inference` table: v1 case; v2 cases from G2-C2; the overlap rule
  (`d_vox` 10 → 16; 28 → 28; 40 → 32); `infer.classes` drops a level that becomes empty.
- **G4-C3** Every class is read at exactly one level; a two-level phantom gives maps for
  all classes in one `predict` call.
- **G4-C4** Coordinate round trip with an oracle model (noise-free phantom of
  target-shaped dark blobs; the model returns `logit(clip(-x_norm, ε, 1-ε))` per channel):
  v1, and v2 with resampling (apix_native 5 Å, diameters 120 and 350 Å), recover every
  centre within 0.5 native voxel.
- **G4-C5** `cross_class_suppression`: peaks of different classes 1 voxel apart → the higher
  stays; 3 radii apart → both stay; same-class peaks are untouched; counts are correct.
- **G4-C6** STAR round trip: the RELION output has `rlnClassNumber` with the original
  numbers and reads back with `tomocpt.labels.star.Star`; Warp/IMOD per-class names;
  `skip_inference` re-reads per-class maps.
- **G4-C7** `float16` CPU accumulation gives maps within 1e-3 of `float32` and the same peaks.
- **G4-C8** Header voxel size 0 → `ValueError` naming `apix_override`.
- **G4-C9** (slow) `synthetic_multisize.py --tiny` runs all arms for 1 epoch and writes
  `report.json` with the required keys.

## Acceptance for humans (not a CI gate)
Recommend v2 as the default pipeline only if, on CZII and on synthetic data, arm (b) is
within 0.02 F-beta of arm (a) on every species and better on the same-size pairs, and
arm (b) needs less total wall time than arm (a) summed over species. Keep
`scale_conditioning` recommended only if (b) beats (c) on the two-level synthetic set.
