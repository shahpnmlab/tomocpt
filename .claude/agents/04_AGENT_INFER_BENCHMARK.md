# 04_AGENT_INFER_BENCHMARK.md — Gate 4: v2 inference, multi-length picking, benchmark harness

## Mission
Make `tomocpt predict` run pipeline-v2 checkpoints, pick several particle sizes from
one backbone pass, and keep v1 output identical. Build the benchmark harness that
Gates 5, 6 and 7 (and a human) use to decide what ships.

## Allowed write paths
- `tomocpt/predict/**`
- `tomocpt/defaultConfigs/infer_config.py`
- `benchmarks/**` (new)
- `README.md` (usage and changelog)
- `.claude/reports/gate_04_report.json`

## Tasks

### T4.1 Read the training settings from the checkpoint
Use `safe_getattr` on `model.hparams.config`: `train.data_pipeline_version`,
`train.scale`, `train.network.size_conditioning`, `train.network.moe.enabled`,
`prepData.desired_particle_pixel_size`. Never read these from `mainConfig.train`.

### T4.2 Plan the run
`plan_inference(apix_native, lengths_ang, scale_cfg, desired_px, pipeline_version)
-> list[InferenceGroup]`, where
`InferenceGroup = NamedTuple(apix_work: float, lengths: list[float], d_vox: list[float], overlap: int)`.
- Lengths come from `infer.lengths` (comma-separated) or `[infer.length]`.
- v1: one group per length, `apix_work` from the `fixed_particle` rule, overlap 16
  (the legacy `64 // PATCH_OVERLAP_FACTOR`). Several lengths with a v1 checkpoint run
  the legacy path once per length. This is the multi-size baseline.
- v2: `dataManager.scaling.group_lengths` with the band from the checkpoint.
  `overlap = min(32, max(64 // PATCH_OVERLAP_FACTOR, ceil(max(d_vox))))`.

### T4.3 Predict a group
- Preprocess as in training (normalise at native size, then resample with the
  checkpoint's `scale.resample`), pad, grid-sample 64³ with the group overlap.
- Conditioned model: per patch batch call `encode` once, then `decode` once per
  length, into one `GridAggregator(overlap_mode="hann")` per length. Unconditioned
  v2 model: one map per group, shared by its lengths.
- TTA as in v1, per length.
- Resize each map back to the native shape, apply `masks_dir`, and save:
  single length → `<stem>.mrc` (v1 name); several lengths → `<stem>_L{int(L)}.mrc`.
- `skip_inference` finds `_L{int(L)}` maps when `infer.lengths` is set.

### T4.4 Peaks and output
- Change `_extract_peaks_from_mask` to take `length` and `distance_threshold`
  arguments; defaults come from `infer_config` so v1 calls are unchanged.
- One length: v1 file names. Several lengths: one coordinate file per length,
  `<filename stem>_L{int(L)}<suffix>`; IMOD `<tomo>_L{int(L)}.mod`.
  Do not invent RELION labels.
- `infer.apix_override` passes to `read_apix`.

### T4.5 Benchmark harness (`benchmarks/`, not part of the fast tests)
- `benchmarks/metrics.py`: greedy one-to-one matching inside
  `match_radius_frac × radius` (default 0.5); precision, recall, F1, F-beta (β = 4,
  the CZII challenge metric). Per species and macro mean.
- `benchmarks/synthetic_multisize.py`: uses `tests/utils/phantoms.py`; at least
  4 tomograms; species diameters across the band (for example 100, 160, 240, 320 Å
  at 10 Å/px), wedge ±60°, two SNR levels. Arms:
  (a) legacy v1, one model per species; (b) v2 conditioned, one model for all;
  (c) v2 conditioned, leave-one-species-out (the held-out species is picked by its
  length only). Same epochs and seeds per arm; 3 seeds.
- `benchmarks/czii_portal.py`: protocol and loader for the CZII
  "CryoET Object Identification" data from the CryoET Data Portal (six labelled
  species). The user downloads the data; the script takes paths. Same arms as above,
  plus the public pretrained TomoCPT model as a reference.
- `benchmarks/report.py`: writes `benchmarks/out/<run>/report.json` and `report.md`
  with per-species metrics (mean ± sd over seeds), wall time per tomogram, peak GPU
  memory, and voxels processed per tomogram.

## Public names
`tomocpt.predict.helpers.{plan_inference, InferenceGroup, _extract_peaks_from_mask}`,
`benchmarks.metrics.{match_points, prf_beta}`.

## Pass criteria
- **G4-C1** v1 lock: legacy tiny checkpoint on a synthetic tomogram through
  `infer_tomos` (no Dask): confidence map equals the `gate-0-base` golden
  (atol 1e-5); coordinates equal (same count, atol 1e-3 voxel).
- **G4-C2** `plan_inference` table: v1 and v2 cases from Gate 2 G2-C2 and the overlap
  rule (`d_vox` 10 → 16; 28 → 28; 40 → 32).
- **G4-C3** Multi-length sharing: with a conditioned tiny model, the map for length L
  from a multi-length run equals a single-length run for L (atol 1e-5).
- **G4-C4** `encode` runs once per patch batch for any number of lengths (call count).
- **G4-C5** Coordinate round trip with an oracle model (noise-free phantom of
  target-shaped dark blobs; the model returns `logit(clip(-x_norm, ε, 1-ε))`):
  v1 and v2 (apix_native 5 Å, L 350 Å, so resampling runs) recover every centre
  within 0.5 native voxel.
- **G4-C6** File names for one and several lengths; `skip_inference` re-reads `_L` maps.
- **G4-C7** v1 checkpoint with `infer.lengths="150,300"` equals two separate v1 runs.
- **G4-C8** Header voxel size 0 → `ValueError` naming `apix_override`.
- **G4-C9** `benchmarks.metrics`: hand-made cases (exact match, one miss, one false
  positive, a point just outside the radius).
- **G4-C10** (slow) `synthetic_multisize.py --tiny` runs all arms for 1 epoch and
  writes `report.json` with the required keys.

## Acceptance for humans (not a CI gate)
Recommend making v2 the default only if, on CZII and on synthetic data,
arm (b) is within 0.02 F-beta of arm (a) on every species, arm (b) needs less total
wall time than arm (a) summed over species, and arm (c) reaches at least 0.8 × the
F-beta of arm (b) for the held-out species.
