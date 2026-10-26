# 06_AGENT_FILAMENTS.md — Gate 6: microtubules (and other filaments) as picking classes

## Mission
Add filament classes to the Gate 3 model, so that one `predict` run gives particle
coordinates and microtubule centrelines, the latter as points along each filament with a
filament id (ready for subtomogram averaging). Train from the portal's microtubule
segmentation masks. Keep every earlier behaviour unchanged when no filament class is set.

## Data facts (checked against the CryoET Data Portal on 26 Oct 2026)
| Dataset | Content | Voxel size | Microtubule annotation |
|---|---|---|---|
| DS-10490 | CryoETSim, simulated crowded cells, 300 runs | 6.8 Å | `SegmentationMask`, ground truth; also actin filament masks and protein species as points |
| DS-10477 | C. elegans centrosomes, 12 runs | 13.48 Å | `SegmentationMask`, ground truth; also ribosome masks |
| DS-10513 | Mouse DRG axons, 20 runs | 20.58 Å | `SegmentationMask`, **not** ground truth (curator-recommended) |
All three give masks, not polylines, so centrelines are derived from the masks. At 6.8 Å/px
a 250 Å microtubule is 37 voxels, outside the default band, so Gate 2 resampling and the
Gate 3 voxel-size conditioning are exercised. Use DS-10513 as a held-out test set only and
report its numbers separately (its labels are less trustworthy).

## Design
- **Class table.** A filament class is a `ClassSpec(kind="filament")` with the tube
  diameter (microtubule 250 Å, actin 70 Å). Filament channels follow the particle channels.
- **Labels.** `prepData.filament_masks = "<class number>:<mask dir>,..."`: MRC masks matched
  to tomograms by name (as `labels.helpers.match_data_to_tomograms`). The class diameter
  comes from `class_diameters`. `mask_to_centerlines` turns each mask into polylines;
  `prepare_data` writes them to `<dataset_dir>/filaments/<tomo>.csv`
  (`filament_id, x, y, z, class_number, diameter_ang`, native voxels, ordered along each
  filament).
- **Target.** The same radial profile as CLAUDE.md Section 1.3, with `r` the distance to the
  nearest centreline segment and `R = d_vox / 2`, max-combined per channel. Chunks store
  segments (`filaments/<chunk>.npy`, CLAUDE.md Section 5.2); `render_targets` draws them
  together with points.
- **Loss.** The Gate 3 per-class loss, unchanged (the Hessian is scaled by the tube's
  `class_d_vox`).
- **Inference.** Per filament channel: threshold at `confidence_threshold`, skeletonise
  (`skimage.morphology.skeletonize`), trace the skeleton graph into polylines (split at
  branch points, join gaps shorter than `0.5 × diameter` with consistent direction), drop
  polylines shorter than `filament_min_length_ang`, smooth, and resample every
  `filament_spacing_ang`. Peaks of particle classes are not suppressed by filaments.

## Allowed write paths
- `tomocpt/labels/**` (new: `filaments.py`)
- `tomocpt/dataManager/**` (segments in chunks and batches)
- `tomocpt/predict/**` (tracing and filament outputs)
- `tomocpt/defaultConfigs/{prepdata_config,infer_config}.py`
- `scripts/download_portal_data.py` (segmentation masks)
- `benchmarks/filaments.py` (new), `README.md`
- `.claude/reports/gate_06_report.json`

## Config (this gate adds)
`prepData.filament_masks: Optional[str] = None`; `infer.filament_spacing_ang: float = 40.0`;
`infer.filament_min_length_ang: float = 200.0`.

## Tasks
- **T6.1** `labels/filaments.py`:
  - `mask_to_centerlines(mask, apix, diameter_ang, min_length_ang) -> list[np.ndarray]`
    (polylines `[K, 3]` as `x, y, z` native voxels): close small holes, skeletonise, trace,
    prune spurs shorter than the radius, smooth.
  - `render_tubes(segments, out_shape_zyx, n_classes, border_policy="clip") -> np.ndarray`
    (used by `render_targets`).
  - `trace_filaments(prob_map, threshold, apix, diameter_ang, min_length_ang, spacing_ang)
    -> list[np.ndarray]` (inference).
  - `resample_polyline(polyline, spacing_vox) -> np.ndarray`.
- **T6.2** `class_table_from_config` includes filament classes from `filament_masks`;
  `prepare_data` writes the filament CSVs; chunking writes segments for every segment that
  passes within `R` of the chunk.
- **T6.3** Outputs. RELION 3.1 / 5.0: filament points go into the same STAR file as the
  particles, with `rlnClassNumber`, `rlnHelicalTubeID` (1-based per tomogram and class),
  `rlnHelicalTrackLengthAngst`, `rlnAutopickFigureOfMerit` (mean probability along the
  filament); particle rows have no helical columns' values (`-1`/`0`). Warp: one file per
  class. IMOD: one open contour per filament.
- **T6.4** `scripts/download_portal_data.py --masks microtubule,...` downloads
  `SegmentationMask` annotations (MRC) for those object names, checks shape and voxel size
  against the tomogram, and records them (with ground-truth status) in the manifest and the
  `prepData` snippet.
- **T6.5** `benchmarks/filaments.py`: train on DS-10490 and DS-10477 (split by run), test on
  held-out runs and on DS-10513; arms: filaments only, filaments + particles in one model.
  Report centreline precision/recall/F1 (length-weighted, within `0.5 × diameter`), point
  F1 at `filament_spacing_ang`, particle F-beta, time and memory.

## Public names
`tomocpt.labels.filaments.{mask_to_centerlines, render_tubes, trace_filaments, resample_polyline}`,
`benchmarks.filaments.centerline_scores`.

## Pass criteria
- **G6-C1** With no filament class, chunks, batches, model outputs and predictions equal
  Gate 4 (atol 1e-6).
- **G6-C2** `mask_to_centerlines` on a phantom tube mask (curved, 250 Å diameter) recovers the
  true centreline within 1 voxel (mean) and its length within 5 %; two crossing tubes give
  two polylines; a tube broken by a gap shorter than half its diameter gives one polyline.
- **G6-C3** `render_tubes`: profile maximum on the centreline; target > 0 for
  `r < R - 0.5`, == 0 for `r > R + 1`; segments stored per chunk render the same as a crop of
  the tomogram-level render (atol 1e-6).
- **G6-C4** `trace_filaments` on a rendered target recovers the centreline within 0.5 voxel
  and drops polylines shorter than `filament_min_length_ang`; `resample_polyline` spacing
  error < 1 %.
- **G6-C5** STAR round trip: particle and filament rows in one file with the original class
  numbers and `rlnHelicalTubeID`; reads back with `starfile`.
- **G6-C6** Portal masks: on a fake portal client (as in `tests/test_portal_download.py`),
  `--masks` writes masks with the tomogram's shape and voxel size and records ground-truth
  status in the manifest.
- **G6-C7** DDP rule: one step with a batch that has filament channels unavailable gives
  every parameter a gradient (not `None`).
- **G6-C8** (slow) Synthetic: tubes (250 Å and 70 Å) and spheres of two sizes in one model;
  held-out centreline F1 ≥ 0.8 for both tube classes and particle F1 ≥ 0.8.
- **G6-C9** (slow) Interference: particle F1 of the joint model is at most 0.03 below the
  Gate 3 particle-only model on the same data.

## Acceptance for humans (not a CI gate)
Recommend joint filament + particle models only if, on DS-10490 / DS-10477 held-out runs,
the joint arm is within 0.03 of the filaments-only arm on centreline F1 and of the
particle-only arm on particle F-beta.
