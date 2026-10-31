# Gate 6 tests (filaments)

The Test Author wrote these tests from `.claude/agents/06_AGENT_FILAMENTS.md` (amended 31 Oct 2026)
before the implementer started. Questions where the spec is silent:
`.claude/reports/gate_06_ta_questions.md`.

- Fast tier: `pytest tests/gate6 -m "not slow"` (CPU, fp32, seeded). On the development machine:
  about 40 s on `v2-dev` HEAD (most tests fail fast there); about 25 s for the geometry tests with a
  scratch reference implementation. Expected well under 2 min with an implementation.
- Slow tier: `pytest tests/gate6 -m slow` (G6-C8, G6-C9; CUDA; two tiny-model trainings).

New public names (`tomocpt.labels.filaments.*`) are imported inside the test functions, so
collection works before the implementation exists and the G6-C1 lock runs on any code.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse), `slow` marker |
| `g6_helpers.py` | Helpers (no tests): analytic curves, exact point-to-polyline distances, the CLAUDE.md 1.3 profile, segment rows, filament STAR / IMOD writers, RELION `Euler_angles2matrix`, v2 configs and tiny checkpoints with filament classes, the sign oracle and its tomogram |
| `test_g6_c1_lock.py` | C1 |
| `test_g6_geometry.py` | C2, C3 (function), C4, C10, C11, C13 (function) |
| `test_g6_labels.py` | C3 (per-chunk segments), C12 (labels from coordinates, class table), C13 (`write_filament_masks`) |
| `test_g6_outputs.py` | C5, C12 (`predict` -> `read_filament_coords` round trip) |
| `test_g6_portal.py` | C6 |
| `test_g6_ddp.py` | C7 |
| `test_g6_bench.py` | `benchmarks.filaments.centerline_scores` (orchestrator ruling) |
| `test_g6_learning.py` | C8, C9 (slow) |
| `../golden/gate6_common.py` | Shared by the C1 golden generator and the C1 test |
| `../golden/make_goldens_gate6.py` | C1 golden generator |
| `../golden/gate6_c1.npz`, `gate6_c1.json` | C1 golden (300 kB) and provenance |

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| G6-C1 No filament class = Gate 4 | `test_G6_C1_chunks_equal_gate4`, `test_G6_C1_batches_equal_gate4`, `test_G6_C1_model_outputs_equal_gate4`, `test_G6_C1_predictions_equal_gate4` |
| G6-C2 `mask_to_centerlines` | `test_G6_C2_curved_tube_centreline_and_length`, `test_G6_C2_uint8_mask_is_accepted`, `test_G6_C2_two_crossing_tubes_give_two_polylines`, `test_G6_C2_gap_shorter_than_half_the_diameter_gives_one_polyline`, `test_G6_C2_hollow_curved_tube_centreline_and_length`, `test_G6_C2_hollow_crossing_tubes_give_two_polylines` |
| G6-C3 `render_tubes` | `test_G6_C3_reference_profile_is_generate_gaussian_sphere`, `test_G6_C3_straight_tubes_profile_centreline_and_no_ripple`, `test_G6_C3_helical_tube_profile`, `test_G6_C3_crossing_tubes_max_combine_and_round_caps`, `test_G6_C3_empty_segments`, `test_G6_C3_segments_per_chunk_render_the_tomogram_crop` |
| G6-C4 `trace_filaments`, `resample_polyline` | `test_G6_C4_trace_recovers_centreline_and_drops_short`, `test_G6_C4_min_length_is_applied`, `test_G6_C4_resample_polyline_spacing[3.0, 4.7]`, `test_G6_C4_resample_polyline_irregular_straight_line` |
| G6-C5 Outputs | `test_G6_C5_one_file_per_class`, `test_G6_C5_relion_filament_star`, `test_G6_C5_particle_file_unchanged`, `test_G6_C5_warp_filament_star_has_angles`, `test_G6_C5_imod_one_open_contour_per_filament` |
| G6-C6 Portal masks | `test_G6_C6_masks_written_with_tomogram_shape_and_voxel_size`, `test_G6_C6_mask_with_a_different_shape_is_refused` |
| G6-C7 DDP rule | `test_G6_C7_filament_channel_unavailable_every_parameter_gets_a_gradient[sigmoid, softmax]` |
| G6-C8 (slow) | `test_G6_C8_tubes_and_spheres_in_one_model` |
| G6-C9 (slow) | `test_G6_C9_particle_interference` |
| G6-C10 `fit_filament_spline` | `test_G6_C10_noisy_helix[0, 1, 2]`, `test_G6_C10_same_curve_at_two_voxel_sizes`, `test_G6_C10_few_points[2, 3]` |
| G6-C11 Angle convention | `test_G6_C11_reference_matrix_is_self_consistent`, `test_G6_C11_random_directions_map_to_the_filament_axis`, `test_G6_C11_landmarks` (C5 also checks the STAR angles against the coordinates) |
| G6-C12 Labels from coordinates | `test_G6_C12_class_table_from_config_has_filament_rows[filament_masks, filament_coords]`, `test_G6_C12_read_filament_coords[star_tl, star_nn, imod]`, `test_G6_C12_prepare_data_from_coordinates[star_tl, star_nn, imod]`, `test_G6_C12_filament_label_spacing_ang`, `test_G6_C12_masks_and_coordinates_for_one_class_raise`, `test_G6_C12_predict_output_round_trip[relion, imod]` |
| Benchmark scorer | `test_G6_bench_centerline_scores` |
| G6-C13 Masks from splines | `test_G6_C13_centerlines_to_mask_radius`, `test_G6_C13_write_filament_masks`, `test_G6_C13_default_writes_no_masks` |

55 fast test cases, 2 slow ones.

## G6-C1 golden

The reference is "Gate 4" behaviour, i.e. the `v2-dev` code before Gate 6 (commit 6830bc1; Gates
2-5, no filament code), **not** `v2-gate-0-base` (which has no pipeline v2). Generated on that
commit, CPU only:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python tests/golden/make_goldens_gate6.py --expect-root $PWD

The script refuses to run when `tomocpt.labels.filaments` exists. Two runs gave bit-identical
output. Content (`gate6_common.py`): multi-class `prepare_data` + pipeline-v2 chunking of two
64x96x96 phantoms at 10 A/px with classes 1 "small" (120 A) and 2 "big" (200 A) (8 chunks: every
`points/*.npy` in full, volumes as digests); the first validation and the first training batch of
`Data` (default `AugmentationConfig`, seeds 0; every key); a tiny conditioned v2 model with
`gate3_common.reinit_` weights (logits at 10 and 15 A/px); `predict()` with a tiny v2 checkpoint
(Gate 3 helper, then `reinit_` weights) on a blob tomogram: real network (per-class maps and STAR
coordinates, cross-class NMS on) and the Gate 4 `NegInputOracle` (coordinates, NMS off). Dense
arrays are compared through a `[::4, ::4, ::4]` subsample (atol 1e-6) plus float64 sum, sum of
squares and a seeded random projection (rtol 1e-6); points and coordinates in full (atol 1e-6).
Manifest keys of the golden must keep their values (new keys allowed); new batch keys are not
allowed.

## Results on the current code (`v2-dev` HEAD 6830bc1, no Gate 6 code)

`pytest tests/gate6 -m "not slow"`: **7 passed, 48 failed** (37 s).

- Pass: the four G6-C1 tests (the lock), `test_G6_C3_reference_profile_is_generate_gaussian_sphere`
  and `test_G6_C11_reference_matrix_is_self_consistent` (checks of the test's own references), and
  `test_G6_C5_particle_file_unchanged` (a lock: on HEAD the filament class is an ordinary particle
  class, the particle channel is still identical).
- Fail, as intended: every other test. Reasons: `tomocpt.labels.filaments` does not exist
  (C2, C3, C4, C7, C10, C11, C12, C13), `benchmarks.filaments` does not exist (scorer test); `class_table_from_config` has no filament rows (C12 class
  table, C3 chunks, C7); `PrepdataConfig` has no `write_filament_masks` (C13 default); `predict`
  writes no filament output (C5, C12 round trip); the portal script has no `--masks` (C6).

`v2-gate-0-base` was not run separately: it lacks pipeline v2 altogether, so every new-behaviour
test fails there for the same reasons (missing module / fields), and the C1 golden is defined on
`v2-dev`, not on base.

Self-check with a scratch reference implementation (not committed; closing at the tube radius
to fill the lumen, skeletonize, junction pairing, end extension along the tangent to the mask
boundary, `splprep`): all 26 tests of `test_g6_geometry.py` pass. C2 margins: solid curved tube
mean 0.18 voxel, length -0.7 %; hollow curved tube mean 0.13 voxel, length -1.4 %, coverage 1.0
(thresholds 1 voxel, 5 %, 0.95); hollow crossing tubes: two polylines, mean 0.0 voxel, coverage
1.0. Without the closing at the tube radius and the end extension the reference failed the
hollow case (length -8.6 %); C12 spline numbers of the labels inputs:
max deviation 0.013 voxel, tube-target difference 0.005 (thresholds 0.5 and 0.05).

## Slow tier (G6-C8, G6-C9)

Thresholds are the spec's (centreline F1 >= 0.8 for both tube classes, particle F1 >= 0.8;
joint particle F1 >= particle-only - 0.03). They are **not calibrated against an implementation**
(none exists). One run with the scratch reference implementation is recorded below.
`TOMOCPT_G6_SEED` selects the seed; results are printed as `G6_RESULT C8_C9 ...` (`-s`).

Reference run (seed 0, scratch reference implementation of `labels.filaments` on HEAD, RTX A6000,
10 min 55 s for both trainings):

| Model | particle F1 (tp, fp, fn) | mt F1 (P / R) | actin F1 (P / R) |
|---|---|---|---|
| joint | 0.768 (48, 29, 0) | 1.00 (1.00 / 1.00) | 0.94 (1.00 / 0.89) |
| particles only | 0.691 (48, 43, 0) | - | - |

Run 1 used pick threshold 0.3 without the cross-class rule: C9 passed, C8 failed on particle F1.

Run 2 (current setup: pick threshold 0.5 plus the Gate 4 cross-class rule; seed 0; 11 min 52 s):

| Model | particle F1 (tp, fp, fn) | at 0.3, no rule | mt F1 | actin F1 (P / R) |
|---|---|---|---|---|
| joint | 0.667 (32, 16, 16) | 0.658 (48, 50, 0) | 1.00 | 0.94 (0.97 / 0.90) |
| particles only | 0.958 (46, 2, 2) | 0.686 (48, 44, 0) | - | - |

**C8 and C9 both fail against the reference in run 2.** The particle-only model reaches the 0.8
threshold, so the evaluation can be passed. The joint reference model scores particles lower:
half its particles fall below 0.5. That is real interference in this tiny setup (2000 steps,
4 classes), not an evaluation artefact. The threshold stays at 0.8 (human decision). The setup is
not yet proven reachable for a joint model; see the questions file.

## Contracts assumed (the spec gives no exact form)

1. **Coordinates.** Polylines in and out of every `labels.filaments` function are `[K, 3]`
   `x, y, z` in native voxels (spec T6.1). `trace_filaments` returns them too. Volumes and masks
   are `(Z, Y, X)`. `mask_to_centerlines` accepts bool or uint8 masks.
2. **`render_tubes(segments, out_shape_zyx, n_classes, border_policy="clip")`**: `segments` is the
   CLAUDE.md 5.2 array `[M, 8]` = `z0, y0, x0, z1, y1, x1, class_index, d_vox` (chunk/array voxels,
   `R = d_vox / 2`); returns float32 `[n_classes, Z, Y, X]`; the target is the profile of the
   distance to the union of the segments (round caps, max-combined). `render_targets` with
   segments is not called directly (its segment argument is not specified).
3. **Chunks.** `filaments/<name>.npy` next to `points/<name>.npy` (same stem), float32 `[M, 8]`;
   absent = no segments. Native voxel size, no padding in the test, so chunk offsets are TorchIO
   grid starts (stride 32).
4. **`fit_filament_spline`** returns `(points [K,3], tangents [K,3], arc_length_ang [K])`, arc
   length 0 at the first point and spaced `spacing_ang`; the last step may be shorter (the end
   point may be appended). Unit tangents, no sign flips.
5. **`tangents_to_relion_angles(tangents)`** returns `(tilt_deg [K], psi_deg [K])`; the test
   accepts `A(0, tilt, psi) @ Z = +d` or `-d` (polarity unknown, spec).
6. **`read_filament_coords(path, apix)`**: STAR files as `predict` writes RELION 3.1
   (`rlnCoordinateX/Y/Z` in voxels, optics `rlnImagePixelSize`), grouped by `rlnHelicalTubeID`,
   ordered by `rlnHelicalTrackLengthAngst` or by nearest-neighbour chaining (either direction);
   IMOD models with one open contour per filament, points in voxels. One polyline per filament,
   in order along it.
7. **`prepare_data`** (multi-class STAR particles plus `filament_masks` / `filament_coords`):
   per-class source directory with files matched by tomogram name (`<dir>/<tomo>.mrc|.star|.mod`);
   CSV `<dataset_dir>/filaments/<tomo>.csv` with rows of each `filament_id` in order along it;
   default label spacing 0.5 x voxel; masks at `<dataset_dir>/filament_masks/<class name>/<tomo>.mrc`
   (integer values 0/1; the MRC mode is not checked). A class in both fields raises `ValueError`
   from `prepare_data`.
8. **`class_table_from_config`**: a class number listed in `filament_masks` or `filament_coords` is
   `kind="filament"`, after all particle rows (sorted by number within each kind).
9. **`centerlines_to_mask(polylines, shape_zyx, apix, diameter_ang)`** returns uint8; tested with
   smooth dense polylines, so fitting a spline inside or not gives the same mask within ±1 voxel.
10. **Outputs** (Gate 4 rule): `<name>_<class name>.star` for RELION 3.1 and Warp,
    `<tomo stem>_<class name>.mod` for IMOD. The filament STAR has `rlnMicrographName` (3.1).
11. **Portal (`--masks`)**: the fake client mimics the object model the script already uses
    (`Annotation.annotation_shapes[].annotation_files[]`, `shape_type="SegmentationMask"`,
    `format="mrc"`) and serves files through `download_file`, `requests.get` and `download()`
    methods. The mask location is not specified: the test looks for an MRC outside `tomograms/`
    whose relative path contains the run name and the object name. The manifest must hold, inside
    each run record, a dict with a `ground_truth` key that mentions the microtubule (as key path or
    value), and the `prepdata_snippet.yaml` must mention `filament_masks`. A mask whose shape
    differs from the tomogram's must stop the script (non-zero exit or exception) or not be written
    with the wrong shape.
12. **Predict oracle** (`g6_helpers.sign_oracle_factory`): particles dark, tubes bright; slabs of
    +1 / -1 at the X ends fix the robust normalisation at median 0, `p95 - p5 = 2`, and map to 0 in
    both channel types. Channel types come from `class_table_from_config` of the checkpoint config.
13. **C2 masks** (human decision 31 Oct 2026: DS-10490 masks are hollow walls): tube masks have
    **flat open ends** (no caps, as a tube cut by the border or ending in a segmentation); the
    true centreline runs from end face to end face. Solid case: all voxels within R = 6.25 voxels
    (250 A at 20 A/px); hollow case: the wall between r = 3.75 (150 A lumen) and 6.25, the lumen
    open at both ends. An implementation must fill the lumen and reach the end faces (length
    within 5 %).
14. **`benchmarks.filaments.centerline_scores(pred_polylines, true_polylines, tolerance_vox) -> dict`**
    with `"precision"`, `"recall"`, `"f1"`, length-weighted (orchestrator ruling). Polylines
    `x, y, z` voxels.
