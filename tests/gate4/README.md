# Gate 4 tests (v2 inference, calibration, benchmarks)

The Test Author wrote these tests from `.claude/agents/04_AGENT_INFER_BENCHMARK.md` (including the
human change of 27 Oct 2026: one RELION file per class, no merged STAR) before the implementer started.
Fast tier: `pytest tests/gate4 -m "not slow"` (about 35 s on the development machine, CPU only).
Slow tier: `pytest tests/gate4 -m slow` (G4-C9, runs the tiny benchmark).

New public names are imported inside the test functions, so the v1 locks run on code without them.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse), `legacy_ckpt`, `v2_ckpts` (tiny v2 checkpoints from the Gate 3 helper `build_v2_checkpoint`, one-batch CPU fits, built once per session) |
| `g4_helpers.py` | Helpers (no tests): `run_predict` (calls `tomocpt.predict.run.predict()`), in-process Dask, oracle networks, blob phantoms, output readers |
| `test_g4_v1_lock.py` | C1 |
| `test_g4_plan.py` | C2, C3 (planning property; level-tagging oracle through `predict`), `infer.classes` |
| `test_g4_roundtrip.py` | C3 (two-level phantom), C4, C6, C7, C8 |
| `test_g4_nms_merge.py` | C5, C12 |
| `test_g4_calibration.py` | C10, C11 |
| `test_g4_benchmark.py` | C9 (slow) |
| `../golden/gate4_common.py` | Shared by the golden generator and the C1 test (tomogram, cases, re-initialised checkpoint, direct `infer_tomos` call) |
| `../golden/make_goldens_gate4.py` | Golden generator (run in the `v2-gate-0-base` worktree) |
| `../golden/gate4_v1_infer.npz`, `gate4_v1_infer.json` | C1 golden (map as stored, coordinates) and provenance; code = base + unpad fix (human decision) |

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| G4-C1 v1 lock | `test_G4_C1_v1_infer_tomos_matches_base_golden` |
| G4-C2 `plan_inference` | `test_G4_C2_v1_one_level[4 cases]`, `test_G4_C2_v2_cases_from_G2_C2[9 cases]`, `test_G4_C2_overlap_rule[10->16, 28->28, 40->32, 20.2->21]`, `test_G4_C2_infer_classes_drops_the_empty_level` |
| G4-C3 One level per class | `test_G4_C3_every_class_is_read_at_exactly_one_level` (200 random tables), `test_G4_C3_two_levels_one_predict_call_maps_for_all_classes` (level-tagging oracle), `test_G4_C3_two_level_phantom_every_class_picked_in_one_call` (blob oracle) |
| G4-C4 Oracle round trip | `test_G4_C4_v1_oracle_round_trip` (no padding), `test_G4_C4_v1_oracle_round_trip_thin_tomogram_with_z_padding` (unpad fix), `test_G4_C4_v2_oracle_round_trip_with_resampling` |
| G4-C5 Cross-class suppression | `test_G4_C5_one_voxel_apart_the_higher_score_stays[2]`, `test_G4_C5_three_radii_apart_both_stay`, `test_G4_C5_distance_uses_factor_times_the_smaller_radius[2]`, `test_G4_C5_factor_argument_is_used`, `test_G4_C5_same_class_peaks_are_untouched`, `test_G4_C5_counts_per_class_pair` |
| G4-C6 Outputs | `test_G4_C6_relion_one_star_per_class_with_original_numbers` (also: no merged STAR), `test_G4_C6_maps_per_class_and_merged`, `test_G4_C6_skip_inference_rereads_the_class_maps`, `test_G4_C6_relion50_warp_imod_names` (RELION 5.0, Warp), `test_G4_C6_imod_one_model_per_class` |
| G4-C7 float16 accumulation | `test_G4_C7_float16_cpu_accumulation` |
| G4-C8 Voxel size 0 | `test_G4_C8_header_voxel_size_zero_raises_naming_apix_override` |
| G4-C9 Benchmark (slow) | `test_G4_C9_synthetic_multisize_tiny_report` |
| G4-C10 `calibrate_thresholds` | `test_G4_C10_true_07_false_04_threshold_in_between`, `test_G4_C10_few_particles_get_none`, `test_G4_C10_unavailable_tomogram_adds_no_false_positives`, `test_G4_C10_ties_pick_the_higher_threshold`, `test_G4_C10_matches_a_reference_sweep` |
| G4-C11 Calibration round trip | `test_G4_C11_calibration_round_trip` |
| G4-C12 `merge_class_maps` | `test_G4_C12_merge_class_maps_random`, `test_G4_C12_specific_voxels`, `test_G4_C12_uint16_above_255` |

49 fast test cases and 1 slow one.

## How the predict path is driven

- `run_predict` sets the global `mainConfig.infer` (tiny defaults plus the test's fields, `use_cuda=False`) and calls
  `tomocpt.predict.run.predict()`, the code behind `tomocpt predict`. Every Dask cluster it starts is replaced by an
  in-process `LocalCluster(n_workers=1, threads_per_worker=1, processes=False)` (patched on `dask.distributed`,
  `dask_cuda` and, if present, on `tomocpt.predict.run`), so monkeypatches and the global config reach the worker.
  The implementer may restructure `run.py` freely as long as clusters come from `dask.distributed.LocalCluster` /
  `dask_cuda.LocalCUDACluster` (or names re-exported in `tomocpt.predict.run`).
- **Errors must reach the caller of `predict()`**: an unknown name in `infer.class_thresholds` (C11) and a header voxel
  size of 0 without `apix_override` (C8) must raise `ValueError` out of `predict()` (a worker exception re-raised by
  `future.result()` is fine). Logging the error and continuing, as the legacy per-tomogram `try/except` does, fails.
- **Oracle networks.** `InferenceModel.load_from_checkpoint` is wrapped so the real checkpoint loads (class table,
  config, conditioning flag) and then `model.model` is replaced by an oracle module with the network contract
  `forward(x, apix=None) -> (logits [B,C,...], hidden)`. A conditioned oracle has a `cond` attribute, so
  `BasePickingModel.forward` passes `apix` through. Any call path through `predict_step` / `forward` / `model.model`
  works. The checkpoint must be loaded with `tomocpt.networks.inference.InferenceModel.load_from_checkpoint`.
  - `NegInputOracle` (spec C4): every channel is `logit(clip(-x_norm, 1e-4, 1 - 1e-4))`.
  - `LevelTagOracle`: constant output `0.15 + 0.02 * apix` in every channel and voxel. A class map then *shows* the
    working voxel size it was read at; a class read at two levels (or averaged over them) gives a different value.
- **Blob phantoms** (`blob_tomogram`): noise-free, one dark blob per particle with the CLAUDE.md 1.3 target profile
  and a sub-voxel centre, plus a bright slab (value 2, smooth edge) at low X covering 8 % of the voxels. The slab sets
  the robust normalisation to `p95 - p5 = 2`, so `-x_norm = profile / 2` (peak 0.5, no saturation; without it the
  peak would be clipped to a flat top). Particles are placed away from the borders that the legacy border exclusion
  (one particle diameter) removes, and the small blobs are more than one large diameter from the large blob.
- The oracle gives the same score to a blob in every channel, so cross-class suppression would be a tie; the
  coordinate tests run with `cross_class_nms=False` (C5 tests suppression itself).

## Choices where the spec is silent

1. **G4-C1 golden: base + unpad fix (human decision, 27 Oct 2026).** The golden is `v2-gate-0-base` code with only the
   legacy `unpad` call corrected (`gate4_common.fixed_unpad`, patched onto `tomocpt.predict.helpers.unpad` at runtime by
   the generator; the base worktree is not edited), so it encodes the correct crop of the Z-padded case. `infer_tomos` is a plain function at `v2-gate-0-base` and runs without Dask
   (`infer_tomos([tomo], None, 0, ckpt, cfg)`, CPU). The golden case: 48x112x112 tomogram at 6 A/px with dark spheres,
   `length=150` (25 px, so the fixed-particle rescale to 20 px runs and Z is padded to 64), `confidence_threshold=0.5`,
   no TTA, `save_prediction_confidence_map=True`. The checkpoint is the cached legacy tiny checkpoint with its network
   weights replaced by `gate3_common.reinit_` (sorted names, seeded `RandomState`), so the golden does not depend on the
   legacy checkpoint's fit; the pickled legacy config and optimizer state are kept. Compared: the confidence-map MRC
   as stored (float16, atol 1e-5, so practically exact) and the `coords` rows `infer_tomos` returns
   (`[z, y, x, score]`, same count, atol 1e-3 voxel). The v1 return value of `infer_tomos` (list of
   `{"name", "coords", "voxel_size"}`) is therefore part of the lock. Torch uses 4 CPU threads in both runs. TTA was
   left out to keep the fast tier short (13 s on CPU). Generated in the worktree at commit `ce6f7b0`:
   `cd ../tomocpt-base && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python ../tomocpt/tests/golden/make_goldens_gate4.py --expect-root $PWD --legacy-ckpt ../tomocpt/tests/.cache/legacy/legacy_picking_ce6f7b0ce8_198804b61e.ckpt`
   (the worktree's tracked files were not modified). Provenance key `code` in `gate4_v1_infer.json` says
   "base + unpad fix (human decision, 27 Oct 2026)".
2. **`plan_inference`.** `class_table` is a list of `ClassSpec`; `scale_cfg` a `ScaleConfig`; `classes` are class-table
   row indices (as `plan_levels.assigned`); v1 gets a one-row table and `length`. v1 `apix_work` is the nominal
   fixed-particle value `length / desired_px` (or `apix_native` when within 1 voxel of the target), rel 1e-6.
   `overlap` is the spec value `min(32, max(16, ceil(max d_vox)))`; note that TorchIO's `GridSampler` needs an even
   overlap, so the sampler may round it up; the tests only read `InferenceLevel.overlap`.
3. **`infer.classes`** is not a `plan_inference` argument, so it is tested through `predict`: the remaining classes keep
   their planned level (planning on the full class table, then dropping classes and empty levels). The test is built so
   re-planning on the reduced table would give another working voxel size (9.375 instead of 15.625 A/px).
4. **`cross_class_suppression` formats.** `peaks_by_class`: `{class key: [N, 4] rows (z, y, x, score)}` (the rows of
   `_extract_peaks_from_mask`); `radii`: `{class key: radius in voxels}`; returned peaks in the same format (row order
   free); `removed_counts[(kept, removed)] = n` = peaks of class `removed` dropped by a peak of class `kept`
   (zero entries may be absent; a nested `{kept: {removed: n}}` dict is also accepted). "Closer than" is strict.
5. **`merge_class_maps`** takes numpy `maps [C, D, H, W]`, `class_numbers` and `thresholds` as sequences in channel
   order; returns numpy or torch (both accepted); `p_c >= thr_c` passes (the spec's wording). The uint8/uint16 rule is
   checked on the returned array. **In the MRC file** the class map is only checked to be an integer dtype: `mrcfile`
   stores uint8 data as mode 6 (uint16), so a uint8 MRC cannot be written with the pinned package.
6. **Output names (v2).** RELION 3.1/5.0 and Warp: `<stem of predictions_coord_filename>_<class name>.star`
   (`picks.star` -> `picks_small.star`), no file named `predictions_coord_filename` (no merged STAR). IMOD:
   `<tomogram stem>_<class name>.mod` (the v1 IMOD name is `<tomogram stem>.mod`). Maps: `<stem>_<class name>.mrc`,
   `<stem>_heatmap.mrc` (float32), `<stem>_classmap.mrc`, native shape and voxel size. Each RELION file has
   `rlnClassNumber` equal to that class's original number (3 and 7 in the tests) and `rlnAutopickFigureOfMerit` in
   [0, 1], and reads back with `tomocpt.labels.star.Star` (`version` `rln31` / `rln50`).
7. **`skip_inference` (v2)** is run with `weights` set (the class table comes from the checkpoint) and
   `tomogram_file`; it must read `<stem>_<class name>.mrc` from `predictions_dir` (the test copies only the maps there)
   and give the same picks as the full run (same count, within 0.1 voxel: maps may be stored as float16).
8. **IMOD.** `imodmodel` is pinned at 0.1.0 (human decision, 27 Oct 2026), which has `imodmodel.write`; the legacy
   `write_imod_model` uses it. `test_G4_C6_imod_one_model_per_class` requires per-class `.mod` files that
   `imodmodel.read` reads back with that class's picks (the reference uses the library writer).
9. **C7.** Both runs use `accumulate_device="cpu"` and compare every class map and the heatmap (max abs diff <= 1e-3)
   and the picks (same count, within 0.1 voxel). The device choice (`auto`) cannot be observed on a CPU-only run and
   is not tested. Hazard found with the reference: in float16 the Hann weights near the volume border underflow, which
   gives NaN or errors of a few 1e-2 at the border unless the weights are handled in float32 or clamped.
10. **C8.** Tested on a v2 checkpoint through `predict()` (`ValueError` matching `apix_override`); the same tomogram with
    `infer.apix_override=10` runs and writes maps whose header voxel size is 10.
11. **C10 formats.** `picks`, `truths`, `radii` are dicts keyed by class number (`picks[c]` = list per tomogram of
    `(xyz [N,3], score [N])`, `truths[c]` = list per tomogram of `xyz [M,3]` or `None`, `radii[c]` in the unit of the
    coordinates); the result is keyed by class number (int or str accepted). Test scores never sit on a grid value
    (no `>=` / `>` dependence). The default grid must be `0.05, 0.10, ..., 0.95` with `min == 0.05`.
12. **C11.** `calibrate_checkpoint(ckpt, val_tomograms, infer_cfg)` gets one Gate 2 style entry
    (`tomogram_path, coords_path, dataset_id, apix_native, class_available`; CSV columns `x,y,z,class_number,diameter_ang`
    in native voxels) and an `InferConfig` (`cross_class_nms=False`, `calibration_min_particles=1`), with the blob
    oracle. Checked: the `"tomocpt_calibration"` record (`version == 1`, `beta == 4`, `grid` with min 0.05, `classes`
    with `threshold`/`fbeta`/`n_true` for both classes and the true counts, `val_tomograms`); every other checkpoint
    key unchanged (`torch.equal` for tensors, pickle bytes otherwise); `read_calibration` (and `None` without the key);
    strict `InferenceModel` load with the same logits as before. Then the test **edits the stored thresholds** (class 1:
    0.95, class 2: 0.05) and checks that `predict` follows them (class 1 has no picks although
    `confidence_threshold=0.3` would give some), that `class_thresholds="s100:0.3"` overrides them, that an unknown
    name raises `ValueError` naming it, and that the uncalibrated copy uses `confidence_threshold` (0.6: no picks;
    0.3: picks).

## report.json key contract (G4-C9)

`python benchmarks/synthetic_multisize.py --tiny` (run from the repository root, no other arguments) must exit 0 and
write `benchmarks/out/<run>/report.json` and `report.md`. `report.json`:

```
{
  "benchmark": "synthetic_multisize",
  "tiny": true,
  "classes": [{"number": int, "name": str, "diameter_ang": float}, ...],          # >= 2 classes
  "arms": {
    "a" | "b" | "c" | "d": {                                                      # all four required
      "wall_time_per_tomogram_s": float >= 0,
      "peak_gpu_memory_mb": float >= 0 or null,                                   # null without a GPU
      "cross_class_suppression": {"<kept class name>-><removed class name>": int >= 0, ...},   # may be {}
      "scoring": {
        "global_0.3": {"per_class": {<every class name>: CLASS_ENTRY}},          # every arm
        "calibrated": {"per_class": {<every class name>: CLASS_ENTRY}}           # arms b, c, d
      }
    }
  }
}
CLASS_ENTRY = {
  "fbeta":     {"mean": float in [0,1], "sd": float >= 0},                        # beta = 4, over seeds
  "precision": {"mean": ..., "sd": ...},
  "recall":    {"mean": ..., "sd": ...},
  "threshold": float,                                                             # the threshold used
  "threshold_source": "class_thresholds" | "calibration" | "confidence_threshold"
}
```
Under `global_0.3` every class has `threshold == 0.3` and `threshold_source == "confidence_threshold"`. Under
`calibrated`, a class whose calibration is `None` reports the fallback threshold and its source. The
`cross_class_suppression` key format is checked only as `str -> int`. `check_report` in `test_g4_benchmark.py` is the
executable form of this contract.

## Results on HEAD (`v2-gate-3-passed`, before Gate 4)

`pytest tests/gate4 -m "not slow"`: 48 failed, 1 passed, no collection errors (22 s). Passing on HEAD:
`test_G4_C4_v1_oracle_round_trip` (legacy behaviour, no padding). Expected failures: G4-C1 (the golden includes the
unpad fix, which HEAD lacks), `test_G4_C4_v1_oracle_round_trip_thin_tomogram_with_z_padding` (same defect), and every
v2 test (missing names `plan_inference`, `InferenceLevel`, `cross_class_suppression`, `merge_class_maps`,
`tomocpt.predict.calibration`; v2 checkpoints go through the v1 path, which needs `infer.length`).
The throwaway reference (with the unpad fix for v1 and v2) passes all 49.

## Unpad axis-order defect (fixed in Gate 4, human decision; CLAUDE.md 1.10)

`_predict_single_tomogram` passed the padding tuple of `symmetrize_and_pad` (F.pad order `(x0, x1, y0, y1, z0, z1)`) to
`unpad`, which reads `(z0, z1, y0, y1, x0, x1)`. When an axis is shorter than 64 after the rescale (thin tomograms), the
map was cropped on the wrong axis and stretched back. Gate 4 fixes it for v1 and v2: the C1 golden (Z padded) encodes the
correct crop, `test_G4_C4_v1_oracle_round_trip_thin_tomogram_with_z_padding` checks v1 centres on a Z-padded tomogram
(0.5 native voxel), and the v2 C4 tomogram is Z padded (45 -> 64 working voxels).

## Mutation check

A throwaway reference implementation (scratchpad, not committed) passes all fast tests (48 at the time of the sweep; 49 after the thin-tomogram test was added). Mutations
(environment switch in the reference), each run against the fast tier:

| Mutation | Killed by |
|---|---|
| overlap rule: `max(16, round(d_vox))` (no cap, rounds) | `test_G4_C2_overlap_rule[40->32, 20.2->21]`, `test_G4_C3_every_class_is_read_at_exactly_one_level` |
| class read at every level (maps averaged) | `test_G4_C3_two_levels_one_predict_call_maps_for_all_classes`, `test_G4_C2_infer_classes_drops_the_empty_level` |
| cross-class NMS keeps the lower score | 5 C5 tests |
| cross-class NMS uses the larger radius | `test_G4_C5_distance_uses_factor_times_the_smaller_radius[4.0-False]` |
| calibration ties pick the lower threshold | all 5 C10 tests and C11 |
| unavailable tomogram counts false positives | `test_G4_C10_unavailable_tomogram_adds_no_false_positives` |
| classmap ignores the class thresholds | `test_G4_C6_maps_per_class_and_merged` (first version of the test missed it: Fourier ringing put a few 0 voxels in the map; strengthened to check every voxel below / above 0.3) |
| `accumulate_dtype="float16"` accumulates in bfloat16 | `test_G4_C7_float16_cpu_accumulation` |
| `apix_override` missing from the error message | `test_G4_C8_header_voxel_size_zero_raises_naming_apix_override` |
| a merged RELION STAR is also written | `test_G4_C6_relion_one_star_per_class_with_original_numbers`, `test_G4_C6_skip_inference_rereads_the_class_maps` |
| `predict` ignores the stored calibration | `test_G4_C11_calibration_round_trip` |

Bugs of the first reference version that the tests caught (kept as evidence that they discriminate): the legacy
`unpad` padding order (C4 v2, C3), float16 Hann weights underflowing at the border (C7: NaN, then 0.05 error),
missing IMOD output (C6 IMOD).
