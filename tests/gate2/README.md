# Gate 2 tests (data pipeline v2)

The Test Author wrote these tests from `.claude/agents/02_AGENT_SCALE_DATA.md` before the implementer started.
Fast tier: `pytest tests/gate2 -m "not slow"`. Slow tier: `pytest tests/gate2 -m slow` (one test, needs a CUDA GPU).

New public names are imported inside the test functions, so the v1-equivalence tests that share a module with
new-behaviour tests still run on code that does not have the new names yet.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds, `applied_main_config` restore (autouse), session fixtures: `v1_run` (C1 golden inputs), `v2main_inputs` / `v2main` / `v2main_dense` (main v2 dataset), `chunks_copy` |
| `g2_helpers.py` | Helpers (no tests): v2 input writer (tracking table + `coords/<tomo>.csv`), in-process chunking, chunk readers, `make_data`, augmentation presets |
| `test_g2_v1_equivalence.py` | C1, the v1 parts of C18 |
| `test_g2_scaling.py` | C2, C3, C4, C17 |
| `test_g2_labels.py` | C5 (parsing and table), C6, C7, C12 |
| `test_g2_chunking.py` | C5 (manifest and training start), C8, C9, C10, C11, C13, slow GPU test |
| `test_g2_dataloading.py` | C14, C15, C16, C18 |
| `test_g2_worker_config.py` | C19 (real separate-process Dask cluster, CPU) |
| `../golden/gate2_common.py` | Shared by the golden generator and the C1 test: dataset, controlled chunking environment, summaries |
| `../golden/make_goldens_gate2.py` | Golden generator (run in the `v2-gate-0-base` worktree) |
| `../golden/gate2_v1_chunks.json` | C1 golden (183 kB) |

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| G2-C1 v1 equivalence | `test_G2_C1_v1_chunking_matches_base_golden[default_alpha, keep_all]` |
| G2-C2 Band rule table | `test_G2_C2_band_rule_table[...]` (the 5 listed cases + CZII), `test_G2_C2_group_lengths_table[...]` (the 3 listed group cases + CZII), `test_G2_C2_no_single_value_raises`, `test_G2_C2_czii_one_level_no_resampling`, `test_G2_C2_random_cases_follow_the_rule_and_never_upsample` |
| G2-C3 `plan_levels` | `test_G2_C3_plan_levels_properties[...]`, `test_G2_C3_plan_levels_random`, `test_G2_C3_below_band_class_only_at_native_level` |
| G2-C4 Fourier resampling | `test_G2_C4_cosine_above_new_nyquist_is_removed[...]`, `test_G2_C4_mean_is_kept[...]`, `test_G2_C4_blob_centroid_moves_less_than_0_1_voxel[...]`, `test_G2_C4_round_trip_centroid_moves_less_than_0_25_native_voxel[...]`, `test_G2_C4_trilinear_is_the_legacy_resize`, `test_G2_C4_working_shape_uses_v1_rounding` |
| G2-C5 Class table | `test_G2_C5_parse_tolerates_spaces`, `test_G2_C5_parse_diameters_errors_name_the_entry[...]`, `test_G2_C5_parse_names_duplicate_raises`, `test_G2_C5_build_class_table_order`, `test_G2_C5_class_table_from_config_single_class`, `test_G2_C5_class_table_from_config_multiclass`, `test_G2_C5_class_without_diameter_raises_naming_it`, `test_G2_C5_training_start_records_and_checks_the_class_table` |
| G2-C6 Multi-class prepare_data | `test_G2_C6_multiclass_prepare_data` |
| G2-C7 `render_targets` | `test_G2_C7_render_targets_isolated_particles`, `test_G2_C7_integer_centre_peak_is_one_and_profile_is_max_combined`, `test_G2_C7_overlapping_particles_of_different_classes_keep_both_peaks`, `test_G2_C7_empty_points` |
| G2-C8 v2 chunking | `test_G2_C8_manifest_and_chunk_meta`, `test_G2_C8_one_task_per_tomogram_when_rows_share_it`, `test_G2_C8_batches_match_section_5_1`, `test_G2_C8_gpu_cluster_matches_cpu` (slow) |
| G2-C9 Points storage | `test_G2_C9_stored_points_render_the_tomogram_crop`, `test_G2_C9_points_file_is_small`, `test_G2_C9_dense_and_points_give_the_same_batches` |
| G2-C10 Two levels | `test_G2_C10_two_levels` |
| G2-C11 `scales_per_level=3` | `test_G2_C11_scales_per_level` |
| G2-C12 Border policy | `test_G2_C12_render_skip_drops_and_clip_draws_a_border_particle`, `test_G2_C12_v1_label_skip_is_the_default`, `test_G2_C12_v1_label_skip_explicit_matches_default`, `test_G2_C12_v1_label_clip_draws_the_border_particle` |
| G2-C13 Format guard | `test_G2_C13_v2_training_on_v1_chunks_raises`, `test_G2_C13_v2_training_without_manifest_raises`, `test_G2_C13_v1_training_on_v2_chunks_raises` |
| G2-C14 Wedge preservation | `test_G2_C14_flips_keep_the_missing_wedge[False, True]` (True = `use_mw_aug`; default `AugmentationConfig` apart from `rotation_mode`/`flip_p`; asserts no `RandomElasticDeformation`), `test_G2_C14_affine_mode_unchanged`, `test_G2_C14_flip_only_pipeline_order` |
| G2-C15 Target stays a LabelMap | `test_G2_C15_v1_target_is_a_labelmap`, `test_G2_C15_v2_intensity_and_fourier_augmentations_leave_the_target_unchanged`, `test_G2_C15_linear_label_interpolation_keeps_the_peak` |
| G2-C16 Zoom | `test_G2_C16_zoom_augmentation` |
| G2-C17 `read_apix` | `test_G2_C17_read_apix_header_zero_raises_naming_apix_override`, `test_G2_C17_read_apix_override_wins` |
| G2-C19 Workers use the caller's values (T2.6) | `test_G2_C19_explicit_values_reach_separate_process_workers`, `test_G2_C19_parent_config_reaches_separate_process_workers`, `test_G2_C19_chunk_size_and_stride_kwargs` |
| G2-C18 Repeat-factor sampling | `test_G2_C18_repeat_factor_weights_numbers`, `test_G2_C18_repeat_factor_train_loader`, `test_G2_C18_uniform_gives_the_legacy_chunk_order`, `test_G2_C18_v1_data_order_is_a_plain_shuffled_loader`, `test_G2_C18_v1_with_repeat_factor_raises`, `test_G2_C18_manifest_lists_validation_tomograms` |

That is 62 test functions: 86 fast test cases plus 1 slow one.

## Golden (G2-C1)

`tests/golden/make_goldens_gate2.py` was run in the `v2-gate-0-base` worktree (commit ce6f7b0):
`cd ../tomocpt-base && PYTHONPATH=$PWD python ../tomocpt/tests/golden/make_goldens_gate2.py --expect-root $PWD`
(the script checks that `tomocpt` comes from the worktree). The worktree's tracked files were not modified.

- Input: `make_dataset(n_tomos=2, seed=0)` with the default species (class 1, 120 A, 6 per tomogram, 64x128x128
  at 10 A/px), legacy `prepare_data`, then `do_chunking` with its legacy arguments only.
- Legacy `do_chunking` runs on a Dask cluster whose workers use the GPU when one is visible, and it drops empty
  chunks with an unseeded `random.random()`. `gate2_common.inprocess_chunking` makes the run reproducible
  without changing the code path: the cluster runs in-process with one worker and one thread (tasks run in
  submission order), `torch.cuda.is_available()` returns False (all resizes run on the CPU in fp32), and
  `chunking.random` is replaced by `random.Random(0)`. Two identical runs gave byte-identical summaries.
- Two variants: `default_alpha` (legacy empty-chunk drop, seeded) and `keep_all`
  (`ALPHA_FOR_DROPPING_EMPTY_CUBES = 1e9`, so every grid chunk of both tomograms is compared: 216 chunks).
- Each file under `chunks_dir` (except `.monai_cache`) is recorded by relative path. `.nii.gz` files are
  decoded (gzip bytes are not deterministic) and compared by the sha256 of the float32 array bytes, which
  means the decoded arrays must be identical. `chunk_layout.json` must have the same v1 keys and values; an
  explicit `"pipeline_version": 1` is tolerated (invariant 4: missing means 1). `done_labels_True.txt` must
  have the same text.
- The C1 runs pass no new keyword (all T2.6 kwargs stay None). `keep_all` sets the legacy
  `ALPHA_FOR_DROPPING_EMPTY_CUBES` in the (in-process) caller's config, which T2.6 resolves to the same value.
- For the `default_alpha` variant, v1 must keep drawing its drop decision with `random.random()` from the
  chunking module, in the same order (the v1 code path unchanged, as the spec requires).

## Results on current HEAD (7cb354e = `v2-gate-1-passed`; the same as `v2-gate-0-base` for every Gate 2 path)

`pytest tests/gate2 -m "not slow"`: **5 passed, 70 failed, 11 errors** (the errors are fixtures that call
`do_chunking(..., pipeline_version=2, ...)` or import new modules). The slow test errors as well.

These tests pass on HEAD. They lock legacy behaviour and must keep passing:
- `test_G2_C1_v1_chunking_matches_base_golden[default_alpha]` and `[keep_all]`
- `test_G2_C12_v1_label_skip_is_the_default`
- `test_G2_C15_v1_target_is_a_labelmap`
- `test_G2_C18_v1_data_order_is_a_plain_shuffled_loader`

All the other tests fail or error on HEAD because they test new behaviour:
- New modules or names do not exist: `tomocpt.dataManager.scaling`, `tomocpt.dataManager.metadata`,
  `tomocpt.labels.multiclass`, `tomocpt.defaultConfigs.scale_config`, `ChunkSampling`, `render_targets`,
  `points_to_working`, `repeat_factor_weights` (C2–C11, C16–C18).
- New config fields are missing: `AugmentationConfig(rotation_mode=..., label_interpolation=...)` (C14, C15),
  `PrepdataConfig(label_border_policy=..., multiclass=...)` (C6, C12), and the new `Data` arguments
  (`test_G2_C18_uniform_gives_the_legacy_chunk_order` and `test_G2_C18_v1_with_repeat_factor_raises` fail with
  a TypeError or ImportError). The order part of C18 that can run on legacy code is locked separately by
  `test_G2_C18_v1_data_order_is_a_plain_shuffled_loader`.
- `do_chunking` has no `pipeline_version` / `scale_cfg` / `class_table` / `seed` keywords (every v2 fixture).
- C13: legacy training has no format guard. `test_G2_C13_v2_training_on_v1_chunks_raises` and
  `..._without_manifest_raises` fail with "DID NOT RAISE": legacy code ignores the attribute and trains one
  batch (`OVERFIT_N_BATCHES=1` keeps this short).
- C14: `rotation_mode` does not exist (`test_G2_C14_affine_mode_unchanged` also builds
  `AugmentationConfig(rotation_mode="affine")`).
- C19: `do_chunking` has no `alpha_drop_empty` / `chunk_size` / `chunk_stride` keywords (TypeError), and
  with the kwargs left at None the workers ignore the parent's ALPHA and the explicit
  `desired_particle_pixel_size` (13 resized chunks with drops instead of 9 unresized ones).
- `test_G2_C12_v1_label_skip_explicit_matches_default` fails on HEAD only because the `label_border_policy`
  field does not exist yet.

Full fast suite on HEAD (`pytest tests -m "not slow"`): 282 passed (the previous 277 plus the 5 above),
27 skipped, 70 failed, 11 errors. Without `tests/gate2`: 277 passed, 27 skipped (unchanged).

## Check against a reference implementation

To make sure a correct implementation can pass every new-behaviour test, I wrote a minimal throwaway reference
implementation of the Gate 2 spec in a scratch copy of the package (not in the repo). Against it:
`pytest tests/gate2 -m "not slow"` gave 86 passed, and `-m slow` gave 1 passed (after the 27 Oct spec update:
flips mode drops elastic deformation, T2.6 worker values; C1 still matches the base golden). Mutations of the reference were
caught as follows:
- chunks storing only particles centred inside them: C9 crop test fails;
- zoom that does not scale `d_vox` in the target: C16 fails;
- trilinear in place of Fourier: 8 C4 cases fail;
- no class-table check at training start: the C5 training test fails;
- particles drawn at rounded centres: one C7 test fails.

## Runtime (RTX A6000 box, 64 cores, CPU tests)

- `pytest tests/gate2 -m "not slow"`: about 95 s with the reference implementation, about 50 s on HEAD
  (most v2 fixtures error out early there). The C1 golden runs (two legacy chunking runs plus decoding about
  640 chunk files) take about 25 s of that.
- C19 starts three separate-process Dask clusters: about 12 s in total with the reference, so it stays in the
  fast tier.
- `pytest tests/gate2 -m slow`: about 5 s.

## Spec ambiguities and how the tests resolve them

1. **`class_annotated` format** (comma or colon list, numbers or names) is not given. C6 uses one class number
   (`class_annotated="2"`), which reads the same under every interpretation.
2. **JSON form of `class_table` rows in `chunk_layout.json`** is not given. Tests accept
   `dataclasses.asdict(ClassSpec)` dicts (`number`, `name`, `diameter_ang`, `kind`) or lists in field order
   (`g2_helpers.table_from_manifest`).
3. **Points-file names** ("`points/<chunk stem>.npy`") and chunk locations are not fixed. C9 does not parse
   names. It renders each points file and finds the TorchIO grid position whose crop of the tomogram-level
   render matches (atol 1e-6), and it requires distinct positions for non-empty chunks and one points file per
   volume file. The tomogram-level points are `points_to_working(coords, apix_native, apix_work, table)`,
   shifted by the v1 symmetric padding of `working_shape(...)`.
4. **Chunk directories**: `<split>/<dir name>__L{level}s{k}/` exactly as the spec writes it. After rows are
   merged, the test accepts any `<dir name>` that contains the tomogram stem, but requires exactly one directory
   per (tomogram, level, k).
5. **Fourier centroid convention** (C4): "moves < 0.1 voxel" is measured against `c * N_out / N_in`. A DFT crop
   keeps voxel 0 at the origin, and coordinates in pixels x apix assume the same, so `points_to_working` can
   scale by the voxel-size ratio. A half-voxel (centre-aligned) convention would fail this test. The round-trip
   test does not depend on the convention.
6. **C15 tolerances** (decided by the human, 27 Oct 2026, now in the spec): d_vox 20, 30 degree rotation, peak
   within 5 % of the analytic profile sampled at the output grid, max abs error <= 0.06, linear closer than
   nearest neighbour. The spec's original 1 % was not achievable (TorchIO trilinear loses 1.3–4.5 %).
7. **C14** (decided, 27 Oct 2026): `rotation_mode="flips"` drops `RandomElasticDeformation` (it moved wedge power
   by up to 0.16). The test uses the default `AugmentationConfig` apart from `rotation_mode`/`flip_p`, asserts
   that no `RandomElasticDeformation` remains (the `OneOf` keeps `RandomBlur`), and checks that affine mode still
   contains `RandomAffine` and `RandomElasticDeformation`.
8. **C16 "with a fixed s"**: there is no public way to fix `s`. The test reads `s = apix_in / apix_work` per
   sample and checks every relation for that `s` (`class_d_vox` rtol 1e-6, radius within 1 voxel, availability
   at band_max), plus `max(s) > 1.05` over the samples.
9. **C18 loader share**: with threshold 0.1 and the small lattice dataset (about 16 training chunks), the
   repeat-factor share is too close to the uniform share to tell them apart. The test uses threshold 1.0
   (r = 4 for the rare class; expected share about 21 % against 6 % uniform; 10 % tolerance is about 7 sigma at
   20 000 draws). It draws from `train_dataloader().sampler` (it must be a `WeightedRandomSampler` with weights
   r and `num_samples = round(sum(r))`, as the spec says) and does not load 20 000 volumes. Chunk class sets
   come from `dataset_training[i]` (no spatial augmentation). The particles are on a lattice where each one is
   fully inside or fully outside every chunk.
10. **Where the format and class-table guards live** is not fixed. C13 and the mismatch part of C5 call
    `tomocpt.training.run.train` (tiny CPU config applied to the global `mainConfig`), so a guard in
    `_prepare_data_if_needed` or in `Data.setup` both pass. Training reads the class table through
    `class_table_from_config(mainConfig.prepData)`; the test sets it there.
11. **v2 inputs** for chunking tests are written directly with the columns the spec defines (legacy tracking
    columns plus `coords_path` and `apix_native`; coordinate CSVs with `x, y, z, class_number, diameter_ang`),
    so most chunking tests do not depend on `prepare_data`. C6 goes through `prepare_data`.
12. **"Missing diameter"** (C5) is tested twice: as a parse error (`"1:120,2:"`) and as a class named in
    `class_names` without a diameter (`class_table_from_config` -> ValueError naming it).
13. **CZII diameters** are not in the repo. C2 uses the public CZII challenge radii x 2:
    120, 130, 180, 300, 260, 270 A at 10.012 A/px.

## Notes on legacy code (for the orchestrator)

- **Dask workers do not see the parent's `mainConfig`** (now CLAUDE.md 1.10, fixed by T2.6 and tested by G2-C19). `PreprocessTomogramd` reads
  `mainConfig.prepData.desired_particle_pixel_size`, and the chunk task reads
  `ALPHA_FOR_DROPPING_EMPTY_CUBES` and `CHUNK_SIZE` / `CHUNK_STRIDE`, from the global config inside the worker.
  With the default process-based cluster, workers import a fresh default `mainConfig`. A direct `do_chunking`
  call with `desired_particle_pixel_size=12` and ALPHA = 1e9 set in the parent gave the default-settings output
  (45 chunks, resized to 20 px, empty chunks dropped). The `desired_particle_pixel_size` argument of
  `do_chunking` is never used. The CLI path was not checked. CLAUDE.md 1.2 describes the rescale as using
  `prepData.desired_particle_pixel_size`; that holds only for the default value. Gate 2 v2 code passes values
  explicitly, so this matters for v1 only.
- **Label padding.** `process_label_to_match_tomogram` resizes the raw label straight to the *padded* tomogram
  shape. When an axis needs padding (< 64 after the rescale), the label is stretched over the padding instead of
  being padded, so label and volume no longer line up on that axis. CLAUDE.md 1.2 ("The label is resized to the
  same shape (trilinear)") describes exactly this, but it is not listed as a defect in 1.10.
- **The legacy empty-chunk drop is unseeded** (`random.random()` in the workers), so v1 chunk sets are not
  reproducible between runs. The C1 golden controls this as described above.
