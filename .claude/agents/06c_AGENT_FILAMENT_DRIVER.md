# 06c_AGENT_FILAMENT_DRIVER.md — Gate 6c: filament driver, fixed split, pictures (TODO 3c, handoff step 3)

Added 6 Nov 2026. Human decisions: fixed runs for DS-10490 (`sample_1`–`sample_5` train, `sample_6`,
`sample_7` validation; `HANDOFF_filaments.md` 5.1a) and pictures of target and prediction ("so I can get
some visual feedback of whats going on").

## Design
- **`train.val_stems: Optional[str] = None`**, comma-separated tomogram stems (pipeline v2). None = the
  random split. `dataManager.chunking.split_rows_by_stems(df, stems)`: unknown stem or an empty train
  split raises ValueError. Cannot be combined with `cv_folds` (ValueError). Recorded in `chunk_layout.json`
  as sorted `val_stems`, `split` = `"fixed"`; `_prepare_data_if_needed` refuses existing chunks whose
  recorded stems differ. Absent from the layout when None (invariant 1).
- **Driver recipe.** `benchmarks.filaments.apply_recipe(train)` sets the recipe of the handoff, section 4;
  `_train_arm` uses it. Other drivers are unchanged.
- **Fixed runs.** `--train_runs`, `--val_runs` (one training set): `_stage_fixed` links them under
  `stage/train`; the validation runs are the `held_out` report. Those scores share their tomograms with
  the threshold calibration, so they are slightly optimistic (the report says so).
- **Pictures** (`benchmarks/filament_views.py`): `save_filament_view`, `save_threshold_curve`,
  `dump_training_images`; written to `<arm>/views/`. A failure to draw is logged and never ends a run.

## Allowed write paths
`tomocpt/defaultConfigs/train_config.py` (one field), `tomocpt/dataManager/chunking.py` (split),
`tomocpt/training/helpers.py` (`_prepare_data_if_needed`), `benchmarks/filaments.py`,
`benchmarks/filament_views.py`, `README.md`, `CLAUDE.md` (tables), `TODO_v2.md`, `HANDOFF_filaments.md`,
`.claude/reports/gate_06c_report.json`.

## Pass criteria
- **G6c-C1** `split_rows_by_stems`: exact split, rows of a stem stay together, ValueError cases.
- **G6c-C2** Default None: `TrainConfig().val_stems` is None; `_do_chunking_v2` wiring and refusal of a
  mismatch (checked at the `_prepare_data_if_needed` level); combination with `cv_folds` raises.
- **G6c-C3** `apply_recipe` sets every field of the handoff recipe, and they exist on the config.
- **G6c-C4** `_stage_fixed` links the named runs; a missing run or an overlap raises.
- **G6c-C5** `classify_traces` separates matched, false and missed; `target_map` is 1 on the centreline;
  `save_filament_view` and `save_threshold_curve` write PNG files.
- **G6c-C6** `dump_training_images` writes the first and last epoch of each tag.
