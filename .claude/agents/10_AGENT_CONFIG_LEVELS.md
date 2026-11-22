# Gate 10 — Option levels, commented `init`, and user documentation

Human request (22 Nov 2026): users configure runs through a config file or CLI options, not through the
benchmark drivers. Today nothing tells them which settings exist, what they do, or when to change them.
Expose the advanced options trivially **without** making `--help` of `prepare_data`, `train` and `predict`
longer. Human decisions:
- **No recipes or presets inside tomocpt.** We cannot anticipate the runs people will make. Worked examples
  live in `docs/` for users to read and adapt.
- **Stratify the settings into levels** (basic, advanced, diagnostic). A user writes a config file at the
  level they want with one CLI flag of `tomocpt init`.
- **A document that explains the options** and how to change their values in which circumstances.

Branch `v2-feat/config-levels` (worktree `.claude/worktrees/config-levels`), from `v2-dev`. Do not merge or
push. A long training run (the DS-10490 boss run) is using the GPU and the main checkout: do not run GPU
jobs, and do not touch the main checkout or `benchmarks/out/`.

## 1. Option levels

New module `tomocpt/configManager/option_levels.py`, one registry for the whole config tree:

- `LEVELS = ("basic", "advanced", "diagnostic")`, ordered; a higher level includes the lower ones.
- `option_level(section: str, dotted_field: str) -> str` for every field of `MainConfig` under `shared`,
  `prepData`, `train`, `infer` (the `dotted_field` is relative to the section, e.g. `discriminative.embed_dim`).
- **basic** = the fields in `cli_options.CLI_FLAGS` for that section, plus required fields (no default).
  `CLI_FLAGS` stays the single source of the basic set (do not duplicate the list).
- **diagnostic** = an explicit list in `option_levels.py`: settings that only produce extra output for
  inspection or debugging, never change training or the picks (e.g. `prepData.write_filament_masks`,
  image or view logging, extra maps). Propose the list from the code; report it for human review.
- **advanced** = every other user-facing field (the default when a field is in neither list).
- Fields that are not user settings (unannotated internals such as `train.optimizer._target_`, values copied
  from `shared`) are **internal**: never written by `init`, not documented as options. `option_level`
  returns `"internal"` for them. List them in the report.

The 40 non-`Annotated` fields today (e.g. `train.optimizer.weight_decay`, `betas`,
`train.network.SwinUNETR.drop_rate`, `prepData.ALPHA_FOR_DROPPING_EMPTY_CUBES`, the `UNET` sub-config) are
not written by `init` today. **Human decision (22 Nov 2026): all of them get help text.** Wrap each in
`Annotated[..., typer.Option(help=...)]` with the **same default and type** (help text only; no behaviour
change, no new CLI flag since they are not in `CLI_FLAGS`). Then decide per field whether it is a user
setting (a level) or internal (help text kept, but not written by `init` and not in `docs/options.md`).
Check that wrapping does not change how `init` or the config loader treats them beyond what this spec asks
(e.g. a field that `init` now writes must still round-trip). Report the decision per field.

A test sweeps every `MainConfig` field and checks it has a level, and that every non-internal field has a
non-empty help string (so a new gate's field cannot be added without help and a level).

## 2. `tomocpt init --level`

`tomocpt init` gains one option, `--level basic|advanced|diagnostic` (default `basic`). It writes the
sections `shared`, `prepData`, `train`, `infer` with every field up to that level:

- each field preceded by a YAML comment with its help text (wrapped at ~100 characters), its type and, for
  enums, the allowed values; the value written is the current default (`???` for required fields, as now);
- within a section, fields are grouped by level under a comment header (`# --- advanced ---`), nested
  sections (`particles`, `discriminative`, `otf`, `augmentation`, `scale`, `ssl`, `network`, ...) kept nested;
- a short header comment at the top: how levels work, `--config_file`, `key=value` overrides, and pointers
  to `docs/options.md`, `docs/tuning.md`, `docs/examples/`.

Requirements (tests):
- The written file loads with `--config_file` for `prepare_data`, `train` and `predict` and gives a config
  equal to the defaults, at every level (round trip). In particular a **partial** file (level `basic`)
  must load, with omitted fields at their defaults. If the config loader cannot do this today, fix it in
  `configManager/configManager.py` (minimal change) and say so in the report.
- Level `advanced` contains every basic field; `diagnostic` contains every non-internal field.
- `init` still accepts `key=value` overrides (they appear in the written file).
- Output is deterministic (same file twice).
- The YAML stays valid with help strings containing `:`, `#`, quotes or newlines.

The `HELP_EPILOG` of the commands (in `cli_options.py`) adds one sentence: write all options with
`tomocpt init --level advanced` (or `diagnostic`), and see `docs/options.md` / `docs/tuning.md`. No other
change to the flags of `prepare_data`, `train`, `predict`.

## 3. Generated option reference: `docs/options.md`

Generated from the dataclasses and the registry by `python -m tomocpt.configManager.option_docs`
(writes to stdout or `--out`). One table per section and sub-section: option (dotted, as used in
`key=value`), level, type / allowed values, default, help. Fields in the order of the dataclasses.
A test regenerates it and compares with the committed file (stale docs fail the suite). Internal fields
are left out.

## 4. Hand-written documentation

All in `docs/`; keep the README short and point to these.

### 4.1 `docs/tuning.md` — what to change, when, and why
Organised by task, not by section. For each option discussed: what it does, the default, when to change it,
which direction, the cost, and the evidence from this project when there is some (cite the measurement,
e.g. "CZII 5-fold CV, out-of-fold F4 0.741 -> 0.737"). At least:
- **Data and scale**: pipeline v1 vs v2; band (`band_min_vox`, `band_max_vox`) and levels; `apix_override`;
  multi-class labelling (`class_diameters`, `class_annotated`); border policy; exact coordinates;
  negative masks (membranes).
- **Class head**: `sigmoid` vs `softmax` (softmax fixes same-size species confusion on synthetic data),
  `softmax_scope` `particles` for runs with filaments (filaments train faster; DS-10490 particle F4 tie).
- **Many or rare classes**: `chunk_sampling=repeat_factor`, `repeat_factor_threshold`, `repeat_factor_max`
  (log of factors in train.log), batch size and memory (automatic Swin / Hessian checkpointing).
- **Contrastive and embedding head**: `contrastive_weight` (0.1; 0.5 hurt microtubules), warmups,
  `bank_max_age`, `embed_dim`, `embed_ce_weight`, `prototype_head` (DS-10490 ablation: a tie; opt-in).
- **Optimisation and schedule**: lr, `lr_warmup_steps`, early stopping (`early_stop_rel_delta` 0.007,
  patience), matched LR plateau, EMA vs SWA, schedule-free, `lr_scaling` with batch size / GPUs.
- **Augmentation**: Fourier and missing-wedge (on by default for v2; why: generalisation across datasets),
  flips vs affine rotation, intensity options (off).
- **Speed and memory**: on-the-fly sampling, GPU augmentation, `samples_per_epoch`, dataloader workers,
  predict accumulation device.
- **Fine-tuning**: `fine_tune_from` vs `resume_from`, differential learning rates, what carries over.
- **Inference and thresholds**: calibration (`calibration_beta`, `calibration_min_particles`; calibrate on
  tomograms not used for training, same-tomogram calibration is optimistic), `class_thresholds`,
  peak spacing, cross-class NMS (off by default in v2, and why), TTA, filament tracing settings.
- **Cross-validation and ensembles**: when CV is worth it (few tomograms) and when a fixed split is enough.
Sources: `TODO_v2.md`, the gate reports in `.claude/reports/`, `README.md`, the agent specs, and the
evidence list in section 6 of this spec. Do not invent numbers; when there is no evidence, say so.

### 4.2 `docs/examples/` — worked scenarios
One directory or file pair per scenario: a complete, commented config (`*.yaml`, only the settings that
differ from the defaults plus the required ones) and a short `README.md` (the situation, the commands
`prepare_data` / `train` / `predict` in order, what each non-default setting is for, what to look at after
the run). At least:
1. single species, legacy pipeline (v1);
2. several particle species of different sizes (pipeline v2, softmax head; the CZII setting);
3. many species plus microtubules and actin from masks, membrane negatives (the DS-10490 boss setting:
   translate `experiments/run_boss_ds10490.sh` and `benchmarks/filaments.py`'s recipe into a config file);
4. fine-tuning a trained model on a new dataset with few labels;
5. prediction only, with re-picking from saved maps and per-class thresholds;
6. k-fold CV with a fold ensemble.
Paths in the examples are placeholders (`/path/to/...`). Tests: every example YAML loads into `MainConfig`
with no unknown keys and valid values; example 3's train settings equal the `--train_set` values of
`run_boss_ds10490.sh` plus the driver recipe (compare the parsed values, not the text).

## 5. Ownership, invariants, process

May write: `tomocpt/configManager/{option_levels,option_docs,cli_options,initializer}.py`,
`tomocpt/configManager/configManager.py` (partial-file loading only, if needed), `tomocpt/defaultConfigs/*.py`
(help strings and `Annotated` wrappers only; no default, type or behaviour changes), `docs/**`, `README.md`
(shorten, point to docs), `TODO_v2.md`, `tests/gate10/**`, `.claude/reports/gate_10_report.json`.

- Invariant 1: training, inference and chunk outputs unchanged. Nothing here may change a default.
- No new dependencies (write the commented YAML by hand or with what is pinned).
- No new flags on `prepare_data`, `train`, `predict`; one new option `--level` on `init`.
- Frozen tests are not edited. If one breaks (e.g. it checks `init` output or `--help` text), write
  `.claude/reports/gate_10_test_dispute.md` with evidence instead of editing it.

Process: write the tests in `tests/gate10/` first and commit them (TA step, `TA_TEST_AUTHOR.md`), then the
implementation, then the documents. Run `pytest tests -m "not slow"` (baseline on v2-dev: 1352 passed,
27 skipped, 0 failed). Report with the schema in `00_ORCHESTRATOR.md`: per-field level and internal
decisions for review, the diagnostic list, test counts, commits.

## 6. Evidence to use in `docs/tuning.md`

- CZII (7 tomograms, 6 species), 5-fold CV out-of-fold F4: v2 defaults 0.741 (cross-class NMS off,
  peak spacing 0.5 d); focal 0.730; contrastive 0.742; focal + contrastive 0.737.
- Synthetic sizes (Gate 6 ablation): softmax head fixes sphere-size confusion; CE terms with background hurt peaks.
- Filaments, DS-10490 sample_1..22: actin F1 0.617 (5 train) -> 0.782 (20 train, one level `band_max_vox 37`)
  -> 0.810 with membrane negatives and contrastive 0.1; microtubule F1 ~1.0.
- DS-10490, C = 134 (132 species + MT + actin), Gate 3e: `softmax_scope` `all` vs `particles`, particle F4
  0.277 vs 0.280 cross-calibrated (threshold calibrated on one validation tomogram, scored on the other);
  filaments train faster with `particles`; same-tomogram calibration overstates F4 (0.299 vs 0.277).
- Gate 9b ablation (6144-step fine-tunes): contrastive 0.1 0.278, 0.5 0.272 (hurts microtubule loss),
  embedding CE 0.279, CE + prototype head 0.279: a tie.
- Post-processing on DS-10490: cross-class NMS 0.177, argmax gate 0.245 vs per-class peaks 0.277; every
  suppression costs more recall than it gains precision.
- Memory: C = 134, batch 16 at 64^3: peak ~34-37 GB with Hessian-loss checkpointing; Swin checkpointing
  switches off automatically when it fits; ~3.3-3.9 s per step on one 48 GB GPU.
- Small proteins (49-81 Å at 6.8 Å/px, 7-12 voxels) are confused with each other and inside large
  complexes; rare classes with very few examples are not learned.
