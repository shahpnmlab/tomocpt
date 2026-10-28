# 03c_AGENT_CONFIG_SNAPSHOT.md — Gate 3c: stop checkpoint loads leaking into each other

Added 28 Oct 2026 by human decision (found at Gate 4, needed before Gate 5, where SSL and
picking checkpoints are loaded in one process).

## Problem
`BaseModel.__init__(config=<checkpoint config>)` computes "changes made on the command line" as
`compare_dataclasses(_mainConfigNoChanges, mainConfig)` on the **live** global `mainConfig`, then
writes the checkpoint's `train` section into `mainConfig.train` (deliberate: legacy code reads
`mainConfig.train`). A second checkpoint load in the same process therefore treats the first
checkpoint's train settings as CLI changes and applies them to the second checkpoint.
Also, `config=None` makes `hparams.config` the global object itself (an alias), so later global
changes alter an existing model's config (seen at Gate 3).

## Allowed write paths
- `tomocpt/configManager/configManager.py` (the snapshot only)
- `tomocpt/networks/baseModel.py` (`__init__` config handling only)
- `.claude/reports/gate_03c_report.json`

## Tasks
1. `configManager.py`: public functions `set_cli_snapshot(config)` (stores a deep copy),
   `get_cli_snapshot() -> Optional[MainConfig-like]` and `clear_cli_snapshot()`. The CLI command
   wrapper calls `set_cli_snapshot(mainConfig)` once, after YAML and CLI values are applied and
   before the command function runs.
2. `BaseModel.__init__` with a checkpoint config uses `get_cli_snapshot()` as `configAfterCli`
   when a snapshot exists; without a snapshot it uses the live `mainConfig` (legacy behaviour,
   so library and test code that never went through the CLI is unchanged). The injection into
   `mainConfig.train` stays.
3. `BaseModel.__init__(config=None)` stores a deep copy of `mainConfig`, not the global object.

## Pass criteria
- **G3c-C1** With a snapshot set, loading checkpoint A then checkpoint B (different train
  settings, e.g. `n_epochs`, `batch_size`, `data_pipeline_version`/class table) in one process
  gives B its own settings, and both load with `strict=True`; the order A→B and B→A agree.
- **G3c-C2** A value changed on the command line (in the snapshot) still overrides the
  checkpoint value, on the first and on the second load.
- **G3c-C3** Without a snapshot, behaviour equals the legacy code (single load: same config as
  `v2-gate-4-passed`).
- **G3c-C4** `config=None`: the model's `hparams.config` is not the global object; changing
  `mainConfig` afterwards does not change it; its values equal `mainConfig` at construction.
- **G3c-C5** The CLI wrapper sets the snapshot: invoking a registered command (with a stub
  function) leaves `get_cli_snapshot()` equal to the YAML+CLI-applied config.
- All earlier tests pass unchanged (legacy equivalence, checkpoint compatibility).
