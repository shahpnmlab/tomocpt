# Gate 3c tests (CLI config snapshot)

The Test Author wrote these tests from `.claude/agents/03c_AGENT_CONFIG_SNAPSHOT.md` before the implementer started.
Fast tier only: `pytest tests/gate3c` (CPU, about 5 s after the legacy checkpoint is cached). There is no slow tier.

The snapshot API (`set_cli_snapshot`, `get_cli_snapshot`, `clear_cli_snapshot` in
`tomocpt.configManager.configManager`) is imported inside the test functions, so the module imports on code without it.

## Files

| File | Content |
|---|---|
| `conftest.py` | `tests/` on `sys.path`, seeds; autouse: `applied_main_config` restore and `clear_cli_snapshot()` before and after each test; session fixture `ckpts` |
| `test_g3c_config_snapshot.py` | API, C1 to C5 |

`ckpts` holds three tiny picking checkpoints, loaded with `BasePickingModel.load_from_checkpoint(strict=True)` on CPU:
- `A`: v1 tiny config, `n_epochs=3`, `batch_size=4`;
- `B`: v2, two classes (`gate3.g3_helpers.v2_config`, classes A 150 Å and B 300 Å, 2 output channels), `n_epochs=9`, `batch_size=6`;
- `legacy`: the `v2-gate-0-base` tiny picker (`utils.legacy.legacy_checkpoint`).

`A` and `B` are saved without a Trainer (a `state_dict` + `hyper_parameters` + version dict), which is enough for `load_from_checkpoint`.

## Criteria and tests

| Criterion | Test(s) |
|---|---|
| API (task 1) | `test_G3c_api_set_get_clear_deep_copy`: `set` stores a deep copy, `get` returns it, `clear` gives `None` |
| G3c-C1 | `test_G3c_C1_sequential_loads_keep_own_settings` (A then B and B then A: each keeps its own `n_epochs`, `batch_size`, `data_pipeline_version`, class table; configs agree across orders; `mainConfig.train` still receives the last checkpoint's train section), `test_G3c_C1_legacy_and_v2_checkpoint_in_one_process` |
| G3c-C2 | `test_G3c_C2_cli_change_overrides_on_both_loads[A→B, B→A]`: snapshot `n_epochs=42` (live global at defaults) wins on both loads; other fields keep the checkpoint values |
| G3c-C3 | `test_G3c_C3_no_snapshot_uses_live_global`: without a snapshot, the loaded config equals the legacy formula `update_config_with_changed_values(backfill(ckpt), _mainConfigNoChanges, mainConfig)` for A, B and legacy |
| G3c-C4 | `test_G3c_C4_config_none_is_a_copy`: `hparams.config` (and its sections) is not the global; equal at construction; unchanged after global edits |
| G3c-C5 | `test_G3c_C5_cli_wrapper_sets_snapshot`: real `ConfigurableApp.register_command` + `typer.testing.CliRunner`, YAML (`n_epochs=17`, `batch_size=5`) + `--batch_size 7` + `optimizer.lr=0.0001`; the stub sees a snapshot (a copy, not the global) equal to the applied `mainConfig`, before it runs |

## Notes

- The wrapper reads `--config_file` from `sys.argv` when the command is registered, so the C5 test sets `sys.argv`
  (monkeypatch) before registering, and passes the same arguments to `CliRunner`.
- Two stub commands are registered so the first is invoked by name.
