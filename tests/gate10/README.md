# Gate 10 tests: option levels, `tomocpt init --level`, documentation

Spec: `.claude/agents/10_AGENT_CONFIG_LEVELS.md`. Fast tests only (CPU, no model, < 30 s).

| Criterion | What | Test |
|---|---|---|
| C1 | `LEVELS` ordered basic, advanced, diagnostic | `test_g10_levels.py::test_G10_C1_levels` |
| C2 | every `MainConfig` leaf has a level (or `internal`) | `test_G10_C2_every_field_has_a_level` |
| C3 | basic = `CLI_FLAGS` + required (section copies of `shared` fields are internal) | `test_G10_C3_basic_is_cli_flags_plus_required` |
| C4 | every non-internal field has help | `test_G10_C4_every_user_setting_has_help` |
| C5 | the 40 formerly unannotated fields: `Annotated`, help, same type and default; no new flag | `test_G10_C5_*` |
| C6 | diagnostic list (explicit, leaves only) | `test_G10_C6_diagnostic_list` |
| C7 | unknown paths raise `KeyError`; `optimizer._target_` internal | `test_G10_C7_unknown_paths_and_internal_examples` |
| C8 | round trip of the written file at every level for prepare_data / train / predict; partial file | `test_g10_init.py::test_G10_C8_*` |
| C9 | a partial file does not leak `discriminative.temperature` into `train.temperature` | `test_G10_C9_partial_file_does_not_leak_nested_keys` |
| C10 | levels nest; internal never written; default level basic | `test_G10_C10_*` |
| C11 | `key=value` overrides appear in the file; `--level` on the CLI | `test_G10_C11_*` |
| C12 | deterministic | `test_G10_C12_deterministic` |
| C13 | valid YAML with awkward help text | `test_G10_C13_yaml_valid_with_awkward_help` |
| C14 | comments carry help and enum choices; level headers; top header | `test_G10_C14_*` |
| C15 | `HELP_EPILOG` sentence; `init --help` shows `--level` | `test_G10_C15_*` |
| C16 | `docs/options.md` regenerated equals the committed file; user settings only | `test_g10_docs.py::test_G10_C16_*` |
| C17 | `docs/tuning.md` covers the spec topics; names only existing options | `test_G10_C17_*` |
| C18 | six examples; each YAML loads with no unknown keys and valid values; placeholder paths | `test_G10_C18_example_loads` |
| C19 | example 3 = `run_boss_ds10490.sh` `--train_set` values + driver arguments + `apply_recipe` | `test_G10_C19_example3_matches_the_boss_run` |

On the code before Gate 10 (`f37ff42`) 40 tests fail and 2 pass:
`test_G10_C5_wrapping_adds_no_cli_flag` (a guard: the wrapped fields must not become flags) and
`test_G10_C8_partial_file_loads_with_defaults` (a partial file without same-named keys already loaded).
C9 fails on the old loader: `merge_config` looked up a missing key in sibling sub-sections, so
`train.discriminative.temperature` also set `train.temperature`.
