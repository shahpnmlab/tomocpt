# Gate 10 test dispute: `tests/cli_slim/test_cli_slim.py::test_help_lists_only_everyday_flags`

**Status.** 3 failures (parametrised over `prepare_data`, `train`, `predict`), caused by the spec's
required help sentence, not by a new flag. The frozen test was not edited.

## What the spec asks
`10_AGENT_CONFIG_LEVELS.md`, section 2: "The `HELP_EPILOG` of the commands (in `cli_options.py`)
adds one sentence: write all options with `tomocpt init --level advanced` (or `diagnostic`), and
see `docs/options.md` / `docs/tuning.md`. No other change to the flags of `prepare_data`, `train`,
`predict`."

Implemented in `tomocpt/configManager/cli_options.py`:
```
... Write every option with its help to a commented YAML with tomocpt init --level advanced
(or --level diagnostic), and see docs/options.md and docs/tuning.md.
```

## Why the frozen test fails
`_help_options` collects **every** whitespace token of the whole `--help` output that starts with
`--`, including the epilog text, and treats it as an option of the command:
```python
for line in out.splitlines():
    for tok in line.replace("│", " ").split():
        if tok.startswith("--"):
            opts.add(tok.split("=")[0].rstrip(","))
```
The epilog now names `init`'s option `--level`, so the test reports
`prepare_data --help shows non-everyday flags: ['--level']` (likewise `train`, `predict`).
`prepare_data`, `train` and `predict` have no `--level` option: `CliRunner().invoke(app, ["train",
"--model_dir", "/m", "--level", "advanced"])` exits 2 with "No such option", and apart from
`--level` the tokens the scan finds in `train --help` are exactly the everyday flags, the required
fields, `--config_file`, `--config_merge_preference` and `--help` (checked 22 Nov 2026).

Any wording that tells the user the actual command (`tomocpt init --level advanced`) contains the
token `--level`. Avoiding it (e.g. "init's level option") would satisfy the test but not the spec's
sentence, and a user could not copy the command.

## Proposed resolution (orchestrator / human)
Either (a) is preferred:
- (a) The test scans only the Options panel, or ignores the epilog: e.g. strip
  `cli_options.HELP_EPILOG`'s words from `out` before the scan, or stop at the first line of the
  epilog; or allow `--level` in `extra` explicitly as text of the epilog.
- (b) Drop `--level` from the epilog sentence (spec change: "write all options with
  `tomocpt init` at level advanced").

Gate 10's own test `tests/gate10/test_g10_init.py::test_G10_C15_epilog_points_to_levels_and_docs`
asserts the spec's sentence (`"tomocpt init --level advanced" in HELP_EPILOG`).

## Resolution (human decision 22 Nov 2026)
Option 1: the epilog no longer names `--level`; it says "tomocpt init at level basic, advanced or diagnostic
(see tomocpt init --help for the levels)". The frozen cli_slim test is unchanged and passes. Gate 10 test C15
(not frozen) updated to the new wording.
