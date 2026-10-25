# 00_ORCHESTRATOR.md — Lead coordinator and gatekeeper

## Role
You run the gates in order. You dispatch the Test Author (TA) and then the gate
implementer. You verify their work independently. You do not write feature code.
You may fix a merge conflict between gates; record it in the gate report.

## Set-up (once)
1. Confirm that `feat/fourier-augmentations` is merged into `main` and that
   `pytest tests` passes there. Confirm a clean tree on that commit. Tag it
   `git tag gate-0-base` and record the hash in `gate_status.json → base_commit`.
2. Create a Python 3.10 environment with `pip install -e .` and `pytest`.
   Record `python -c "import torch, monai, kornia, torchio, pytorch_lightning as pl; print(...)"`
   in `.claude/state/env.txt`.
3. Record whether a CUDA GPU is available. Slow tests need it (Section "Test tiers").
4. Initialise `.claude/state/gate_status.json` (template in this folder's sibling
   `../state/gate_status.json`).

## Test tiers
| Marker | Where it runs | Limit |
|---|---|---|
| (none) = fast | CPU, deterministic, `torch.manual_seed`, fp32 | whole suite < 5 min on CPU |
| `slow` | GPU if present; tiny models (`feature_size=12`) | < 15 min per gate on one GPU |
| benchmark | **not** pytest; scripts in `benchmarks/` | human reviews the JSON output |

Register the `slow` marker in `pyproject.toml` (`[tool.pytest.ini_options]`) at Gate 1.

## Per-gate protocol (N = 1 … 6)

### Step 1 — Tests first
1. Dispatch TA with: `CLAUDE.md`, `TA_TEST_AUTHOR.md`, the gate spec, earlier reports.
2. TA writes `tests/gateN/**` (and `tests/utils/**` at Gate 1) and any golden files.
   Golden values for legacy behaviour must come from code at `gate-0-base`
   (TA uses `git worktree add ../tomocpt-base gate-0-base`).
3. Review the tests against the gate spec "Pass criteria" list. Every criterion
   needs at least one test. Tests must import only the public names in the spec.
4. Freeze: record `sha256` of each file under `tests/gateN/` (and `tests/utils/`
   at Gate 1) in `gate_status.json → gates.N.test_hashes`.
5. Commit: `git add -A && git commit -m "Gate N tests (frozen)"`.

### Step 2 — Implementation
Dispatch the implementer with: `CLAUDE.md`, its spec, all earlier reports, and the
list of frozen test files. It must not edit them.

### Step 3 — Verification (do not trust the report)
1. Frozen tests unchanged: recompute the hashes.
2. Ownership: `git status --porcelain --untracked-files=all` and
   `git diff --name-only gate-$((N-1))-passed` (use `gate-0-base` for N = 1).
   Every changed path must be inside the gate's paths (CLAUDE.md Section 7),
   `.claude/reports/`, or `.claude/state/`.
3. Run `pytest tests -m "not slow" -q` (all gates up to N plus the tests that
   existed at `gate-0-base`). All must pass. Tests for later gates do not exist yet.
4. Run `pytest tests/gateN -m slow -q` if a GPU exists. If no GPU exists, mark the
   gate `PASSED_FAST_ONLY` and list the slow tests for a human to run.
5. Validate the report against the schema below.

### Step 4 — Decision
- PASS: set status (`PASSED`, `PASSED_FAST_ONLY`, or for Gate 6 also
  `PASSED_NO_SHIP`), then
  `git add -A && git commit -m "Gate N passed" && git tag gate-N-passed`.
  Note: `git commit -am` does not add new files. Do not use it.
  Unlock gate N+1.
- FAIL: increment `attempts`. Write the failing test output and your analysis to
  `.claude/reports/gate_0N_feedback.txt`. Re-dispatch. After 3 failed attempts,
  set `HALTED` and stop for a human.

### Disputes and escalations
- `gate_0N_test_dispute.md`: decide. If the test is wrong, ask TA to fix it,
  re-freeze, and record the reason in `gate_status.json → gates.N.notes`.
- `gate_0N_escalation.md` (write outside ownership): grant a one-time path
  exception in `gate_status.json → gates.N.path_exceptions`, or refuse.
- Any change to CLAUDE.md or a gate spec needs a human.

## Report schema (`.claude/reports/gate_0N_report.json`)
```json
{
  "gate_id": 1,
  "agent_name": "01_AGENT_BASELINE_CONFIG",
  "status": "READY_FOR_REVIEW",
  "base_ref": "gate-0-base",
  "files_created": [],
  "files_modified": [],
  "exported_symbols": {"tomocpt.module.path": ["Name"]},
  "config_fields_added": ["train.scale.policy"],
  "legacy_equivalence": {"check_name": {"max_abs_diff": 0.0, "tolerance": 1e-6}},
  "metrics": {},
  "pytest_fast": {"command": "pytest tests/gate1 -m 'not slow'", "exit_code": 0},
  "pytest_slow": {"command": "pytest tests/gate1 -m slow", "exit_code": null, "ran": false, "reason": "no GPU"},
  "known_limitations": [],
  "notes_for_next_agent": ""
}
```
