# Gate 8 tests

Spec: `.claude/agents/08_AGENT_PERFORMANCE.md`. Written and frozen by the orchestrator acting as TA
(usage-frugal, as Gates 3b, 3d, 4b, 4c), 9 Nov 2026, on branch `v2-feat/perf`.

## 8a (frozen 7cb10a3)
| Criterion | Test |
|---|---|
| G8a-C1 JSON keys, deterministic CPU loss sequence | `test_g8a_harness.py::test_G8a_C1_step_json_keys_and_determinism` |
| G8a-C1 CLI writes the JSON | `test_g8a_harness.py::test_G8a_C1_step_cli_writes_json` |
| G8a-C2 self-comparison is zero (JSON, NPZ) | `test_g8a_harness.py::test_G8a_C2_compare_self_is_zero` |
| G8a-C2 perturbed JSON / NPZ, missing / added keys | `test_g8a_harness.py::test_G8a_C2_compare_perturbed_json`, `..._npz` |
| G8a-C3 paths | orchestrator check (`git diff --name-only v2-dev...v2-feat/perf`), not a test |

All 8a tests fail on `v2-dev` (b8042e7): `benchmarks.perf` does not exist there (new behaviour).
Runtime on the development machine (CPU, tiny model): ~35 s.
