# Gate 9 tests (training recipe for the DS-10490 boss run)

Spec: `.claude/agents/09_AGENT_BOSS_RECIPE.md`. All CPU, no `slow` tests.

| Criterion | Tests |
|---|---|
| G9-C1 defaults reproduce current behaviour | `test_G9_C1_*` in every file |
| G9-C2 LR warmup | `test_g9_lr_warmup.py::test_G9_C2_*` |
| G9-C3 contrastive weight ramp | `test_g9_contrastive.py::test_G9_C3_*` |
| G9-C4 bank age limit, state_dict compatibility | `test_g9_bank.py::test_G9_C4_*` |
| G9-C5 repeat-factor cap (chunk and on-the-fly samplers) | `test_g9_repeat_factor.py::test_G9_C5_*` |
| G9-C6 config backfill | `test_g9_repeat_factor.py::test_G9_C6_*` (and the Gate 1 sweep) |
