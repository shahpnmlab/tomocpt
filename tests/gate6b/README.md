# Gate 6b tests (tracing threshold calibration; TODO 3c, handoff step 2)

Spec: `.claude/agents/06b_AGENT_FILAMENT_CALIBRATION.md`. All tests are fast (CPU); checkpoint tests
replace `predict_class_maps` with oracle maps. The real-data check is the driver's smoke run.

| Criterion | Tests |
|---|---|
| G6b-C1 metric in the package, exact | `test_G6b_C1_*` |
| G6b-C2 scores on a two-tube map | `test_G6b_C2_*` |
| G6b-C3 threshold choice | `test_G6b_C3_*` |
| G6b-C4 over-full region guard | `test_G6b_C4_*` |
| G6b-C5 manifest | `test_G6b_C5_*` |
| G6b-C6 checkpoint calibration | `test_G6b_C6_*` |
| G6b-C7 particle-only tables unchanged | `test_G6b_C7_*` |
| G6b-C8 ensemble and CV report | `test_G6b_C8_*` |
