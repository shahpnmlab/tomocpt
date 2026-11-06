# Gate 4h tests (relative improvement threshold; TODO 3b.2b)

Spec: `.claude/agents/04h_AGENT_REL_THRESHOLD.md`.

| Criterion | Tests |
|---|---|
| G4h-C1 default (absolute, legacy) | `test_g4h_rel_threshold.py::test_G4h_C1_*` |
| G4h-C2 relative callback and scheduler | `test_G4h_C2_*` |
| G4h-C3 invalid values | `test_G4h_C3_*` |
| G4h-C4 replay of the 3b EMA curve, scale invariance | `test_G4h_C4_*` |
| G4h-C5 state dict round trip | `test_G4h_C5_*` |

C4 reuses the curve in `tests/gate4g/test_g4g_lr_plateau.py` (recorded at a constant LR: trigger
logic only).
