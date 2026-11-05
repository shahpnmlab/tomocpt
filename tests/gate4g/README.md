# Gate 4g tests (LR plateau matched to early stopping; TODO 3b.2a)

Spec: `.claude/agents/04g_AGENT_LR_PLATEAU.md`.

| Criterion | Tests |
|---|---|
| G4g-C1 defaults (legacy scheduler) | `test_g4g_lr_plateau.py::test_G4g_C1_*` |
| G4g-C2 matched scheduler | `test_G4g_C2_*` |
| G4g-C3 invalid combinations | `test_G4g_C3_*` |
| G4g-C4 replay of the 3b EMA pilot curve | `test_G4g_C4_*` |

C4 replays the recorded `val_loss_smooth` curve (59 epochs) through the configured scheduler and
Lightning's `EarlyStopping`; the curve was recorded at a constant LR, so it checks the trigger logic,
not the effect of the drop.
