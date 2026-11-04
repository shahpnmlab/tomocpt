# Gate 4e tests (configurable early stopping, schedule-free RAdam; TODO 3b.2, 3b.3)

Spec: `.claude/agents/04e_AGENT_TRAINING_SCHEDULE.md`.

| Criterion | Tests |
|---|---|
| G4e-C1 defaults | `test_g4e_schedule.py::test_G4e_C1_*` |
| G4e-C2 early stopping fields | `test_G4e_C2_early_stopping_fields` |
| G4e-C3 schedule-free optimiser | `test_G4e_C3_*` |
| G4e-C4 train / eval modes, checkpoint | `test_G4e_C4_fit_switches_modes_and_saves_averaged_weights` |
| G4e-C5 missing package | `test_G4e_C5_*` |

Callbacks are inspected through `tomocpt.training.helpers._setup_callbacks(config, model,
checkpointer)` with the global config applied. C4 fits a tiny pipeline-v2 model (Gate 3 helpers)
for 2 epochs on CPU with `ModelCheckpoint(monitor="val_loss")`, as `tomocpt train` does.
