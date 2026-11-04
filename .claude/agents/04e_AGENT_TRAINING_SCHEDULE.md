# 04e_AGENT_TRAINING_SCHEDULE.md — Gate 4e: configurable early stopping, schedule-free RAdam (TODO 3b.2, 3b.3)

Added 4 Nov 2026 by human decision (TODO 3b; dependency `schedulefree==1.4.1` approved 4 Nov 2026
as an optional extra).

## Why
In all 15 Gate 7 CZII CV folds `EarlyStopping(monitor="val_loss_smooth", patience=10,
min_delta=0.005)` stopped training at epoch 25–28 while `val_loss_smooth` (about 0.02) was still
improving by less than 0.005 per epoch: the learning rate never left 4e-4 (`ReduceLROnPlateau`,
patience 9, relative threshold 1e-4) and SWA (from epoch 48 of 60) never started. The absolute
threshold of 0.005 is a quarter of the loss itself.

Schedule-free optimisers (Defazio et al. 2024) replace the LR schedule: the optimiser keeps the
interpolated sequence used for gradients and an online average of the iterates; evaluation and
checkpoints use the average. No decay schedule, no plateau detection, no SWA start epoch.
`schedulefree.RAdamScheduleFree` combines this with RAdam's warm-up-free rectification, matching
the default `torch.optim.RAdam`.

## Design
- **D1 Early stopping.** `train.early_stop_min_delta: float = 0.005` and
  `train.early_stop_patience: int = 10` feed `EarlyStopping` (monitor `val_loss_smooth`, absolute
  `min_delta`, mode min). Defaults reproduce the legacy callback exactly.
- **D2 Schedule-free.** `train.schedule_free: bool = False`. When True:
  - `configure_optimizers` builds `schedulefree.RAdamScheduleFree(param_groups, lr=optimizer.lr,
    betas=optimizer.betas, weight_decay=optimizer.weight_decay)` with the same parameter groups and
    LR multipliers as today (fine-tuning keeps `swinViT` x0.01, `encoder*` x0.1, ...), and returns
    **no** LR scheduler;
  - `StochasticWeightAveraging` is not added (the optimiser already averages; SWA would also reset
    the LR through its own scheduler);
  - the optimiser is switched to train mode at the start of every training epoch and to eval mode
    at the start of every validation, test and predict epoch and at the end of fit, so validation,
    `val_loss_smooth`, early stopping and saved checkpoints all use the averaged weights; a
    checkpoint therefore loads for `predict` / fine-tuning like any other;
  - missing package: `ImportError` naming `pip install "tomocpt[schedulefree]"` (or
    `schedulefree==1.4.1`), raised when the optimiser is built.
  - `optimizer._target_` other than `torch.optim.RAdam` with `schedule_free=True` raises ValueError.
- Default (`schedule_free=False`): optimiser, scheduler and callbacks unchanged (invariant 1).
- Checkpoints from before this gate load with the defaults (backfill).
- `pyproject.toml`: optional extra `schedulefree = ["schedulefree==1.4.1"]`.

## Allowed write paths
`tomocpt/defaultConfigs/train_config.py` (three fields), `tomocpt/training/helpers.py`
(`_setup_callbacks`), `tomocpt/networks/baseModel.py` (`configure_optimizers`, optimiser mode
hooks), `pyproject.toml` (extra), `README.md`, `.claude/reports/gate_04e_report.json`.

## Pass criteria
- **G4e-C1** Defaults: `_setup_callbacks` gives EarlyStopping(min_delta 0.005, patience 10, monitor
  `val_loss_smooth`, mode min), SWA, and `configure_optimizers` gives `torch.optim.RAdam` +
  `ReduceLROnPlateau` on `val_loss_smooth`, as before.
- **G4e-C2** `early_stop_min_delta` / `early_stop_patience` reach the EarlyStopping callback.
- **G4e-C3** `schedule_free=True`: `RAdamScheduleFree`, no scheduler, no SWA callback; from-scratch
  one group at `optimizer.lr`; fine-tuning keeps the per-group LR multipliers.
- **G4e-C4** Mode switching: a tiny CPU fit of 2 epochs with `schedule_free=True` runs; during
  training steps the optimiser is in train mode and during validation in eval mode; the saved
  checkpoint's weights equal the model's weights in eval mode at the end of fit and differ from
  the train-mode weights; the checkpoint loads with `InferenceModel.load_from_checkpoint(strict=True)`.
- **G4e-C5** Without the package installed (import blocked), `schedule_free=True` raises ImportError
  mentioning `schedulefree`; `schedule_free=False` does not import it.
- All earlier tests pass unchanged.
