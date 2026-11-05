# 04g_AGENT_LR_PLATEAU.md — Gate 4g: LR plateau matched to early stopping (TODO 3b.2a)

Added 5 Nov 2026 by human decision: "give the scheduler a fixed threshold, matched to early
stopping, and a shorter patience (about 5), so the rate halves once or twice before training stops."

## Why
`ReduceLROnPlateau` (Section 1.6) uses patience `int(1.5 * 6) = 9` and PyTorch's default relative
threshold 1e-4 (about 1.5e-6 at `val_loss_smooth` ≈ 0.015), while `EarlyStopping` uses an absolute
`train.early_stop_min_delta`. A slowly improving `val_loss_smooth` (1e-5 to 3e-5 per epoch, smoother
still under EMA validation) counts as progress for the scheduler but not for early stopping, so in
every run so far (Gate 7, the 3b pilots: 59 epochs at 4e-4) the learning rate never dropped before
early stopping ended training.

## Design
- `train.lr_plateau_match_early_stop: bool = False`. When True, the scheduler is
  `ReduceLROnPlateau(opt, mode="min", factor=FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS (0.5),
  patience=train.lr_plateau_patience, threshold=train.early_stop_min_delta, threshold_mode="abs",
  cooldown=max(1, PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS // 4) (1))`, monitor `val_loss_smooth`: the same
  improvement test as early stopping, with a shorter patience, so the rate halves before training stops.
- `train.lr_plateau_patience: int = 5`, used only when matched. With matching on, it must be smaller
  than `train.early_stop_patience` (ValueError otherwise, raised when the optimiser is configured).
- With `train.schedule_free` there is no scheduler; `lr_plateau_match_early_stop=True` together with
  `schedule_free=True` raises ValueError.
- Default (False): the legacy scheduler unchanged (invariant 1). Old checkpoints backfill the defaults.

## Allowed write paths
`tomocpt/defaultConfigs/train_config.py` (two fields), `tomocpt/networks/baseModel.py`
(`configure_optimizers`), `README.md`, `.claude/reports/gate_04g_report.json`.

## Pass criteria
- **G4g-C1** Defaults: the scheduler is `ReduceLROnPlateau` with patience 9, factor 0.5, cooldown 1,
  threshold 1e-4, `threshold_mode="rel"`, monitor `val_loss_smooth` (legacy).
- **G4g-C2** Matched: threshold = `early_stop_min_delta`, `threshold_mode="abs"`, patience =
  `lr_plateau_patience`, factor 0.5, cooldown 1, monitor `val_loss_smooth`; fine-tuning parameter
  groups unchanged.
- **G4g-C3** ValueError: matched with `lr_plateau_patience >= early_stop_patience`; matched with
  `schedule_free`.
- **G4g-C4** On the recorded `val_loss_smooth` curve of the 3b EMA pilot, the matched scheduler
  (min_delta 1e-4, patience 5) halves the rate at least once before Lightning's `EarlyStopping`
  (min_delta 1e-4, patience 10) stops, and the legacy scheduler never does.
