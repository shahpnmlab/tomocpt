# 04h_AGENT_REL_THRESHOLD.md — Gate 4h: relative improvement threshold (TODO 3b.2b)

Added 6 Nov 2026 by human decision: make the early-stopping and LR-plateau thresholds relative
before the filament runs, so the recipe carries over to datasets with a different loss scale.

## Why
`train.early_stop_min_delta` is absolute. 1e-4 suits CZII (`val_loss_smooth` ≈ 0.015, so 0.68 %).
A run whose loss is 10× larger would need ten times the gain per epoch to count as progress, and one
whose loss is 10× smaller would stop almost at once. Gate 4g's matched scheduler copies the same
absolute value.

## Design
- `train.early_stop_rel_delta: Optional[float] = None`. None = legacy (absolute `early_stop_min_delta`).
  When set (0 < value < 1), an epoch counts as an improvement only if
  `val_loss_smooth < best · (1 − early_stop_rel_delta)`, and `early_stop_min_delta` is ignored.
- Early stopping: `tomocpt.training.callbacks.RelativeEarlyStopping(EarlyStopping)` with argument
  `rel_delta`; it sets `min_delta` from the current best before Lightning's test (0 while the best is
  not finite). Mode `min` only. Lightning's state dict (wait count, best) is unchanged, so resume works.
- Matched scheduler (Gate 4g on): `threshold=early_stop_rel_delta, threshold_mode="rel"` (PyTorch's
  relative test for mode `min` is the same inequality). Without matching the legacy scheduler is kept.
- ValueError (in `_setup_callbacks`) if the value is not in (0, 1).
- Considered recipe value 0.007: 1e-4 / 0.0147 (CZII EMA floor). On the recorded 3b EMA curve the
  matched scheduler halves at epochs 41 and 48 and early stopping fires at 52 (best 0.01469 at
  epoch 42; the absolute setting gives 0.01463 at 48, a difference 7× below run-to-run noise).

## Allowed write paths
`tomocpt/defaultConfigs/train_config.py` (one field), `tomocpt/training/callbacks.py`,
`tomocpt/training/helpers.py` (`_setup_callbacks`), `tomocpt/networks/baseModel.py`
(`_lr_plateau_scheduler`), `README.md`, `.claude/reports/gate_04h_report.json`.

## Pass criteria
- **G4h-C1** Default None: early stopping is a plain `EarlyStopping` with `early_stop_min_delta`;
  the matched scheduler stays absolute (Gate 4g tests still pass).
- **G4h-C2** Set: the callback is `RelativeEarlyStopping` with that `rel_delta` and
  `early_stop_patience`; with matching on, the scheduler has `threshold=rel`, `threshold_mode="rel"`.
- **G4h-C3** ValueError for values ≤ 0 or ≥ 1.
- **G4h-C4** Replay of the 3b EMA curve with 0.007, patience 5 / 10: LR drops at 41 and 48, stop at
  52; the curve scaled by 100 or by 0.01 gives the same epochs (scale invariance).
- **G4h-C5** `RelativeEarlyStopping` state dict round-trips (resume) and keeps `best_score`.
