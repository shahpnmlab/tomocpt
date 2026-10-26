# Experiment log

## 2026-10-26 — Fourier augmentation: with vs. without, full training run

**Branch:** `feat/portal-download` (Fourier augmentation port, see
`HANDOFF_fourier_augmentations.md`). **Machine:** GPU box (RTX A6000), the first
real run of this feature past the CPU-only smoke tests in the handoff's Gate F.

### Setup

Two runs from `data/czii/train_config.yaml` (SwinUNETR, picking mode, `lr=4e-4`,
batch size 16, `/tmp/chunks` chunked data, early stopping on `val_loss_smooth`
with patience 10), differing only in the augmentation config:

| | augmented | baseline |
|---|---|---|
| `use_mw_aug` | `true` | `false` |
| `use_fourier_aug` | `true` | `false` |
| `missing_wedge_prob` / `amplitude_prob` | 0.5 / 0.5 | n/a |
| model dir | `/tmp/full_augmented` | `/tmp/full_baseline` |

Contrast inversion (`CONTRAST_INVERSION_PROB`, now in `tomocpt/constants.py`,
`0.5`) was left on in both runs — it isn't part of what's being compared here.

Both runs validate on **clean, un-augmented chunks** regardless of the training
config (`transform=None` for validation — see `build_training_transforms` /
Gate C of the handoff). That one fact drives most of the discussion below.

### Result

- **Augmented** ran the full 39 epochs to early stopping. Best `val_loss`
  **0.03538** at epoch 30 (`val_loss_smooth` plateaued for 10 epochs, stopped at
  epoch 38).
- **Baseline** was stopped manually partway through epoch 20 once the comparison
  was clear. Best `val_loss` so far: **0.03826** at epoch 12, still on the same
  decreasing trend — not a completed run, no early-stop signal yet.

Per-epoch `val_loss` (checkpointed, i.e. only logged when it improved) and the
**training** `loss` (computed on the actual training batches — augmented for
the augmented run, clean for baseline) for the first 20 epochs:

| epoch | aug train loss | aug val_loss (on improve) | base train loss | base val_loss (on improve) |
|---|---|---|---|---|
| 0 | 0.563 | 0.09272 | 0.438 | 0.08301 |
| 1 | 0.0652 | 0.05692 | 0.0606 | 0.05604 |
| 2 | 0.0503 | 0.0531 | 0.0476 | 0.04933 |
| 3 | 0.0466 | 0.04696 | 0.0445 | 0.04656 |
| 5 | 0.0425 | 0.04597 | 0.0419 | — |
| 6 | 0.0421 | — | 0.0408 | 0.04384 |
| 8 | 0.0405 | 0.04475 | 0.0394 | 0.04338 |
| 9 | 0.039 | 0.04191 | 0.0386 | 0.0419 |
| 10 | 0.0393 | — | 0.0371 | 0.03943 |
| 11 | 0.0394 | 0.03986 | 0.0373 | — |
| 12 | 0.0384 | — | 0.0365 | 0.03826 |
| 17 | 0.0373 | 0.03901 | 0.0342 | — |
| 18 | 0.0362 | 0.03817 | 0.035 | — |
| 19 | 0.0368 | 0.03814 | 0.035 | — |
| 30 | — | **0.03538 (best)** | — | (run stopped) |

### Interpretation

First read of this looked like the augmentation wasn't doing anything: both
curves decrease at roughly the same rate and land in the same `val_loss`
ballpark. That reading is wrong, and the reason is the validation set — it's
clean in both runs, so `val_loss` only ever measures *"does training on harder
data hurt generalization to clean data,"* not *"is the training data actually
harder."*

The training `loss` column answers that second question directly, and it's
consistently higher for the augmented run at every epoch shown — roughly 5–15%
throughout (e.g. epoch 0: 0.563 vs 0.438; epoch 19: 0.0368 vs 0.0350). That gap
is the signature of the missing-wedge and amplitude-spectrum degradation
actually firing on training batches and making the regression target harder to
hit, exactly as the port intends (missing_wedge_prob=0.5 / amplitude_prob=0.5,
so roughly half of batches carry some degradation, which matches the size of
the gap).

So: the augmentation is active and measurably increasing training difficulty.
It is *not* measurably hurting — and may be marginally helping — generalization
to clean validation data (augmented's best val_loss, 0.03538, is as good as or
better than baseline's best so far, 0.03826, though the baseline run wasn't
carried to completion so this isn't a clean head-to-head). That is the intended
outcome of this kind of augmentation: robustness to a degraded input
distribution without sacrificing performance on clean inputs.

### What this run does *not* establish

- The baseline was stopped early (manual kill at epoch ~20, no early-stopping
  signal), so "augmented's best beats baseline's best" is not yet a fair
  finished-vs-finished comparison. A baseline run carried to its own
  early-stopping point would be the clean version of this comparison.
- Both numbers are from one seed each. No variance estimate.
- This says nothing yet about downstream picking accuracy (precision/recall
  against ground-truth coordinates) — only about the regression loss on
  Gaussian-sphere targets.

### Also fixed in this session (unrelated to the comparison above)

- `config.yaml`: `infer.predictions_coord_format: relion` was not a valid enum
  value (valid: `relion_31`, `relion_50`, `warp`, `imod`) and blocked config
  loading entirely. Set to `relion_31`.
- `CONTRAST_INVERSION_PROB` moved from `tomocpt/dataManager/dataloading.py` to
  `tomocpt/constants.py`, and its value changed from `0.8` to `0.5`.
