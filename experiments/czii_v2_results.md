# CZII 10440 pipeline-v2 experiments

Appended by `experiments/eval_czii_v2.py`. F-scores use `tomocpt.metrics` (match radius 0.5 × particle radius).

## Baseline v2 multi-class, k = 0 (2026-10-28 09:05)

- run: `/home/pranav/Desktop/pranav/software/tomocpt/data/czii/weights/tomocpt_v2/tomocpt`; best epoch 17; cross_class_neg_weight = 0.0
- validation tomograms: TS_5_4, TS_69_2 (used for early stopping and calibration)
- calibrated thresholds: beta-amylase 0.45, beta-galactosidase 0.45, thyroglobulin 0.35, ribosome 0.35, apo-ferritin 0.5, virus-like-particle 0.6

| scalar | n | first | min | last |
|---|---|---|---|---|
| val_loss | 34 | 0.1346 | 0.00871 | 0.009474 |
| val_loss_apo-ferritin | 34 | 0.0786 | 0.00784 | 0.009054 |
| val_loss_beta-amylase | 34 | 0.08822 | 0.006239 | 0.007119 |
| val_loss_beta-galactosidase | 34 | 0.08669 | 0.006128 | 0.006936 |
| val_loss_ribosome | 34 | 0.1973 | 0.0121 | 0.01372 |
| val_loss_smooth | 34 | 2.566 | 0.009495 | 0.009607 |
| val_loss_thyroglobulin | 34 | 0.1273 | 0.01547 | 0.01731 |
| val_loss_virus-like-particle | 34 | 0.2295 | 0.002563 | 0.002709 |

| class | TP | FP | FN | P | R | F1 | F4 |
|---|---|---|---|---|---|---|---|
| beta-amylase | 12 | 275 | 8 | 0.04 | 0.60 | 0.08 | 0.34 |
| beta-galactosidase | 13 | 62 | 12 | 0.17 | 0.52 | 0.26 | 0.47 |
| thyroglobulin | 48 | 133 | 15 | 0.27 | 0.76 | 0.39 | 0.69 |
| ribosome | 49 | 22 | 19 | 0.69 | 0.72 | 0.71 | 0.72 |
| apo-ferritin | 51 | 29 | 30 | 0.64 | 0.63 | 0.63 | 0.63 |
| virus-like-particle | 17 | 0 | 3 | 1.00 | 0.85 | 0.92 | 0.86 |
| **mean** | | | | | | | **0.62** |

| true \ picked | amy | gal | thy | ribo | ferr | vlp | none |
|---|---|---|---|---|---|---|---|
| amy | 12 | 0 | 1 | 0 | 0 | 0 | 7 |
| gal | 2 | 13 | 5 | 0 | 0 | 0 | 5 |
| thy | 0 | 4 | 48 | 0 | 0 | 0 | 11 |
| ribo | 0 | 0 | 0 | 49 | 0 | 0 | 19 |
| ferr | 3 | 2 | 0 | 0 | 51 | 0 | 25 |
| vlp | 0 | 0 | 0 | 0 | 0 | 17 | 3 |

Cross-class suppression (all tomograms):

- Cross-class suppression, all tomograms: beta-amylase removed 40 beta-galactosidase peak(s)
- Cross-class suppression, all tomograms: beta-amylase removed 32 thyroglobulin peak(s)
- Cross-class suppression, all tomograms: beta-amylase removed 5 apo-ferritin peak(s)
- Cross-class suppression, all tomograms: beta-galactosidase removed 31 beta-amylase peak(s)
- Cross-class suppression, all tomograms: beta-galactosidase removed 63 thyroglobulin peak(s)
- Cross-class suppression, all tomograms: beta-galactosidase removed 1 apo-ferritin peak(s)
- Cross-class suppression, all tomograms: thyroglobulin removed 84 beta-galactosidase peak(s)
- Cross-class suppression, all tomograms: thyroglobulin removed 1 ribosome peak(s)
- Cross-class suppression, all tomograms: ribosome removed 1 thyroglobulin peak(s)
- Cross-class suppression, all tomograms: apo-ferritin removed 54 beta-amylase peak(s)
- Cross-class suppression, all tomograms: apo-ferritin removed 5 beta-galactosidase peak(s)
- Cross-class suppression, all tomograms: apo-ferritin removed 7 thyroglobulin peak(s)
- Cross-class suppression, all tomograms: virus-like-particle removed 13 ribosome peak(s)

## Synthetic calibration of cross_class_neg_weight (Gate 3b, 2026-10-28)

Gate 3 C13 scenario (four classes, solid vs hollow spheres of the same size, SNR 2, 1800 steps,
tiny SwinUNETR), test author's reference implementation. Cross-pick = fraction of one
same-size class picked in the other's channel.

| k | seed 0 | seed 1 | seed 2 | mean cross-pick |
|---|---|---|---|---|
| 0 | pass | fail (solid F1 0.57-0.59, cross 0.88-1.0) | pass | 0.167 |
| 0.25 | all F1 1.0 | all F1 1.0 | all F1 1.0 | 0.0 |
| 1.0 | min F1 0.97 | all F1 1.0 | all F1 1.0 | 0.0 |

Recommended k = 0.25.

## Cross-class negative weighting, k = 0.25 (2026-10-28 13:53)

- run: `/home/pranav/Desktop/pranav/software/tomocpt/data/czii/weights/cross_class`; best epoch 34; cross_class_neg_weight = 0.25
- validation tomograms: TS_5_4, TS_69_2 (used for early stopping and calibration)
- calibrated thresholds: beta-amylase 0.3, beta-galactosidase 0.05, thyroglobulin 0.25, ribosome 0.1, apo-ferritin 0.3, virus-like-particle 0.5

| scalar | n | first | min | last |
|---|---|---|---|---|
| val_loss | 35 | 0.2298 | 0.01395 | 0.01395 |
| val_loss_apo-ferritin | 35 | 0.1318 | 0.008236 | 0.008627 |
| val_loss_beta-amylase | 35 | 0.08313 | 0.007971 | 0.007988 |
| val_loss_beta-galactosidase | 35 | 0.1191 | 0.01163 | 0.01258 |
| val_loss_ribosome | 35 | 0.1945 | 0.01637 | 0.01637 |
| val_loss_smooth | 35 | 13.18 | 0.01516 | 0.0153 |
| val_loss_thyroglobulin | 35 | 0.3091 | 0.02117 | 0.02117 |
| val_loss_virus-like-particle | 35 | 0.08954 | 0.003483 | 0.003494 |

| class | TP | FP | FN | P | R | F1 | F4 |
|---|---|---|---|---|---|---|---|
| beta-amylase | 10 | 142 | 10 | 0.07 | 0.50 | 0.12 | 0.36 |
| beta-galactosidase | 9 | 157 | 16 | 0.05 | 0.36 | 0.09 | 0.27 |
| thyroglobulin | 48 | 126 | 15 | 0.28 | 0.76 | 0.41 | 0.69 |
| ribosome | 48 | 46 | 20 | 0.51 | 0.71 | 0.59 | 0.69 |
| apo-ferritin | 61 | 46 | 20 | 0.57 | 0.75 | 0.65 | 0.74 |
| virus-like-particle | 17 | 0 | 3 | 1.00 | 0.85 | 0.92 | 0.86 |
| **mean** | | | | | | | **0.60** |

| true \ picked | amy | gal | thy | ribo | ferr | vlp | none |
|---|---|---|---|---|---|---|---|
| amy | 10 | 1 | 0 | 0 | 0 | 0 | 9 |
| gal | 0 | 9 | 1 | 0 | 0 | 0 | 15 |
| thy | 0 | 0 | 48 | 0 | 0 | 0 | 15 |
| ribo | 0 | 0 | 0 | 48 | 0 | 0 | 20 |
| ferr | 0 | 0 | 0 | 0 | 61 | 0 | 20 |
| vlp | 0 | 0 | 0 | 0 | 0 | 17 | 3 |

Cross-class suppression (all tomograms):

- none logged (no peak of one class fell next to a peak of another)

## Comparison: k = 0.25 vs k = 0 (2026-10-28)

| class | n true | F4 k=0 | F4 k=0.25 | FP k=0 | FP k=0.25 | threshold k=0 | threshold k=0.25 |
|---|---|---|---|---|---|---|---|
| beta-amylase | 20 | 0.34 | 0.36 | 275 | 142 | 0.45 | 0.30 |
| beta-galactosidase | 25 | 0.47 | 0.27 | 62 | 157 | 0.45 | 0.05 |
| thyroglobulin | 63 | 0.69 | 0.69 | 133 | 126 | 0.35 | 0.25 |
| ribosome | 68 | 0.72 | 0.69 | 22 | 46 | 0.35 | 0.10 |
| apo-ferritin | 81 | 0.63 | 0.74 | 29 | 46 | 0.50 | 0.30 |
| virus-like-particle | 20 | 0.86 | 0.86 | 0 | 0 | 0.60 | 0.50 |
| **mean** | | **0.62** | **0.60** | 521 | 517 | | |

- Cross-class suppression removed 337 peaks at k = 0 and none at k = 0.25: the loss term
  stops two classes firing on one particle, as it did on synthetic data.
- Overall F4 is unchanged within noise (two validation tomograms, 20-81 particles per class).
  Background false picks, the main error at k = 0, are not reduced in total.
- Calibrated thresholds fall for every class (beta-galactosidase 0.05, ribosome 0.10): peaks are
  lower. beta-galactosidase, which overlapped most with thyroglobulin, loses most.
- k = 0.25 stopped after 35 epochs (best = last epoch, 34); k = 0 had its best at epoch 17 of 34.

## Threshold-free diagnostics, k = 0 vs k = 0.25 (2026-10-28)

`experiments/diag_czii_v2.py`, from the saved per-class maps (peaks at threshold 0.01, cross-class suppression on). AP = average precision; best F4 over all thresholds. False picks (FP) at the calibrated threshold are classed as: within one radius of another class's true particle, within 2 radii of a found particle of the same class (duplicate), within 2 radii of the border, else background. Contrast = mean of the central cube minus the surrounding shell, in normalised units (median over picks; 200 random points per class and tomogram as reference).

| run | class | AP | best F4 | at threshold | calibrated | recall at 0.01 | FP on other class | FP duplicate | FP border | FP background | contrast TP / FP / random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k=0 | beta-amylase | 0.13 | 0.37 | 0.47 | 0.45 | 0.70 | 6 | 3 | 0 | 284 | -0.23 / -0.22 / -0.00 |
| k=0 | beta-galactosidase | 0.13 | 0.47 | 0.47 | 0.45 | 0.52 | 6 | 0 | 1 | 57 | -0.22 / -0.20 / -0.00 |
| k=0 | thyroglobulin | 0.43 | 0.68 | 0.30 | 0.35 | 0.78 | 8 | 1 | 0 | 114 | -0.11 / -0.11 / +0.00 |
| k=0 | ribosome | 0.67 | 0.72 | 0.37 | 0.35 | 0.72 | 0 | 3 | 0 | 18 | -0.18 / -0.13 / +0.00 |
| k=0 | apo-ferritin | 0.57 | 0.67 | 0.51 | 0.50 | 0.70 | 0 | 2 | 1 | 36 | +0.10 / +0.04 / +0.01 |
| k=0 | virus-like-particle | 0.85 | 0.86 | 0.62 | 0.60 | 0.85 | 0 | 0 | 0 | 0 | -0.04 / n/a / +0.00 |
| k=0.25 | beta-amylase | 0.07 | 0.38 | 0.30 | 0.30 | 0.65 | 0 | 1 | 1 | 164 | -0.21 / -0.20 / -0.00 |
| k=0.25 | beta-galactosidase | 0.08 | 0.32 | 0.06 | 0.05 | 0.44 | 3 | 5 | 0 | 128 | -0.18 / -0.07 / -0.00 |
| k=0.25 | thyroglobulin | 0.45 | 0.70 | 0.27 | 0.25 | 0.83 | 5 | 1 | 0 | 114 | -0.12 / -0.10 / +0.00 |
| k=0.25 | ribosome | 0.66 | 0.70 | 0.13 | 0.10 | 0.72 | 1 | 3 | 1 | 41 | -0.18 / -0.13 / +0.00 |
| k=0.25 | apo-ferritin | 0.65 | 0.74 | 0.33 | 0.30 | 0.75 | 0 | 5 | 1 | 47 | +0.10 / +0.07 / +0.01 |
| k=0.25 | virus-like-particle | 0.85 | 0.86 | 0.55 | 0.50 | 0.85 | 0 | 0 | 0 | 0 | -0.05 / n/a / +0.00 |
| k=0 | **mean** | **0.46** | **0.63** | | | | 20 | 9 | 2 | 509 | |
| k=0.25 | **mean** | **0.46** | **0.61** | | | | 9 | 15 | 3 | 494 | |

- Ranking quality is unchanged on average (mean AP 0.46 for both). k = 0.25 lowers AP for beta-amylase (0.13 to 0.07) and beta-galactosidase (0.13 to 0.08) and raises it for apo-ferritin (0.57 to 0.65).
- 494 to 509 of about 540 false picks are in background, not on another class.
- False picks have the same local contrast as true picks (for example beta-amylase -0.22 vs -0.23), and random points have about 0: most false picks sit on real densities.
- Montage `experiments/figures/fp_montage_k0.png` (k = 0): the top beta-galactosidase false picks are compact, very dark round objects with a bright halo, unlike the elongated true picks; the thyroglobulin and beta-amylase false picks look like the true picks.

### Conclusions (2026-10-28)

- Lower peaks at k = 0.25 are mostly a calibration effect (mean AP unchanged), but the penalty
  makes the network hedge on the small look-alike classes (beta-amylase, beta-galactosidase AP
  about halved). Use k = 0 (at most 0.1) on CZII.
- About 95 % of false picks are in background, on real densities with particle-like contrast,
  not on another class. Cross-class confusion is minor and inference suppression handles it, so
  loss reweighting (k, or adaptive boundary / uncertainty weighting) cannot remove them.
- The top beta-galactosidase false picks look like gold fiducials or dense cores (unverified).
- Recall at threshold 0.01 is only 0.52 (beta-galactosidase) and 0.70 (beta-amylase): a
  sensitivity limit separate from the false picks.
- Levers: check and mask fiducials, more labelled tomograms, SSL pretraining (Gate 5), and a
  softmax class head (planned) to make classes mutually exclusive without tuning k.
