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
