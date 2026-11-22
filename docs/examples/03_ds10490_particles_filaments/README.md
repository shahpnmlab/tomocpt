# Example 3: many species, microtubules and actin, membrane negatives

**Situation.** A large class table: 132 particle species from 25 to 388 A, plus microtubules and
actin filaments, at 6.8 A/px, with membrane masks. This is the DS-10490 "final boss" run of the
CryoET Data Portal (simulated tomograms), written as one config file: the settings of
`experiments/run_boss_ds10490.sh` and of the recipe in `benchmarks/filaments.py` (a test checks
that they agree).

**Commands.**
```bash
python scripts/masks_to_centerlines.py ...        # once: portal filament masks -> IMOD centrelines
tomocpt prepare_data --config_file config.yaml    # particles, filament centrelines, membrane masks
tomocpt train --config_file config.yaml           # ~4.5-5 days on one 48 GB GPU at 100 epochs
tomocpt predict --config_file config.yaml         # one file per class; filaments as traced polylines
```

**Settings** (details and evidence: docs/tuning.md).
- `prepData.filament_coords`: filament classes from centreline models (class numbers 232, 233,
  declared in `class_diameters` / `class_names` like particles). `filament_masks` reads masks
  directly, but touching filaments merge there.
- `prepData.negative_masks` with `train.discriminative.contrastive_weight: 0.1`: membranes are
  negatives of the contrastive loss (actin F1 0.782 -> 0.810, false actin near membranes to 0 %).
- `train.scale.band_max_vox: 37`: the 250 A microtubule shares a level with actin.
- `train.network.scale_conditioning: true`: the classes are on two levels (6.8 and 10.5 A/px).
- `train.particles.class_head: softmax` with `softmax_scope: particles`: particles compete,
  filaments are independent sigmoids (filaments train faster; particle F4 a tie).
- `train.chunk_sampling: repeat_factor`, `repeat_factor_threshold: 1.0`, `repeat_factor_max: 10`:
  rare species are oversampled, at most 10x. Check the factors at the top of the training log.
- `train.otf.samples_per_epoch: 16384`: epochs of fixed length, so early stopping and the LR
  plateau act on a regular clock.
- Warmups (`lr_warmup_steps`, `discriminative.contrastive_warmup_steps` 1024), `bank_max_age`
  5000, `embed_dim` 128, `embed_ce_weight` 0.1, `prototype_head`: the Gate 9 / 9b options for large
  class tables. Their values are choices, not measured optima (the head options were a tie).
- `train.augmentation.rotation_mode: flips`: keeps the real missing wedge in place.
- `train.val_stems`: a fixed validation split (20 runs in the real run); keep a separate test set.
- `infer.calibration_min_particles: 5`: many species have few particles per validation tomogram.

**After the run.** Calibrate and score on different tomograms: thresholds calibrated on the
tomograms they are scored on overstate F4 (0.299 vs 0.277 on DS-10490). Expect small proteins (7-12
voxels) to be confused with each other and with large complexes, and very rare classes not to be
learned (docs/tuning.md, section 3). For filaments, look at the traced polylines over the
tomogram: short isolated traces are the usual false positives (`infer.filament_min_length_ang`).
