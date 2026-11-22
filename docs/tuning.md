# Tuning TomoCPT: what to change, when, and why

This guide is organised by task. For each option: what it does, the default, when to change it and
in which direction, what it costs, and the evidence from this project when there is some. Every
option is listed with its type and default in [options.md](options.md); complete configs for common
situations are in [examples/](examples/).

Options are written as `section.path`, e.g. `train.particles.class_head`. In a YAML file this is

```yaml
train:
  particles:
    class_head: softmax
```

and on the command line `tomocpt train ... particles.class_head=softmax` (relative to the
command's section). `tomocpt init --level advanced` writes every option with its help text.

**About the numbers.** Evidence comes from runs made while developing v2: CZII (dataset 10440, 7
tomograms, 6 species, 10.012 Å/px; 5-fold cross-validation, out-of-fold F4), CryoET Data Portal
DS-10490 (simulated, 132 particle species plus microtubules and actin at 6.8 Å/px) and synthetic
phantoms. They are single runs unless stated: differences of 0.01 in F4 are within run-to-run noise
(GPU training is not deterministic). Where no measurement exists, this guide says so.

---

## 1. Data and scale

### Pipeline v1 or v2 (`train.data_pipeline_version`, default 1)

* **v1 (legacy)**: one species per dataset. Each tomogram is rescaled so the particle diameter is
  `prepData.desired_particle_pixel_size` (20) voxels, trilinear without anti-aliasing. Particles
  touching the border become negatives. Use it to reproduce older models or to fine-tune the public
  v1 model on one species.
* **v2**: several species (and filaments) in one model and one `predict` run, from tomograms of
  different voxel sizes. Tomograms are resampled only as far as the size band needs, with Fourier
  cropping (`train.scale.resample`, `fourier`, anti-aliased). All species of a tomogram are labelled
  together, so one species is not a negative for another. The per-class thresholds are calibrated
  after training. v2 is the setting for anything new; the rest of this guide is about v2 unless it
  says otherwise.

v2 needs `prepData.multiclass: true` with `prepData.class_diameters` and `prepData.class_names`
(one STAR file with `rlnClassNumber`), and `train.data_pipeline_version: 2`. Chunk files are
versioned: a v2 run refuses v1 chunks and the reverse.

### The size band and levels (`train.scale.band_min_vox` 10, `train.scale.band_max_vox` 32)

The working voxel size is chosen so that every class diameter is between `band_min_vox` and
`band_max_vox` voxels, never finer than native. When the classes span more than the band ratio
(32 / 10 = 3.2), they are split into *levels*, one working voxel size each; each class is read at
its own level in `predict`. A class smaller than `band_min_vox` at native size is used at native
size.

* **Raise `band_max_vox`** when one large class would otherwise force an extra level at a coarse
  voxel size. Evidence (DS-10490 filaments): with `band_max_vox` 37 a 250 Å microtubule fits the
  same level as actin at 6.8 Å/px; together with 20 instead of 5 training tomograms, actin
  centreline F1 went from 0.617 to 0.782 (microtubule ~1.0 either way). The two changes were made
  together, so the share of each is not known.
* **Cost**: a larger band means larger objects in the 64³ crop (less context per object) or more
  levels (more prediction passes). A wider band was not measured beyond this case.
* `train.scale.zoom_aug_max` (1.0 = off) zooms crops by up to this factor during training, so a
  class is seen at several voxel diameters. Not benchmarked. Use it with
  `train.network.scale_conditioning`.
* `train.scale.scales_per_level` (1) adds jittered working voxel sizes per level. Not benchmarked.

### Scale conditioning (`train.network.scale_conditioning`, default false)

FiLM layers tell the decoder the working voxel size, so the network can separate a 30-voxel object
at 10 Å/px from one at 20 Å/px. Turn it on for training data with mixed voxel sizes, classes on
several levels, or zoom augmentation. It is pointless for one dataset at one voxel size that fits
one level (CZII). Evidence: synthetic two-level phantoms (Gate 3 test, one seed) - the conditioned
model fired less on a class at the other level (mean score 0.125 vs 0.235). It is fixed at training
start (`resume_from` refuses a change; a fine-tune into a conditioned model works, the FiLM layers
start at zero).

### Voxel size (`prepData.apix_override`, `infer.apix_override`, default none)

The voxel size comes from the MRC header. Set the override when the header is wrong or 0 (e.g.
tomograms written by tools that do not set it). A wrong voxel size puts every class at the wrong
voxel diameter, in training and in `predict`.

### Multi-class labelling (`prepData.class_diameters`, `prepData.class_names`, `prepData.class_annotated`)

* `class_diameters`: Å per class number, `"1:130,2:180"`. Use published diameters where they exist;
  they set the target size, the band, the peak spacing and the matching tolerance of the
  calibration.
* `class_annotated`: the classes that are annotated **exhaustively** in this dataset (default: every
  class present in its coordinates). A class not annotated in a sample is left out of the loss for
  that sample, so unlabelled particles of that class are not trained as background. Set it when a
  dataset labels only some species, or when you combine datasets with different label sets.

### Border particles (`prepData.label_border_policy`, default skip)

`skip` (legacy) drops particles whose box touches the tomogram border, so they become negatives;
`clip` draws the part inside the tomogram. Use `clip` for v2 (all example configs and benchmark
runs do). No A/B measurement of the two.

### Exact coordinates (pipeline v2 default, no option)

v2 keeps sub-voxel label coordinates (v1 truncates them to integers, a bias of -0.5 voxel per axis).
There is nothing to set, but data prepared before this change must be prepared again. Rescoring the
CZII CV picks against the exact coordinates changed the baseline F4 little (0.741 both ways).
On CZII, picks sit about (-0.4, -0.45, -1.4) native voxels (x, y, z) from the labels on every
tomogram; synthetic blobs are recovered within 0.025 voxel, so the offset is in the labels. Random
flips then blur the target by the offset. No general fix is in the package yet.

### Membranes and other never-a-class structures (`prepData.negative_masks`)

Masks (MRC, matched to tomograms by name) of structures that are never a class, e.g. membranes:
`"membrane:/path/to/masks/membrane"`. They are used as negatives of the contrastive loss, so they
need `train.discriminative.contrastive_weight > 0`. `train.otf.negative_mask_weight` (0) also
oversamples crops containing them (not benchmarked). Evidence (DS-10490 actin, 20 training
tomograms): membrane negatives with contrastive weight 0.1 raised actin F1 from 0.782 to 0.810, and
false actin traces near membranes fell from 19-27 % to 0 %.

---

## 2. Class head

### `train.particles.class_head`: `sigmoid` (default) or `softmax`

* `sigmoid`: one independent probability per class. A particle can be picked as two classes.
* `softmax`: the classes and an implicit background compete for each voxel. Use it for every
  multi-class model. Evidence: on synthetic phantoms of similar-sized spheres, the sigmoid head
  confused the sizes and failed the Gate 6 joint test; softmax gave particle F1 1.00 (seeds 0 and
  1). Adding cross-entropy terms that include the background made peaks flat-topped and worse (F1
  0.98 and 0.61 in two variants), so the head has none.
* `train.particles.cross_class_neg_weight` (0): for sigmoid models, extra penalty for firing on
  another class's particles. Not needed with softmax (k = 0 and 0.25 both gave F1 1.00 there).

### `train.particles.softmax_scope`: `all` (default) or `particles`

`particles` puts only the particle channels in the softmax; filament channels become independent
sigmoids (filaments can overlap particles). Use `particles` for models with filaments. Evidence
(DS-10490, C = 134): particle F4 0.280 (`particles`) vs 0.277 (`all`), cross-calibrated (threshold
calibrated on one validation tomogram, scored on the other) - a tie; filaments trained faster, and
80S ribosome (0.60 -> 0.68) and GroEL (0.25 -> 0.33) were better. Older checkpoints read as `all`.

---

## 3. Many classes or rare classes

### Repeat-factor sampling (`train.chunk_sampling`, `train.repeat_factor_threshold`, `train.repeat_factor_max`)

`train.chunk_sampling: repeat_factor` (v2 only; default `uniform`) oversamples crops containing
rare classes: each class gets `r_c = max(1, sqrt(t / f_c))` with `f_c` the fraction of crops that
contain it and `t = repeat_factor_threshold` (0.1); a crop is repeated by the largest `r_c` of its
classes. Raise `t` to oversample more classes more strongly; cap it with `repeat_factor_max` (none)
when a handful of very rare classes would dominate the epoch. The training log (train.log) lists
the largest factors at data setup - check them before a long run. The DS-10490 run uses `t` 1.0,
cap 10 (a choice, not a measured optimum). Rare classes with very few examples were still not
learned on DS-10490 (own-class probability ~0.01): sampling cannot replace labels.

### Batch size and memory (`train.batch_size` 16)

Swin gradient checkpointing (`train.network.SwinUNETR.use_checkpoint`, true) is switched off
automatically when memory allows, and the Hessian loss term is checkpointed automatically when
its activations would not fit (`train.particles.loss_checkpoint` forces it on). Evidence: C = 134,
batch 16 at 64³ peaks at ~34-37 GB with Hessian checkpointing, ~3.3-3.9 s per step on one 48 GB GPU.
With few classes (CZII) a step takes ~0.8 s. If you run out of memory, lower `train.batch_size`.
When you change the batch size, see `train.lr_scaling` (section 5).

### Small proteins

On DS-10490 at 6.8 Å/px, small proteins (49-81 Å, 7-12 voxels) are confused with each other and
picked inside large complexes. No option fixed this: cross-class suppression and an argmax gate both
lowered F4 (section 8). It is likely a signal limit at that voxel size.

---

## 4. Contrastive loss and embedding head (`train.discriminative`)

* `train.discriminative.contrastive_weight` (0 = off): a supervised dense contrastive loss on an
  embedding head, with hard negatives (`hard_negatives`, between `hard_negative_min_score` and
  `hard_negative_max_score`) and mask negatives (`prepData.negative_masks`). Use **0.1**. Evidence:
  CZII CV F4 0.742 vs 0.741 (a tie overall; thyroglobulin false positives 937 -> 840); DS-10490
  actin with membrane negatives 0.782 -> 0.810; at 0.5 the microtubule loss got worse and particle
  F4 fell slightly (0.272 vs 0.278 control).
* `train.discriminative.contrastive_warmup_steps` (0): the weight ramps from 0 over N steps while the
  memory bank fills. The DS-10490 run uses 1024 (not compared with 0).
* `train.discriminative.bank_max_age` (0 = no limit): memory-bank entries older than N steps are
  ignored, so rare classes do not compare against embeddings from many epochs ago. DS-10490 run:
  5000 (not compared).
* `train.discriminative.embed_dim` (64): embedding size; the DS-10490 run uses 128 for 134 classes
  (not compared). With 64 dimensions the DS-10490 embeddings were not collapsed (effective rank ~18).
* `train.discriminative.embed_ce_weight` (0) and `train.discriminative.prototype_head` (false):
  make the class head use the embedding. Evidence (DS-10490 fine-tunes, 6144 steps,
  cross-calibrated F4): control 0.278, embedding CE 0.279, CE + prototype head 0.279 - a tie. They
  are opt-in; the boss run uses both (a human choice, not a measured gain).
* `train.discriminative.value_loss` (`huber`): `focal` is a soft focal value term. Evidence: CZII CV
  F4 0.730 vs 0.741 (more thyroglobulin false positives, 937 -> 1282); focal + contrastive 0.737.
  Keep `huber`.

---

## 5. Optimisation and schedule

* `train.optimizer.lr` (4e-4, RAdam). Fine-tunes scale it per parameter group (section 7). No sweep
  was run.
* `train.lr_warmup_steps` (0): linear warmup over N optimizer steps, every group. Use it for large
  class tables or large batches; the DS-10490 run uses 1024 (not compared). Not with
  `train.schedule_free`.
* **Early stopping** on `val_loss_smooth`: legacy `train.early_stop_min_delta` 0.005 (absolute) and
  `train.early_stop_patience` 10. An absolute threshold depends on the loss scale: on CZII it was
  30-50 % of the final loss, so runs stopped at epochs 24-29 while still improving. Set
  `train.early_stop_rel_delta: 0.007` (relative; 0.007 = 1e-4 / 0.0147, the CZII loss floor) with
  `train.lr_plateau_match_early_stop: true` and `train.lr_plateau_patience: 5` (must be smaller than
  `early_stop_patience`): the learning rate halves before training stops (replayed on the CZII
  curve: LR halves at epochs 41 and 48, stop at 52, the same epochs with the loss scaled by 100 or
  0.01). This is the driver recipe of the filament and DS-10490 runs.
* **Weight averaging** (`train.weight_averaging`, `swa`): legacy Lightning SWA starts at 80 % of
  `n_epochs` and its average is applied only at the end, so under early stopping it never reaches a
  checkpoint. Use `ema` (EMA of the weights from step 0, `train.ema_decay` 0.999; validated and
  saved) or `swa_plateau` (SWA started when the loss plateaus). `ema` is the driver recipe.
* `train.schedule_free` (false): RAdamScheduleFree (`pip install 'tomocpt[schedulefree]'`); no LR
  scheduler, no SWA/EMA. Evidence: one CZII fold-0 pilot, weighted F4 0.771 (schedule-free) vs 0.745
  (RAdam + EMA); on that fold one class alone varies by +-0.05, so this cannot rank the two. Not
  run at scale with several GPUs.
* `train.lr_scaling` (`none`) with `train.lr_reference_batch` (16): scale the learning rate (and the
  EMA decay) with the effective batch (`batch_size` x GPUs): `sqrt` for Adam-family optimizers,
  `linear` for SGD. Use `sqrt` when you change the batch size or train on several GPUs. Not
  validated on GPU runs yet.
* `train.n_epochs` (10) is a maximum; with early stopping, set it high (the DS-10490 run: 100).

---

## 6. Augmentation (`train.augmentation`)

* **Fourier degradation**: `train.augmentation.use_mw_aug` (missing wedge) and
  `train.augmentation.use_fourier_aug` (random amplitude spectrum: varied CTF, defocus, dose
  filtering), each at probability 0.5 (`missing_wedge_prob`, `amplitude_prob`). With
  `train.augmentation.fourier_auto` (true) both are **on for pipeline-v2 picking**; v1 and
  self-supervised training are unchanged. Why: the goal is a picker that works on other people's
  data, and these imitate the differences between datasets and microscopes. They were validated
  functionally; there is no in-dataset A/B number (a single-dataset benchmark does not measure
  generalisation). Set `fourier_auto=false` to keep the two switches as written.
* **Rotation** (`train.augmentation.rotation_mode`): `affine` (legacy: random rotation up to 45°
  plus elastic/blur) rotates the real missing wedge of the tomogram away from Z; `flips` (axis
  flips with `train.augmentation.flip_p`, use 0.5) keeps it in place. The filament and DS-10490
  recipes use `flips`. Caveat: flips also reverse any label offset, blurring the target by it (see
  exact coordinates, section 1).
* `train.augmentation.label_interpolation`: `linear` for v2 (smooth soft targets under rotation;
  the example configs set it), `nearest` is legacy.
* **Intensity options** (`brightness_gradient_p`, `local_gamma_p`, `brightness_p`, `contrast_p`,
  `anisotropy_p`, `noise_p`, `gamma_p`, `bias_field_p`): off by default, not benchmarked. Contrast
  inversion (p 0.5) is always on.

---

## 7. Speed and memory

* `train.sampling` (`auto`): v2 picking draws crops on the fly from one array per tomogram
  (`on_the_fly`) unless `chunks_dir` already holds chunk files; v1 and SSL use chunk files.
  Evidence (CZII 5-fold CV): 29.9 h vs 38.4 h wall time, OOF F4 0.763 vs 0.748; store 1 GB vs 8 GB
  of chunks per fold.
* `train.otf.augment_device` (`auto`): augmentation on the GPU when training on a GPU (every
  augmentation has a GPU port). Leave it.
* `train.otf.samples_per_epoch` (none = the number of grid chunks): crops per epoch. Set it to make
  epochs (and validation, early stopping) regular in wall time on large datasets; DS-10490: 16384
  crops = 1024 steps at batch 16, ~1.1-1.2 h.
* `train.n_cpus_for_dataloading` (8 per GPU) and `train.n_cpus_for_preprocessing` (8): raise them
  if the GPU waits for data (low GPU use in `nvidia-smi`). With on-the-fly sampling and GPU
  augmentation the measured wait was 2-7 % of a step at 8 workers.
* `infer.accumulate_device` (`auto`) and `infer.accumulate_dtype` (`float32`): where the per-class
  maps of `predict` are summed. `auto` uses the GPU when the maps fit; `cpu` and/or `float16` save
  GPU memory for large tomograms with many classes. Evidence: accumulating on the GPU made CZII
  prediction 2.3x faster; predict went from 68 to 27 s per CZII tomogram in Gate 8. At C = 134,
  predict needed ~17 GB after the fixes.
* `infer.predictions_batch_size` (2): patches per forward pass in `predict`; raise it on a large GPU.

---

## 8. Fine-tuning (`train.fine_tune_from` vs `train.resume_from`)

* `train.resume_from`: continue an interrupted run (weights, optimizer, scheduler, epoch). The
  architecture (class table, scale conditioning) cannot change.
* `train.fine_tune_from`: a new run starting from the weights of a checkpoint (picker, v1 picker or
  self-supervised). The optimizer starts fresh with differential learning rates: Swin encoder x0.01,
  `encoder*` x0.1, decoder and output x1.0, so the backbone changes slowly.
* **What carries over.** From a picker, the run uses the checkpoint's config with the values **you
  changed from the defaults** applied; settings you leave at their defaults keep the checkpoint's
  values. The width (`train.network.SwinUNETR.feature_size`) must match. A v2 class table may
  differ: classes with the same class number keep their output channel, new classes start from the
  mean of the old channels. A one-channel (v1 or SSL) source starts every class from its output. A
  contrastive picker keeps its projection head (and its memory bank and prototypes when the class
  table is the same).
* With few labels: keep the class head of the source model, freeze nothing (the differential rates
  do the work), lower `train.n_epochs` or rely on early stopping, and use a fixed validation split
  (`train.val_stems`) so the stopping signal is stable. No few-label benchmark has been run with v2.
* `train.use_distillation` (false) keeps a frozen copy of the source as a teacher
  (`train.distill_weight` 0.5); for adapting without forgetting. Not benchmarked in v2.

---

## 9. Inference and thresholds

* **Calibration.** After a v2 training run, each class gets the threshold that maximises F-beta
  on the validation tomograms (`infer.calibration_beta` 4: recall weighs 4x precision, as in the
  CZII challenge; lower it if you prefer precision). Classes with fewer than
  `infer.calibration_min_particles` (20) true particles in validation stay uncalibrated and use
  `infer.confidence_threshold` (0.3). Thresholds calibrated on the same tomograms they are scored
  on are optimistic: DS-10490 F4 0.299 same-tomogram vs 0.277 calibrated on another tomogram.
  Calibrate on tomograms not used for training (the validation split does this), and judge on
  others.
* `infer.class_thresholds` (`"ribosome:0.4,apo-ferritin:0.35"`): overrides the calibration per
  class; the way to trade recall for precision on one class without retraining.
* `infer.peak_spacing_fraction` (0.5): minimum distance between two picks of one class, as a
  fraction of the class diameter. `infer.distance_threshold` (Å) overrides it.
* `infer.cross_class_nms` (false in v2): suppresses peaks of different classes closer than half the
  smaller radius. Off because suppression costs more recall than it gains precision. Evidence: CZII
  OOF F4 0.631 -> 0.670 with suppression off, 0.737 with the smaller peak spacing too, 0.741 after
  recalibration (the v2 defaults); DS-10490 cross-calibrated F4 0.277 (per-class peaks) vs 0.177 with
  cross-class NMS and 0.245 with an argmax gate. Turn it on only for sigmoid-head models that fire
  on the same particle in several channels.
* `infer.use_tta` (false): flip test-time augmentation; slower and more GPU memory. Not benchmarked.
* `infer.subvoxel_refine` (`fit`): picks are refined by a log-quadratic fit; `offset` uses the
  offset head (`train.particles.offset_weight` > 0) and needs inference (not `skip_inference`). Not
  benchmarked; likewise `train.particles.core_weight` (centroid weight).
* **Re-picking** without inference: run once with `infer.save_prediction_confidence_map: true`,
  then `predict` with `infer.skip_inference: true` (and `infer.weights` for the class table) and new
  thresholds or spacing.
* **Filaments**: `infer.filament_spacing_ang` (40 Å between output points),
  `infer.filament_min_length_ang` (200 Å, shorter traces dropped - raise it to remove short false
  fragments; on DS-10490 a length filter gave at most +0.03 F1 on one tomogram and lowered the other),
  `infer.filament_smoothing_ang` (10 Å). The tracing threshold is calibrated like the particle
  thresholds (`infer.filament_calibration_beta` 1).

---

## 10. Cross-validation and ensembles

* `train.cv_folds` (0 = off) with `train.cv_all_folds: true` trains one model per fold of
  tomograms in one command, calibrates thresholds on the pooled out-of-fold predictions, builds a
  fold ensemble (`infer.weights` = the `ensemble.json`) and writes a CV report.
* **When**: few tomograms (CZII: 7). Each tomogram then contributes to calibration once, out of
  fold, and the ensemble uses all data; the cost is k trainings (CZII 5 folds: ~30 h on one GPU).
  The CV report is also the most reliable comparison of settings on small data: a single fold of 2
  tomograms could not rank two recipes (section 5).
* **When a fixed split is enough**: many tomograms (DS-10490: 239 training, 20 validation runs).
  Use `train.val_stems` to fix the validation tomograms, keep them out of training, and keep a
  separate test set for the final number.
* `train.cv_class_weights` weights classes in the report's mean F-beta only (e.g. the CZII challenge
  weights); it does not change training.
