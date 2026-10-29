<img src="assets/logo.png">

### tomoCPT (tomogram Centroid Prediction Tool) is a deep learning based program for enabling centroid prediction of objects in 3D cryo-tomograms.

# Installation
1. Clone the repository in a user writable location
```
git clone https://github.com/shahpnmlab/tomocpt
```
2. ```cd tomocpt```

3. Create a virtual environment to install tomocpt into
```
conda create -n tomocpt python=3.10
conda activate tomocpt
pip install -e .
```
4. Check if things are working by running
```
tomocpt --help
```
You should see the following output
```
  Usage: tomocpt [OPTIONS] COMMAND [ARGS]...                                                                                                                                                
                                                                                                                                                                                           
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                             
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ init           Function to create a template config file for running tomoCPT, only including annotated fields                                                                           
│ prepare_data   Prepares particle picking datasets by processing multiple tomograms and their corresponding coordinate files.                                                            
│ train          Trains a deep learning model for particle picking or self-supervised learning using PyTorch Lightning.                                                                   
│ predict        Performs parallel inference on tomogram data for particle detection and coordinate extraction.                                                                           
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

# Usage
### Outline of tomocpt workflow

# Usage instructions

TomoCPT (Tomogram Centroid Prediction Tool) is a deep learning-based solution for detecting and localizing particles in 3D cryo-electron tomograms. This guide walks you through the complete workflow from setup to prediction.

## 1. Initialize a Configuration File

The first step is to create a configuration file that defines parameters for data preparation, training, and inference.

### Command-line approach:

```bash
# Create a default configuration file in the current directory
tomocpt init --output_path ./my_config.yaml
```

### Expected output:

A `my_config.yaml` file will be created. Edit this file to set the paths and parameters according to your needs before proceeding. If working with multiple datasets, use comma separated values.

A sample with updated training parameters:
```yaml
# my_config.yaml (sample content)
prepData:
  raw_data_dir: ???
  training_data_dir: ???
  particle_length_ang: ???
  coordinate_files: ???
  # ... other prepData fields
train:
  chunks_dir: ???      # Directory with preprocessed data chunks (REQUIRED)
  model_dir: ???       # Directory to save model weights and logs (REQUIRED)
  resume_from: null    # Optional: Path to a checkpoint to RESUME a run
  fine_tune_from: null # Optional: Path to a checkpoint to FINE-TUNE from
  n_epochs: 10
  batch_size: 16
  # ... other training fields
infer:
  tomogram_dir: ???
  predictions_dir: ???
  weights: ???
  # ... other inference fields
```

## 2. Generate Volume-Label Pairs

This step processes your raw tomograms and coordinate files into a format suitable for training.

### Config file approach:

```bash
# After filling out the prepData section in my_config.yaml
tomocpt prepare_data --config_file my_config.yaml
```

### Combined approach:

```bash
# Use config file but override specific parameters
tomocpt prepare_data --config_file my_config.yaml --particle_length_ang 200
```

## 3. Train a Model

This section covers the main training workflows. All parameters can be set either via the command line or in your `.yaml` configuration file.

### Scenario 1: Training from Scratch

This is the most basic scenario, where you train a new model from randomly initialized weights.

```bash
# Using a configuration file is recommended
tomocpt train --config_file my_config.yaml
```

### Scenario 2: Resuming an Interrupted Run

If your training run is aborted, you can resume it seamlessly from the last saved checkpoint. This restores the model, optimizer, and learning rate scheduler to their exact previous states.

```bash
# The command is the same as your original run, just add --resume_from
tomocpt train --config_file my_config.yaml --resume_from /path/to/model_dir/weights.ckpt
```
*Alternatively, set `resume_from: /path/to/model_dir/weights.ckpt` in your `my_config.yaml`.*

### Scenario 3: Fine-Tuning from a Pre-trained Model

Use this to adapt a trained model to a new dataset. This loads the weights but resets the optimizer. It also automatically activates **differential learning rates**, using a much smaller learning rate for the model's core to avoid catastrophic forgetting.

You can download a pre-trained model from [here](https://zenodo.org/records/17087180).

```bash
# Use --fine_tune_from to load weights and start a new training run
tomocpt train --config_file my_config.yaml --fine_tune_from /path/to/previous_model/weights.ckpt
```
*Alternatively, set `fine_tune_from: /path/to/previous_model/weights.ckpt` in your `my_config.yaml`.*

### Scenario 4: Fine-Tuning with Knowledge Distillation

This is an advanced fine-tuning technique where a "student" model learns from a pre-trained "teacher" model.

```bash
tomocpt train \
  --config_file my_config.yaml \
  --fine_tune_from /path/to/teacher/weights.ckpt \
  --use_distillation True \
  --distill_weight 0.5
```
- `--fine_tune_from`: Path to the teacher model's weights.
- `--use_distillation`: Must be set to `True`.
- `--distill_weight`: Balances learning from new labels vs. learning from the teacher.

### Hydra-style Parameter Overrides

You can always override any parameter from your config file directly on the command line using dot notation:

```bash
# Override learning rate and precision
tomocpt train --config_file my_config.yaml optimizer.lr=0.0001 network.TORCH_FLOAT_PRECISION=16
```

## 4. Run Prediction

Finally, apply your trained model to new tomograms for particle detection.

### Config file approach:

```bash
# After filling out the infer section in my_config.yaml
tomocpt predict --config_file my_config.yaml
```

The resulting star file contains particle coordinates, ready for subsequent processing in tools like Relion.

### Re-picking without re-running inference

If a previous run was made with `save_prediction_confidence_map: true`, the confidence maps
are stored as MRC files in `predictions_dir`. Use `--skip_inference` to reuse them and only
redo peak extraction — useful for tuning `confidence_threshold` or `distance_threshold`
without paying for inference again:

```bash
tomocpt predict --config_file my_config.yaml --skip_inference --confidence_threshold 0.5
```

In this mode `weights` is not required (pipeline-v1 maps; for pipeline-v2 maps see section 5). Every MRC file in `predictions_dir` is treated as a
confidence map, unless `tomogram_dir`/`tomogram_file` is set, in which case only the maps
matching those tomogram names are used.

## 5. Multi-class picking (pipeline v2)

Pipeline v2 picks several species of different sizes in one model and one `predict` run, from
tomograms of different voxel sizes. It is opt-in: with the default
`train.data_pipeline_version: 1` everything above works as before.

What changes compared with pipeline v1:

- **No fixed-particle rescale.** A tomogram is resampled only as far as needed to keep every
  class inside a band of voxel diameters (`train.scale.band_min_vox`/`band_max_vox`, default
  10-32), never finer than native. If the classes span more than the band allows (for example
  80 A and 500 A at 10 A/px), they are split into *levels*, one working voxel size each.
- **All species of a tomogram are labelled together** (one STAR file with `rlnClassNumber`),
  so a species is no longer a negative for the others. The model has one output channel per
  class.
- **Per-class thresholds.** After training, the thresholds are calibrated on the validation
  tomograms and stored in the checkpoint; `predict` uses them.

### Example config

A complete example for the CZII 10440 dataset (6 species at 10.012 A/px) is
`data/czii/v2_config.yaml`. The parts that matter:

```yaml
prepData:
  raw_data_dir: data/czii/tomograms
  coordinate_files: data/czii/coords/all_objects.star   # one STAR, rlnClassNumber per particle
  coordinate_file_type: star
  class_id: all
  multiclass: true
  class_diameters: "1:130,2:180,3:260,4:300,5:120,6:270"   # Angstrom, per class number
  class_names: "1:beta-amylase,2:beta-galactosidase,3:thyroglobulin,4:ribosome,5:apo-ferritin,6:virus-like-particle"
  label_border_policy: clip        # keep particles cut by the tomogram border
train:
  data_pipeline_version: 2
  chunk_sampling: repeat_factor    # oversample chunks with rare classes
  augmentation:
    label_interpolation: linear    # smooth targets under rotation
  network:
    scale_conditioning: false      # see below
infer:
  predictions_dir: predictions_v2
  weights: <model_dir>/<experiment>/checkpoints/weights.ckpt
```

### Commands

```bash
tomocpt prepare_data --config_file v2_config.yaml   # one multi-class dataset per STAR file
tomocpt train --config_file v2_config.yaml          # chunks with the band rule, trains, calibrates
tomocpt predict --config_file v2_config.yaml        # every class in one run
```

`prepare_data` writes one coordinate table per tomogram with every class. `train` chunks the
data into the levels the band rule gives (the chunk format is versioned: v1 chunks are
refused by a v2 run and vice versa), trains one output channel per class, and finally
calibrates a threshold per class on the validation tomograms (the F-beta optimum, beta = 4,
for classes with at least `infer.calibration_min_particles` true particles). A failed
calibration is logged and leaves the checkpoint uncalibrated; training does not fail.

`predict` reads everything it needs (class table, band, resampling, conditioning) from the
checkpoint; `infer.length` is not used. Each class is read at the working voxel size that fits
it, and its map is resampled back to the native voxel size. Outputs:

- one coordinate file per class in every format: `<name>_<class name>.star` for RELION 3.1/5.0
  (with `rlnClassNumber`, the original class number, and `rlnAutopickFigureOfMerit`, the
  peak score) and Warp, where `<name>` is the stem of `predictions_coord_filename`;
  `<tomogram>_<class name>.mod` for IMOD. There is no merged file;
- with `save_prediction_confidence_map: true`: `<tomogram>_<class name>.mrc` per class,
  `<tomogram>_heatmap.mrc` (maximum over the classes) and `<tomogram>_classmap.mrc` (the class
  number of the strongest class above its threshold, else 0). `--skip_inference` (with
  `weights` set, for the class table) re-reads the per-class maps.

Useful `infer` options for v2:

| Option | Meaning |
|---|---|
| `classes` | Comma-separated class names to pick (default: all). |
| `class_thresholds` | `"ribosome:0.4,apo-ferritin:0.35"`; overrides the calibration. Unknown names are an error. |
| `confidence_threshold` | Used for classes with neither of the above (and for every class of an uncalibrated checkpoint). |
| `cross_class_nms` | Default on: peaks of different classes closer than half the smaller radius keep only the higher score. Removed counts per class pair are logged. |
| `apix_override` | Voxel size in A when the MRC header is wrong or 0. |
| `accumulate_dtype`, `accumulate_device` | `float16` and/or `cpu` reduce GPU memory for large tomograms with many classes (`auto` uses the GPU when the maps fit). |

### Scale conditioning (`train.network.scale_conditioning`)

**What it does.** The working voxel size of each input is fed to the network as
`t = log(apix / 10)` through a small embedding into FiLM layers (a per-channel scale and
shift) at 6 points of the decoder. The FiLM layers start at zero, so a conditioned model
starts out computing exactly what the unconditioned one does.

**Why.** Pipeline v2 no longer rescales particles to a fixed size, so one class appears at
different voxel diameters across datasets and levels. Without the voxel size, the network
cannot tell a 30-voxel object at 10 A/px (300 A) from a 30-voxel object at 20 A/px (600 A).

**When to enable it.** Training data with mixed voxel sizes, classes split into several
levels, or `train.scale.zoom_aug_max` > 1. It is pointless for a dataset at a single voxel
size whose classes fit one level, such as CZII (all six species fit the band at 10.012 A/px).

**It is fixed at training start.** `--resume_from` refuses a change of the setting.
Fine-tuning (`--fine_tune_from`) from an unconditioned or pipeline-v1 checkpoint into a
conditioned model works: the new FiLM layers start at zero.

### Class head (`train.particles.class_head`)
- `sigmoid` (default): one independent probability per class.
- `softmax`: the classes and an implicit background compete for each voxel, so one particle
  is not picked as two classes. No extra parameters; with one class it equals `sigmoid`.
  Classes not annotated in a training sample are left out of the competition.
- `train.particles.cross_class_neg_weight` (default 0) makes sigmoid models penalise firing on
  another class's particles. With `softmax` it is not needed.

### k-fold training and fold ensembles
Train one model per fold of tomograms, calibrate the per-class thresholds on the pooled
out-of-fold predictions, and predict with the average of the fold models:
```bash
# fold k of K (one chunks_dir and model_dir per fold)
tomocpt train --config_file fold_k.yaml        # with train.cv_folds: K, train.cv_fold: k
python -m tomocpt.predict.ensemble fold*/model/**/*.ckpt --out ensemble.json --config_file config.yaml
tomocpt predict --config_file config.yaml --weights ensemble.json --tomogram_dir tomos/ --predictions_dir out/
```
`experiments/run_cv.sh` runs all steps for CZII and reports the cross-validated F4. An explicit
`infer.class_thresholds` still overrides the ensemble's thresholds.

### Benchmarks

`benchmarks/synthetic_multisize.py` (phantoms from 80 to 500 A, two levels) and
`benchmarks/czii_multiclass.py` (CZII 10440, paths given by you) compare four arms: one v1
model per species, a conditioned v2 model, an unconditioned v2 model, and a conditioned v2
model warm-started from the public TomoCPT model (`--pretrained`). They write
`benchmarks/out/<run>/report.json` and `report.md` (per-class F-beta, precision and recall,
cross-class suppression counts, time per tomogram, GPU memory, thresholds and their source).
`python benchmarks/synthetic_multisize.py --tiny` is a quick smoke run (1 epoch, small network;
arm (d) then warm-starts from the tiny v1 model of the first arm instead of the public model).

## Downloading data from the CryoET Data Portal

`scripts/download_portal_data.py` downloads tomograms and point annotations from the
[CryoET Data Portal](https://cryoetdataportal.czscience.com) and writes STAR files and a
`prepData` block that `tomocpt prepare_data` can use directly.

```bash
pip install -e '.[portal]'

# See what a dataset has, without downloading
python scripts/download_portal_data.py --dataset-id 10440 --out data/czii --list

# CZII object-identification training data at 10.012 A/px
python scripts/download_portal_data.py --dataset-id 10440 --out data/czii \
    --voxel-spacing 10.012 --processing denoised --diameters ribosome=300
```

It writes `tomograms/<run>.mrc`, one STAR file per object in `coords/`, `coords/all_objects.star`
(one class number per object), `manifest.json`, and `prepdata_snippet.yaml`. Objects without a
`--diameters` value get `???` in the snippet. Fill those in before `prepare_data`.
Downloads resume after an interruption and are checked against the portal's file size.
The script stops if annotation points fall outside the tomogram, which usually means a unit
or alignment mismatch.

## Performance Tips

1.  **GPU Acceleration**: Enable CUDA for both training and inference for best performance.
2.  **Batch Size**: Adjust the batch size based on your GPU memory.
3.  **Precision**: Using lower precision (e.g., `network.TORCH_FLOAT_PRECISION=16`) can accelerate training.
4.  **Gradient Accumulation**: If you have limited GPU memory, use `gradient_accumulation_steps` in your config to train with a larger effective batch size.

# Changelog

## Unreleased (v2, multi-class picking)
- Pipeline v2 (opt-in, `train.data_pipeline_version: 2`): band scale policy instead of the
  fixed-particle rescale, multi-class datasets (`prepData.multiclass`), one output channel per
  class, optional FiLM conditioning on the voxel size (`train.network.scale_conditioning`).
- `tomocpt predict` runs pipeline-v2 checkpoints: every class in one run, each at the level
  that fits it, one coordinate file per class in every format, per-class and merged maps.
- Per-class threshold calibration after v2 training, stored in the checkpoint; overridable
  with `infer.class_thresholds`. Cross-class suppression (`infer.cross_class_nms`).
- New `infer` options: `apix_override`, `classes`, `cross_class_nms`, `accumulate_dtype`,
  `accumulate_device`, `class_thresholds`, `calibration_beta`, `calibration_min_particles`.
  `infer.length` is now optional (still required for pipeline-v1 checkpoints).
- Fixed: for thin tomograms (an axis shorter than 64 voxels after the rescale), the prediction
  map was cropped on the wrong axis before being resized back, which displaced or lost picks
  on such tomograms. Other v1 predictions are unchanged.
- Benchmarks in `benchmarks/` (synthetic multi-size phantoms, CZII 10440).
- Softmax class head (`train.particles.class_head: softmax`) and cross-class negative
  weighting for sigmoid heads (`train.particles.cross_class_neg_weight`).
- k-fold training (`train.cv_folds`, `train.cv_fold`), out-of-fold calibration and fold
  ensembles (`python -m tomocpt.predict.ensemble`, `infer.weights` = ensemble file).

## 0.7.1
- Self-supervised training (`train.mode = selfSupervised`) works again. It had been broken
  since October 2025: the model could not be built or loaded from checkpoints, its rotation
  and contrastive heads were never trained, and it did not log `val_loss_smooth` (needed by
  the LR scheduler and early stopping). The "recons" TensorBoard panel now shows the
  reconstruction instead of the input. A picker can be fine-tuned from an SSL checkpoint.
- Fine-tuning now trains the output layer (`out`), at the full learning rate. It was in no
  optimizer group before, so it stayed frozen. **Fine-tuned models will differ from those
  trained with earlier versions.** Training from scratch is unchanged.
- Resuming a fine-tuning run keeps its differential learning rates instead of failing to load
  the saved optimizer state. Runs saved before this release resume with their original groups.
- `prepare_data` with IMOD models no longer fails with a NameError, and draws labels at the
  correct positions (X and Z were swapped).
- The `desired_particle_pixel_size` help text now says diameter (as the code does), not radius.

## 0.7.0
- Added missing-wedge and amplitude-spectrum Fourier augmentation, plus brightness-gradient,
  local-gamma, brightness, contrast, and contrast-inversion intensity augmentations, exposed
  via `train.augmentation.*` config/CLI options.
- Fixed multi-species chunking collisions and train/validation leakage when several datasets
  share the same tomograms.
- Fixed an invalid `config.yaml` value (`infer.predictions_coord_format`) that blocked config
  loading entirely.

## 0.6.0
- Added `--skip_inference` to `tomocpt predict`, which reuses the confidence maps saved by a
  previous run and only redoes peak extraction. Useful for tuning `confidence_threshold` or
  `distance_threshold` without paying for inference again. `weights` is no longer required
  when this flag is used.
- Fixed border exclusion silently discarding every peak. The excluded border was the full
  particle diameter in pixels, so on volumes thinner than twice that — routine for binned
  tomograms in Z — the whole volume was masked and no particles were picked. The border is
  now clamped per axis, with a warning when it is reduced.
- Fixed peak positions on saturated, flat-topped blobs. Every voxel of a plateau compares
  equal to its neighbourhood, so an arbitrary member was reported, up to several voxels off
  centre, and sub-pixel refinement could not recover it. Plateaus now collapse to their
  centroid.
- Improved sub-pixel accuracy by scaling the fitting patch to the particle radius instead of
  fixing it at 9 voxels. The old width overran the particle for small radii, pulling the
  background floor into the fit. Coordinates change slightly for all users; the improvement
  is largest for heavily binned data, where the error drops by up to a factor of 13.

## 0.5.2
- Added gradient accumulation for training on machines with lower memory.
- Added a `shared` section in `config.yaml` to specify shared variables only once.
# Development
tomoCPT is jointly developed by Ruben Sanchez-Garcia and Pranav NM Shah at the University of Oxford.

## CITE
[TomoCPT: a generalizable model for 3D particle detection and localization in cryo-electron tomograms](https://doi.org/10.1107/S2059798325000865)

Shah PNM, Sanchez-Garcia R, Stuart DI. *Acta Crystallographica Section D: Structural Biology*, 81(2):63-76, 2025.
## License

tomocpt is distributed under the GNU Affero General Public License v3 (see
[LICENSE](LICENSE)). Some vendored components carry their own licenses; see
[NOTICE](NOTICE) for attributions.
