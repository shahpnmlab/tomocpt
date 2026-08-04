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
tomocpt init --output-path ./my_config.yaml
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
tomocpt prepare_data --config-file my_config.yaml
```

### Combined approach:

```bash
# Use config file but override specific parameters
tomocpt prepare_data --config-file my_config.yaml --particle-length-ang 200
```

## 3. Train a Model

This section covers the main training workflows. All parameters can be set either via the command line or in your `.yaml` configuration file.

### Scenario 1: Training from Scratch

This is the most basic scenario, where you train a new model from randomly initialized weights.

```bash
# Using a configuration file is recommended
tomocpt train --config-file my_config.yaml
```

### Scenario 2: Resuming an Interrupted Run

If your training run is aborted, you can resume it seamlessly from the last saved checkpoint. This restores the model, optimizer, and learning rate scheduler to their exact previous states.

```bash
# The command is the same as your original run, just add --resume-from
tomocpt train --config-file my_config.yaml --resume-from /path/to/model_dir/weights.ckpt
```
*Alternatively, set `resume_from: /path/to/model_dir/weights.ckpt` in your `my_config.yaml`.*

### Scenario 3: Fine-Tuning from a Pre-trained Model

Use this to adapt a trained model to a new dataset. This loads the weights but resets the optimizer. It also automatically activates **differential learning rates**, using a much smaller learning rate for the model's core to avoid catastrophic forgetting.

You can download a pre-trained model from [here](https://zenodo.org/records/17087180).

```bash
# Use --fine-tune-from to load weights and start a new training run
tomocpt train --config-file my_config.yaml --fine-tune-from /path/to/previous_model/weights.ckpt
```
*Alternatively, set `fine_tune_from: /path/to/previous_model/weights.ckpt` in your `my_config.yaml`.*

### Scenario 4: Fine-Tuning with Knowledge Distillation

This is an advanced fine-tuning technique where a "student" model learns from a pre-trained "teacher" model.

```bash
tomocpt train \
  --config-file my_config.yaml \
  --fine-tune-from /path/to/teacher/weights.ckpt \
  --use-distillation True \
  --distill-weight 0.5
```
- `--fine-tune-from`: Path to the teacher model's weights.
- `--use-distillation`: Must be set to `True`.
- `--distill-weight`: Balances learning from new labels vs. learning from the teacher.

### Hydra-style Parameter Overrides

You can always override any parameter from your config file directly on the command line using dot notation:

```bash
# Override learning rate and precision
tomocpt train --config-file my_config.yaml optimizer.lr=0.0001 network.TORCH_FLOAT_PRECISION=16
```

## 4. Run Prediction

Finally, apply your trained model to new tomograms for particle detection.

### Config file approach:

```bash
# After filling out the infer section in my_config.yaml
tomocpt predict --config-file my_config.yaml
```

The resulting star file contains particle coordinates, ready for subsequent processing in tools like Relion.

### Re-picking without re-running inference

If a previous run was made with `save_prediction_confidence_map: true`, the confidence maps
are stored as MRC files in `predictions_dir`. Use `--skip-inference` to reuse them and only
redo peak extraction — useful for tuning `confidence_threshold` or `distance_threshold`
without paying for inference again:

```bash
tomocpt predict --config-file my_config.yaml --skip-inference --confidence-threshold 0.5
```

In this mode `weights` is not required. Every MRC file in `predictions_dir` is treated as a
confidence map, unless `tomogram_dir`/`tomogram_file` is set, in which case only the maps
matching those tomogram names are used.

## Performance Tips

1.  **GPU Acceleration**: Enable CUDA for both training and inference for best performance.
2.  **Batch Size**: Adjust the batch size based on your GPU memory.
3.  **Precision**: Using lower precision (e.g., `network.TORCH_FLOAT_PRECISION=16`) can accelerate training.
4.  **Gradient Accumulation**: If you have limited GPU memory, use `gradient_accumulation_steps` in your config to train with a larger effective batch size.

# Changelog

## 0.6.0
- Added `--skip-inference` to `tomocpt predict`, which reuses the confidence maps saved by a
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