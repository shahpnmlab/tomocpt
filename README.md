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
│ --help          Show this message and exit.                                                                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ init           Function to create a template config file for running tomoCPT, only including annotated fields                                                                           │
│ prepare_data   Prepares particle picking datasets by processing multiple tomograms and their corresponding coordinate files.                                                            │
│ train          Trains a deep learning model for particle picking or self-supervised learning using PyTorch Lightning.                                                                   │
│ predict        Performs parallel inference on tomogram data for particle detection and coordinate extraction.                                                                           │
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

```yaml
# my_config.yaml (sample content)
prepData:
  raw_data_dir: ???  # Path to directory containing tomogram .mrc files (REQUIRED)
  training_data_dir: ???  # Path where processed volume-label pairs will be stored (REQUIRED)
  particle_length_ang: ???  # Particle diameter in Angstroms, e.g. "150" (REQUIRED)
  coordinate_files: ???  # Path to coordinate files (.star or .mod) with particle positions (REQUIRED)
  coordinate_file_type: star  # Format of coordinate files: "star" or "imod"
  class_id: all  # Particle class ID to use; "all" for all classes or specific IDs like "1" or "1:2:3"
  desired_particle_pixel_size: 20  # Target radius of particles in pixels after preprocessing
train:
  training_data_dir: null  # Optional override for prepared data location
  chunks_dir: ???  # Directory to store preprocessed data chunks for training (REQUIRED)
  model_dir: ???  # Directory to save model weights and training logs (REQUIRED)
  n_epochs: 10  # Number of training epochs
  batch_size: 2  # Number of samples per batch; increase for faster training if GPU memory allows
  use_gpus: true  # Whether to use GPU acceleration for training
  n_cpus_for_train: 2  # Number of CPU workers for data loading
  experiment_name: tomocpt  # Name for this training run (affects logging directories)
  mode: picking  # Training mode: "picking" (supervised) or "selfSupervised"
  train_on: tomos  # How to split train/validation: by "tomos" or "chunks"
  restore_full_state: true  # Whether to restore optimizer state when continuing training
  optimizer:
    lr: 0.0004  # Learning rate
  network:
    model_type: SwinUNETR  # Neural network architecture: "SwinUNETR" or "Unet"
    TORCH_FLOAT_PRECISION: bf16  # Floating-point precision: "bf16", "16" or "32"
  launch_tensorboard: true  # Whether to launch TensorBoard for monitoring training
infer:
  tomogram_dir: ???  # Directory containing tomograms for inference (REQUIRED)
  predictions_dir: ???  # Directory to save prediction results (REQUIRED)
  weights: ???  # Path to trained model weights file (.ckpt) (REQUIRED)
  masks_dir: null  # Optional directory with masks to limit prediction to specific regions
  length: ???  # Particle diameter in Angstroms; should match training value (REQUIRED)
  distance_threshold: null  # Minimum distance between particles in Angstroms; null for auto
  predictions_coord_filename: tomopicker_coords.star  # Filename for output coordinates
  predictions_coord_format: relion  # Format for output: "relion31", "relion50", or "warp"
  save_prediction_confidence_map: false  # Whether to save probability maps as .mrc files
  save_predicted_coords: true  # Whether to save predicted coordinates to star file
  confidence_threshold: 0.3  # Minimum confidence score (0-1) for particle detection
  predictions_batch_size: 2  # Batch size for inference; adjust based on GPU memory
  oversubscribe_factor: 1  # Number of tomograms to process in parallel per GPU
  use_cuda: true  # Whether to use GPU acceleration for inference
  n_cpus_per_gpu: 1  # Number of CPU workers per GPU for preprocessing
```

Edit this file to set the paths and parameters according to your needs before proceeding to the next steps. 
If working with multiple datasets, use comma separated values.

## 2. Generate Volume-Label Pairs

This step processes your raw tomograms and coordinate files into a format suitable for training.

### Command-line approach:

```bash
tomocpt prepare_data \
  --raw-data-dir /path/to/tomograms/ \
  --prepared-data-dir /path/to/output/ \
  --particle-length-ang 150 \
  --coordinate-files /path/to/coordinates.star \
  --coordinate-file-type star \
  --desired-particle-pixel-size 20
```

### Config file approach:

```bash
# After filling out the prepData section in my_config.yaml
tomocpt prepare_data --config-file /path/to/config.yaml
```

### Combined approach:

```bash
# Use config file but override specific parameters
tomocpt prepare_data ---config-file /path/to/config.yaml --particle-length-ang 200 --config_merge_preference command
```

### Expected result:

This process creates a structured directory containing preprocessed tomogram data and corresponding labels. The console will show progress as it processes each tomogram:

```
2025-02-26 23:38:07 - INFO - Close match found for tomogram TS_11: TS_11_8.00Apx
2025-02-26 23:38:07 - INFO - Close match found for tomogram TS_1: TS_1_8.00Apx
2025-02-26 23:38:07 - INFO - Saved tracking information to /path/to/inputs/dataset_1_class_all/star_picking_tracking_all.csv
Generating particle masks: 100%|██████████████████████████████████████| 2/2 [00:00<00:00,  5.62it/s]
2025-02-26 23:38:07 - INFO - Successfully processed dataset 1
2025-02-26 23:38:07 - INFO - Completed processing all datasets
```

## 3. Train a Model

Now train the model using the prepared data.

### Command-line approach:

```bash
tomocpt train \
  --chunks-dir /tmp/chunks/ \
  --model-dir /path/to/where/weights/shoud/be/saved/weights/ \
  --experiment-name "first_model" \
  --n-epochs 100 \
  --batch-size 4 \
  --network.model_type SwinUNETR
```

### Config file approach:

```bash
# After filling out the train section in my_config.yaml
tomocpt train --config-file my_config.yaml
```

### Training with fine-tuning from existing weights:
You can download a pre-trained model from [here].(https://zenodo.org/records/14871940)
```bash
tomocpt train \
  --config-file my_config.yaml \
  --train-continue /path/to/models/previous_model/weights.ckpt
```

### Hydra-style parameter overrides:

```bash
# Override nested parameters using dot notation
tomocpt train --config-file my_config.yaml \
  train.optimizer.lr=0.0001 \
  train.network.TORCH_FLOAT_PRECISION=16
```

### Expected output:

The training process will display progress and metrics:

```
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
Epoch 1/100: 100%|██████████| 45/45 [01:23<00:00]
loss: 0.427, val_loss: 0.392
Epoch 2/100: 100%|██████████| 45/45 [01:22<00:00]
loss: 0.315, val_loss: 0.301
...
```

A TensorBoard instance will automatically launch (if configured) for visualizing training progress.

## 4. Run Prediction

Finally, apply your trained model to new tomograms for particle detection.

### Command-line approach:

```bash
tomocpt predict \
  --tomogram-dir /path/to/new/tomograms/ \
  --predictions-dir /path/to/predictions/ \
  --weights /path/to/models/model1/unnamed/checkpoints/weights.ckpt \
  --length 150 \
  --confidence-threshold 0.35 \
  --predictions-coord-format relion31
```

### Config file approach:

```bash
# After filling out the infer section in my_config.yaml
tomocpt predict --config-file my_config.yaml
```

### With masked regions:

```bash
tomocpt predict \
  --config-file my_config.yaml \
  --masks-dir /path/to/masks/
```

### Expected results:

The prediction process processes each tomogram and outputs coordinates:

```
Processing tomo_01.mrc: 100%|██████████| [02:15<00:00]
Processing tomo_02.mrc: 100%|██████████| [02:08<00:00]
Predicted coordinates are stored here: /path/to/predictions/tomopicker_coords.star
```

The resulting star file contains particle coordinates in the format specified, ready for subsequent processing in tools like RELION.

## Advanced Configuration

### Creating and Using a Complete Config File

Below is an example of a complete configuration file combining all parameters:

```yaml
# complete_config.yaml
prepData:
  raw_data_dir: "/data/tomograms/"
  prepared_data_dir: "/data/prepared/"
  particle_length_ang: "150"
  coordinate_files: "/data/coordinates/particles.star"
  coordinate_file_type: "star"
  class_id: "all"
  desired_particle_pixel_size: 20
  USE_CUDA_FOR_DATA: true
  ALPHA_FOR_DROPPING_EMPTY_CUBES: 1.0

train:
  training_data_dir: "/data/prepared/"
  chunks_dir: "/data/chunks/"
  model_dir: "/models/mymodel/"
  n_epochs: 100
  batch_size: 4
  use_gpus: true
  n_cpus_for_train: 4
  experiment_name: "tomocpt_model1"
  mode: "picking"
  train_on: "tomos"
  restore_full_state: true
  launch_tensorboard: true
  network:
    model_type: "SwinUNETR"
    TORCH_FLOAT_PRECISION: "bf16"
  optimizer:
    _target_: "torch.optim.RAdam"
    lr: 4.0e-4
    weight_decay: 1.0e-8

infer:
  tomogram_dir: "/data/new_tomograms/"
  predictions_dir: "/predictions/"
  weights: "/models/mymodel/unnamed/checkpoints/weights.ckpt"
  masks_dir: null
  length: 150
  distance_threshold: 100
  predictions_coord_filename: "tomopicker_coords.star"
  predictions_coord_format: "relion31"
  save_prediction_confidence_map: false
  save_predicted_coords: true
  confidence_threshold: 0.3
  predictions_batch_size: 2
  oversubscribe_factor: 1
  use_cuda: true
  n_cpus_per_gpu: 1
```

To use this complete configuration file:

```bash
# Run the full pipeline with a single config file
tomocpt initialize_config --output-path complete_config.yaml
# Edit complete_config.yaml with your specific paths and parameters
tomocpt prepare_data --config-file complete_config.yaml
tomocpt train --config-file complete_config.yaml
tomocpt predict --config-file complete_config.yaml
```

## Performance Tips

1. **GPU Acceleration**: Enable CUDA for both training and inference for best performance:
   ```yaml
   train:
     use_gpus: true
   infer:
     use_cuda: true
   ```

2. **Batch Size**: Adjust the batch size based on your GPU memory. Larger batch sizes can speed up training but require more memory:
   ```bash
   tomocpt train --batch-size 8  # Increase if you have sufficient GPU memory
   ```

3. **Parallel Processing**: For prediction, the `oversubscribe_factor` parameter controls how many tomograms are processed in parallel per GPU:
   ```bash
   tomocpt predict --oversubscribe-factor 2  # Process 2 tomograms per GPU simultaneously
   ```

4. **Precision**: Using lower precision can accelerate training with minimal quality loss:
   ```bash
   tomocpt train train.network.TORCH_FLOAT_PRECISION=16  # Use FP16 instead of default
   ```

By following these steps, you'll be able to fully utilize tomoCPT for particle picking in your cryo-electron tomography workflow.

# Changelog
# Development
tomoCPT is jointly developed by Ruben Sanchez-Garcia and Pranav NM Shah at the University of Oxford.

## CITE
[TomoCPT: a generalizable model for 3D particle detection and localization in cryo-electron tomograms](https://doi.org/10.1107/S2059798325000865)

Shah PNM, Sanchez-Garcia R, Stuart DI. *Acta Crystallographica Section D: Structural Biology*, 81(2):63-76, 2025.
