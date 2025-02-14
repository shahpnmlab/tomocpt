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
│ initialize_config         Function to create a template config file for running tomoCPT, only including annotated fields                                                                │
│ prepare_vol_label_pairs   Process multiple datasets based on configuration                                                                                                              │
│ train                                                                                                                                                                                   │
│ predict                                                                                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

# Usage
### Outline of tomocpt workflow

tomoCPT is a command line driven tool for the purposes of training and using a swuinUNETR based network for 3D particle
picking.
<details>
<summary><strong>Training from scratch</strong></summary>
</details>
<details>
<summary><strong>Fine-tuning a network</strong></summary>
</details>
<details>
<summary><strong>Predicting centroids</strong></summary>

Run inference on tomograms to detect and locate particles.

## Required Arguments

- `--tomogram-dir PATH`  
  The directory containing the input tomograms to be processed.
- `--predictions-dir PATH`  
  The directory where prediction outputs will be saved.
- `--weights PATH`  
  Path to the model weights file.
- `--length FLOAT`  
  Particle size in Angstroms. For spherical particles, this is the diameter. For non-spherical particles, this should be the longest axis.

## Optional Arguments
- `--masks-dir PATH`  
  Directory containing masks for targeted picking. Mask filenames should match their corresponding tomogram names.
- `--distance-threshold FLOAT`  
  Minimum distance in Angstroms between particles. Particles closer than this threshold will be filtered out.
- `--predictions-coord-filename STR`  
  Name of the output coordinates file in STAR format.  
  Default: "tomopicker_coords.star"
- `--predictions-coord-format {relion31}`  
  Format for the output coordinates file.  
  Default: relion31
- `--confidence-threshold FLOAT`  
  Confidence threshold for particle selection. Only particles with confidence scores above this threshold will be included in the output.  
  Default: 0.3
- `--predictions-batch-size INT`  
  Batch size for processing predictions.  
  Default: 2
- `--oversubscribe-factor INT`  
  Number of tomograms to process in parallel per GPU.  
  Default: 1
- `--n-cpus-per-gpu INT`  
  Number of CPU workers per GPU for data preprocessing.  
  Default: 1
## Output Control

- `--save-prediction-confidence-map`  
  Save the predicted label tomograms.  
  Default: False
- `--save-predicted-coords`  
  Save particle coordinates to a STAR file.  
  Default: True
## Hardware Options
- `--use-cuda`  
  Enable CUDA GPU acceleration.  
  Default: True
## System Constants
These values are fixed in the system configuration:
- `PATCH_OVERLAP_FACTOR`: 4
- `N_GPUS`: 1
- `N_CPUS_IF_NO_GPU`: 32
- `USE_CUDA_FOR_DATA`: True
</details>

# Changelog
# Development
tomoCPT is jointly developed by Ruben Sanchez-Garcia and Pranav NM Shah at the University of Oxford.

## CITE
[TomoCPT: a generalizable model for 3D particle detection and localization in cryo-electron tomograms](https://doi.org/10.1107/S2059798325000865)

Shah PNM, Sanchez-Garcia R, Stuart DI. *Acta Crystallographica Section D: Structural Biology*, 81(2):63-76, 2025.
