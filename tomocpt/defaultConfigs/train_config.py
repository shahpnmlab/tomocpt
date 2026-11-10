from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Annotated, Tuple

import typer
from omegaconf import MISSING

from tomocpt.defaultConfigs.network_config import NetworkConfig
from tomocpt.defaultConfigs.scale_config import ScaleConfig
from tomocpt.defaultConfigs.particles_config import ParticlesConfig
from tomocpt.defaultConfigs.ssl_config import SSLConfig
from tomocpt.defaultConfigs.discriminative_config import DiscriminativeConfig
from tomocpt.defaultConfigs.otf_config import OnTheFlyConfig


class TrainingModes(str, Enum):
    selfSupervised = "selfSupervised"
    picking = "picking"


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.RAdam"
    lr: Annotated[float, typer.Option(help="Learning rate")] = 4e-4
    weight_decay: float = 1e-8
    betas: Tuple[float, float] = (0.9, 0.999)
    # decoupled_weight_decay=True


class CrossValidationLevelSplit(str, Enum):
    tomos = "tomos"
    cubes = "chunks"


class ChunkSampling(str, Enum):
    uniform = "uniform"
    repeat_factor = "repeat_factor"


class LabelInterpolation(str, Enum):
    nearest = "nearest"
    linear = "linear"


class RotationMode(str, Enum):
    affine = "affine"
    flips = "flips"


@dataclass
class AugmentationConfig:
    """Training-time augmentation pipeline.

    Defaults reproduce tomocpt's historical behaviour exactly: the three original
    TorchIO transforms at their original probabilities, and every new
    augmentation switched off. Existing runs are therefore unchanged until a
    flag is set explicitly.
    """

    # --- Existing spatial transforms (defaults == pre-existing behaviour) ---
    affine_p: Annotated[
        float, typer.Option(help="Probability of applying a random affine (image and label together)")
    ] = 0.8
    affine_degrees: Annotated[
        float, typer.Option(help="Maximum random rotation in degrees for the affine transform")
    ] = 45.0
    elastic_blur_p: Annotated[
        float, typer.Option(help="Probability of applying one of elastic deformation / blur")
    ] = 0.75
    elastic_weight: Annotated[
        float, typer.Option(help="Relative weight of elastic deformation within the elastic/blur choice")
    ] = 0.1
    blur_weight: Annotated[
        float, typer.Option(help="Relative weight of blur within the elastic/blur choice")
    ] = 0.1
    blur_std: Annotated[
        float, typer.Option(help="Standard deviation of the random blur kernel")
    ] = 1.0
    label_interpolation: Annotated[
        LabelInterpolation,
        typer.Option(
            help="Interpolation of the target in the spatial transforms: 'nearest' (legacy) or 'linear' "
                 "(smooth soft targets; recommended with pipeline v2). The target stays a LabelMap."
        ),
    ] = LabelInterpolation.nearest
    rotation_mode: Annotated[
        RotationMode,
        typer.Option(
            help="'affine' (legacy random rotation + elastic/blur) or 'flips' (axis flips, no elastic "
                 "deformation; keeps the real missing wedge in place)"
        ),
    ] = RotationMode.affine
    flip_p: Annotated[
        float, typer.Option(help="Per-axis flip probability when rotation_mode='flips' (suggestion 0.5)")
    ] = 0.0

    # --- Fourier degradation (missing wedge + amplitude spectrum) ---
    # Two independent switches, mirroring membrain-seg's use_mw_aug / use_fourier_aug.
    use_mw_aug: Annotated[
        bool, typer.Option(help="Enable missing-wedge augmentation (zeroes a wedge of Fourier coefficients)")
    ] = False
    use_fourier_aug: Annotated[
        bool, typer.Option(help="Enable amplitude-spectrum randomization (simulates varied CTF/defocus/dose filtering)")
    ] = False
    missing_wedge_prob: Annotated[
        float, typer.Option(help="Probability of applying the missing wedge")
    ] = 0.5
    amplitude_prob: Annotated[
        float, typer.Option(help="Probability of applying amplitude-spectrum randomization")
    ] = 0.5
    sample_kernel_prob: Annotated[
        float,
        typer.Option(
            help="Probability of localizing the degradation to a soft blob when the wedge did not fire. "
                 "Whenever the wedge fires the blend is forced on regardless."
        ),
    ] = 0.5
    keep_angle_min: Annotated[
        float,
        typer.Option(
            help="Minimum half-opening angle in degrees of the RETAINED wedge: 90 keeps everything, 0 keeps nothing"
        ),
    ] = 45.0
    keep_angle_max: Annotated[
        float, typer.Option(help="Maximum half-opening angle in degrees of the retained wedge")
    ] = 45.0
    smooth_sigma_min: Annotated[
        float, typer.Option(help="Minimum Gaussian smoothing sigma for the synthesized amplitude spectrum")
    ] = 2.0
    smooth_sigma_max: Annotated[
        float, typer.Option(help="Maximum Gaussian smoothing sigma for the synthesized amplitude spectrum")
    ] = 4.0
    step_sigma_min: Annotated[
        float, typer.Option(help="Minimum random-walk step sigma for the synthesized amplitude spectrum")
    ] = 0.1
    step_sigma_max: Annotated[
        float, typer.Option(help="Maximum random-walk step sigma for the synthesized amplitude spectrum")
    ] = 4.0
    offset_min: Annotated[
        float, typer.Option(help="Minimum offset for the synthesized amplitude spectrum")
    ] = 2.0
    offset_max: Annotated[
        float, typer.Option(help="Maximum offset for the synthesized amplitude spectrum")
    ] = 10.0
    taper_width: Annotated[
        float,
        typer.Option(
            help="Raised-cosine taper width in degrees on the wedge boundary. 0 is the hard-edged "
                 "reference behaviour, which causes Gibbs ringing a picking model can latch onto."
        ),
    ] = 0.0
    reclip: Annotated[
        bool, typer.Option(help="Re-apply the +/- clip_value hard clip after the Fourier degradation")
    ] = True
    clip_value: Annotated[
        float, typer.Option(help="Clip bound matching robust_normalization's hard clip")
    ] = 3.0

    # --- Intensity augmentations (all off by default) ---
    brightness_gradient_p: Annotated[
        float, typer.Option(help="Probability of adding a localized Gaussian brightness blob")
    ] = 0.0
    local_gamma_p: Annotated[
        float, typer.Option(help="Probability of applying a gamma correction within a soft blob")
    ] = 0.0
    brightness_p: Annotated[
        float, typer.Option(help="Probability of a global Gaussian brightness shift")
    ] = 0.0
    brightness_sigma: Annotated[
        float, typer.Option(help="Standard deviation of the global brightness shift")
    ] = 0.5
    contrast_p: Annotated[
        float, typer.Option(help="Probability of scaling contrast about the image mean")
    ] = 0.0
    contrast_min: Annotated[
        float, typer.Option(help="Minimum multiplicative contrast factor")
    ] = 0.5
    contrast_max: Annotated[
        float, typer.Option(help="Maximum multiplicative contrast factor")
    ] = 2.0
    contrast_preserve_range: Annotated[
        bool, typer.Option(help="Clamp the contrast result back into the input's original range")
    ] = False

    # --- TorchIO stand-ins for reference augmentations (all off by default) ---
    anisotropy_p: Annotated[
        float, typer.Option(help="Probability of tio.RandomAnisotropy (stands in for SimulateLowResolution)")
    ] = 0.0
    noise_p: Annotated[
        float, typer.Option(help="Probability of tio.RandomNoise (additive Gaussian noise)")
    ] = 0.0
    noise_std: Annotated[
        float, typer.Option(help="Maximum standard deviation for tio.RandomNoise")
    ] = 0.1
    gamma_p: Annotated[
        float, typer.Option(help="Probability of tio.RandomGamma (global gamma)")
    ] = 0.0
    bias_field_p: Annotated[
        float,
        typer.Option(
            help="Probability of tio.RandomBiasField, a smooth MULTIPLICATIVE field. "
                 "Complementary to the additive brightness gradient, not redundant with it."
        ),
    ] = 0.0


@dataclass
class TrainConfig:
    
    model_dir: Annotated[
        Path,
        typer.Option(help="The directory where the training weights will be saved"),
    ] = MISSING
    training_data_dir: Annotated[
        Optional[Path], typer.Option(help="Path to where the volume label pairs are stored")
    ] = None
    chunks_dir: Annotated[
        Optional[Path], typer.Option(help="Path to directory containing chunked data")
    ] = MISSING
    n_epochs: Annotated[int, typer.Option(help="Number of epochs to train")] = 10
    batch_size: Annotated[int, typer.Option(help="batch size")] = 16
    gradient_accumulation_steps: Annotated[int, typer.Option(help="Accumulate gradients over N steps")] = 1
    use_gpus: Annotated[bool, typer.Option(help="use cuda for training")] = True
    n_cpus_for_preprocessing: Annotated[
        int, typer.Option(help="Number of CPU workers for the initial chunking/preprocessing step.")
    ] = 8
    n_cpus_for_dataloading: Annotated[
        int, typer.Option(help="Number of CPU workers per GPU for the PyTorch DataLoader during training.")
    ] = 8
    experiment_name: Annotated[
        Optional[str], typer.Option(help="The name of the experiment")
    ] = "tomocpt"
    mode: Annotated[TrainingModes, typer.Option(help="The training mode")] = (
        TrainingModes.picking
    )
    train_on: Annotated[
        CrossValidationLevelSplit,
        typer.Option(help="Whether to split train-val on chunks or tomograms"),
    ] = CrossValidationLevelSplit.tomos
    resume_from: Annotated[
        Optional[Path],
        typer.Option(help="Path to a checkpoint to resume a previously interrupted training run.")
    ] = None
    fine_tune_from: Annotated[
        Optional[Path],
        typer.Option(help="Path to a checkpoint to use as a starting point for fine-tuning or distillation.")
    ] = None
    optimizer: Annotated[OptimizerConfig, typer.Option(help="The optimizer")] = field(
        default_factory=OptimizerConfig
    )
    network: Annotated[NetworkConfig, typer.Option(help="The network config")] = field(
        default_factory=NetworkConfig
    )
    augmentation: Annotated[
        AugmentationConfig, typer.Option(help="Training-time augmentation settings")
    ] = field(default_factory=AugmentationConfig)
    launch_tensorboard: Annotated[
        bool, typer.Option(help="Launch tensorboard for evaluating training")
    ] = False

    # Data pipeline v2 (Gate 2)
    data_pipeline_version: Annotated[
        int,
        typer.Option(help="Data pipeline: 1 = legacy fixed-particle rescale, 2 = band scale policy, multi-class"),
    ] = 1
    scale: Annotated[
        ScaleConfig, typer.Option(help="Scale policy of data pipeline v2")
    ] = field(default_factory=ScaleConfig)
    particles: Annotated[
        ParticlesConfig, typer.Option(help="Multi-class picking settings (pipeline v2)")
    ] = field(default_factory=ParticlesConfig)
    chunk_sampling: Annotated[
        ChunkSampling,
        typer.Option(help="Training chunk sampling: 'uniform' (legacy) or 'repeat_factor' (pipeline v2 only)"),
    ] = ChunkSampling.uniform
    repeat_factor_threshold: Annotated[
        float, typer.Option(help="Threshold t of repeat-factor sampling: r_c = max(1, sqrt(t / f_c))")
    ] = 0.1
    sampling: Annotated[
        str, typer.Option(help="Where training crops come from (pipeline v2): 'chunks' (chunk files) or "
                               "'on_the_fly' (crops drawn from one array per tomogram, Gate 8c)")
    ] = "chunks"
    otf: Annotated[
        OnTheFlyConfig, typer.Option(help="On-the-fly crop sampling (train.sampling=on_the_fly)")
    ] = field(default_factory=OnTheFlyConfig)
    cv_folds: Annotated[
        int, typer.Option(help="k-fold training by tomogram (pipeline v2): number of folds; 0 = off (random split)")
    ] = 0
    cv_fold: Annotated[
        int, typer.Option(help="k-fold training: the fold used for validation, in [0, cv_folds)")
    ] = 0
    val_stems: Annotated[
        Optional[str], typer.Option(help="Fixed validation split by tomogram (pipeline v2): comma-separated tomogram "
                                         "stems; None = random split. Cannot be combined with cv_folds")
    ] = None
    early_stop_min_delta: Annotated[
        float, typer.Option(help="Early stopping on val_loss_smooth: minimum absolute improvement that counts "
                                 "(legacy 0.005)")
    ] = 0.005
    early_stop_patience: Annotated[
        int, typer.Option(help="Early stopping: epochs without improvement before training stops")
    ] = 10
    early_stop_rel_delta: Annotated[
        Optional[float], typer.Option(help="Relative improvement threshold in (0, 1): an epoch counts as progress "
                                           "only if val_loss_smooth < best * (1 - value); replaces "
                                           "early_stop_min_delta and is used by lr_plateau_match_early_stop "
                                           "(None = absolute, legacy)")
    ] = None
    lr_plateau_match_early_stop: Annotated[
        bool, typer.Option(help="ReduceLROnPlateau uses early stopping's improvement test (absolute "
                                "early_stop_min_delta on val_loss_smooth) with patience lr_plateau_patience, so the "
                                "LR halves before training stops (legacy: relative 1e-4, patience 9)")
    ] = False
    lr_plateau_patience: Annotated[
        int, typer.Option(help="lr_plateau_match_early_stop: epochs without improvement before the LR halves "
                               "(must be < early_stop_patience)")
    ] = 5
    schedule_free: Annotated[
        bool, typer.Option(help="Schedule-free RAdam (schedulefree.RAdamScheduleFree, pip install "
                                "'tomocpt[schedulefree]'): no LR scheduler, no SWA; validation and "
                                "checkpoints use the optimiser's averaged weights")
    ] = False
    weight_averaging: Annotated[
        str, typer.Option(help="Weight averaging: 'swa' (legacy Lightning SWA), 'ema' (EMA of the weights from step "
                               "0), 'swa_plateau' (SWA started when val_loss_smooth plateaus) or 'none'. EMA and "
                               "plateau SWA validate and save the averaged weights")
    ] = "swa"
    ema_decay: Annotated[float, typer.Option(help="Maximum EMA decay (weight_averaging=ema)")] = 0.999
    # Gate 8g: lr (and EMA decay) follow the effective batch (batch_size x devices); not CLI flags
    lr_scaling: Annotated[
        str, typer.Option(help="Scale lr with the effective batch: 'none', 'sqrt' (Adam-family rule) or 'linear' "
                               "(SGD rule); EMA decay becomes decay**k")
    ] = "none"
    lr_reference_batch: Annotated[
        int, typer.Option(help="Effective batch size at which lr was tuned (lr_scaling reference)")
    ] = 16
    snapshot_chunks: Annotated[
        int, typer.Option(help="Pipeline v2: chunks per class and split (train, val) predicted after every validation "
                               "and saved as volumes in <log_dir>/snapshots for scripts/train_dashboard.py; 0 = off")
    ] = 0
    swa_plateau_patience: Annotated[
        int, typer.Option(help="weight_averaging=swa_plateau: validations without a relative improvement before "
                               "averaging starts")
    ] = 5
    swa_plateau_rel_delta: Annotated[
        float, typer.Option(help="weight_averaging=swa_plateau: relative improvement of val_loss_smooth that counts")
    ] = 1e-3
    cv_all_folds: Annotated[
        bool, typer.Option(help="k-fold CV in one command (pipeline v2): train every fold of cv_folds, each in its own "
                                "process, from one shared chunk pool; then build the fold ensemble with out-of-fold "
                                "calibration and write the CV report to <model_dir>/<experiment_name>")
    ] = False
    cv_class_weights: Annotated[
        Optional[str], typer.Option(help="CV report only: class weights of the mean F-beta, e.g. "
                                         "'beta-amylase:0,thyroglobulin:2' (classes not named weigh 1)")
    ] = None
    ssl: Annotated[
        SSLConfig, typer.Option(help="Self-supervised pretraining settings (mode=selfSupervised)")
    ] = field(default_factory=SSLConfig)
    discriminative: Annotated[
        DiscriminativeConfig,
        typer.Option(help="Discriminative training of v2 pickers: contrastive loss, focal value term (Gate 7)"),
    ] = field(default_factory=DiscriminativeConfig)

    # Distillation parameters
    use_distillation: Annotated[
        bool,
        typer.Option(help="Enable knowledge distillation when fine-tuning from a checkpoint")
    ] = False

    distill_weight: Annotated[
        float,
        typer.Option(help="Blend factor between task loss and teacher-guided distillation (0-1)")
    ] = 0.5

    feature_distill_weight: Annotated[
        float,
        typer.Option(help="Relative weight for feature matching inside the distillation loss")
    ] = 1.0

    temperature: Annotated[
        float,
        typer.Option(help="Softmax temperature applied to teacher/student logits during distillation")
    ] = 2.0

    OVERFIT_N_BATCHES: Optional[int] = None
    N_GPUS: int = 4
    N_CPUS_IF_NO_GPU: int = 32
    USE_CUDA_FOR_DATA: bool = True

    FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS: float = 0.5
    COSINE_LR_SCHEDULE_N_EPOCHS: int = 6
    PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS: int = 6

    CHUNK_SIZE: int = 64
    CHUNK_STRIDE: int = 32
    RANDOM_FRACTION_TO_SAMPLE_TRAIN: float = -1.0  # Train on all the chunks
    SEED_FOR_TRAIN_VAL_SPLIT: int = 42
