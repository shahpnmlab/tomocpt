
from dataclasses import dataclass, asdict



# ------------------ADVANCED KNOBS --------------------------- #

@dataclass
class NetworkConfig:
    CHUNK_SIZE: int = 64
    CHUNK_STRIDE: int = 32
    RANDOM_FRACTION_TO_SAMPLE_TRAIN: float = -1.  # Train on all the chunks

    MODEL_TYPE: str = "swinunetr" # or unet

    LEARNING_RATE:float = 4e-4
    WEIGHT_DECAY:float = 1e-8

    #### THIS IS CONFIG FOR U-NET
    IN_CHANNELS: int = 1
    FIRST_LAYER_OUT_CHANNELS: int = 32
    NUM_LEVELS: int = 5
    CHANNELS_INCREASE_FACTOR: int = 2
    KERNEL_SIZE: int = 5
    CONV_DILATION: int = 1
    LAST_ACTIVATION_LAYER: str = "linear" #or "sigmoid"
    STRIDE_CONV_INSTEAD_OF_POOLING: bool = False
    OUTPUT_DIM: str = "same"
    FACTOR_REDUCE_LR_PLATEAU_N_EPOCHS:float = 0.5
    COSINE_LR_SCHEDULE_N_EPOCHS: int = 6
    PATIENT_REDUCE_LR_PLATEAU_N_EPOCHS: int = 6

    ##### THIS IS CONFIG FOR SWINUNETR
    SWINUNETR_FEAT_SIZE: int = 12*3  # Should be multiple of 12
    DROP_RATE:float = 0.3
    ATTN_DROP_RATE:float = 0.3
    DROPOUT_PATH_RATE:float = 0.2

    # THIS IS FOR PRETRAIN
    NOISE2NOISE_RAND_MASK_SIZE: int = 15
    CONTRAST_LOSS_L1_EMB_REGULARIZATION:float = 1e-4
    CONTRAST_LOSS_TEMPERATURE: float = 1e-1

    #BCE Loss val
    BCE_EPS: int = 1 #TODO: Not used, should it be removed?

    #FASTER TRAINING ARGS
    LIMIT_VALIDATION_BATCHES: int = 1
    VAL_CHECK_INTERVAL: int = 1

    TORCH_MATMUL_PRECISION: str = "medium" # "medium" "high" "highest"
    TORCH_FLOAT_PRECISION: str = '32' # '32' # '16' 'bf16'

    SEED_FOR_TRAIN_VAL_SPLIT: int = 113

if __name__ == "__main__":
    import yaml
    print(yaml.dump(asdict(NetworkConfig())))