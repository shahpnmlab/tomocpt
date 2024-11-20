from dataclasses import dataclass, field
from functools import wraps
from typing import Annotated, List, Optional
import typer
from omegaconf import DictConfig

from tomocpt.configManager.configManager import create_app
from tomocpt.configManager.initializer import init_config
from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.defaultConfigs.network_config import NetworkConfig
from tomocpt.defaultConfigs.train_config import TrainConfig
from tomocpt.infer.infer import infer
from tomocpt.mainConfig import mainConfig
from tomocpt.training.train import train

app = create_app()


configForTrain = mainConfig.train
configForInference = mainConfig.infer

app.register_command(train, configForTrain)
app.register_command(infer, configForInference)
app.register_command(init_config, None)

if __name__ == "__main__":
    app.run()
    """

python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet --n_epochs 2 OVERFIT_N_BATCHES=10
python -m tomocpt.main train --network.model_type unet --chunks_dir data/refactor/chunks/ --model_dir /tmp/unet   --n_epochs 2 OVERFIT_N_BATCHES=10 network.KERNEL_SIZE=7
 

python -m tomocpt.main infer --tomosDir data/refactor/datasets/particle1/tomograms/ --predsDir /tmp/kk_inference --modelFname /tmp/unet/unnamed/checkpoints/weights.ckpt  --particleLengthAng 250

    """