from typing import Annotated

import hydra
import typer
from omegaconf import OmegaConf, DictConfig
from tomocpt.config import MainConfig
from configManager import create_configurable_app
from training.train import train, ModelTypes, TrainingModes

app = create_configurable_app(MainConfig)

# Annotated[str, typer.Option(help="Model name")]
TrainingModes
@app.command()
def train_cli(
          chunks_dir: Annotated[str, typer.Option(help="Model name")]=None,
          model_dir: Annotated[str, typer.Option(help="Model name")]=None,
          experimentName: Annotated[str, typer.Option(help="Model name")] = None,
          n_epochs: Annotated[int, typer.Option(help="Model name")] = None,
          trainingMode: Annotated[TrainingModes, typer.Option(help="Model name")] = "picking",
          learning_rate: Annotated[float, typer.Option(help="Model name")] = None,
          continueModelDir: Annotated[str, typer.Option(help="Model name")] = None,
          restoreFullStateWhenContinue: bool = True,
          compileModel: bool = False,
          batch_size: Annotated[int, typer.Option(help="Model name")] = None,
          use_cuda: bool = True,
          use_tensorboard: bool = True,
          config: DictConfig = None
          ):
    #TODO: Set the config as global
    config.train.n_epochs = n_epochs
    update_config_from_args()
    register_config(config)
    train(chunks_dir=chunks_dir, model_dir=model_dir, experimentName=experimentName)

if __name__ == "__main__":
    app.run()

    """

    """