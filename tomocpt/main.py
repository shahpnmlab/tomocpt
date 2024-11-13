from pathlib import Path
from typing import Annotated, Optional

import typer
from omegaconf import DictConfig

import tomocpt.mainConfig
from tomocpt.mainConfig import MainConfig
from tomocpt.configManager import create_configurable_app, update_dataclass_from_config

from tomocpt.training.train import train
from tomocpt.configManager.initializer import init
from functools import wraps


def app_register_command(func):
    @app.command()
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def set_global_config_fn(config: DictConfig):
    update_dataclass_from_config(tomocpt.mainConfig.mainConfig, config)


app = create_configurable_app(MainConfig, set_global_config_fn=set_global_config_fn)


@app.command()
def init_config(
        config: DictConfig,  # Add this parameter to accept the config
        output_path: Annotated[
            Optional[Path],
            typer.Option(
                "--output-path", "-o",
                help="Path where to save the config file"
            )
        ] = Path.cwd() / "config.yaml"
):
    """Initialize a new configuration file with default values."""
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    init(output_path)


# Register other commands
app_register_command(train)

if __name__ == "__main__":
    app.run()

    """
python -m tomocpt.main  --config-file /home/sanchezg/sideProjects/tomocpt/externalConfExamples/externalConf.yaml
python -m tomocpt.main  --train-experiment-name kk --config-file /home/sanchezg/sideProjects/tomocpt/externalConfExamples/externalConf.yam
python -m tomocpt.main train  --config-file /home/sanchezg/sideProjects/tomocpt/externalConfExamples/externalConf.yaml train.learning_rate=100 --config-merge-preference command
    """
