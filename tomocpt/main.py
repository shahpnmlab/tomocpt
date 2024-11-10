from typing import Annotated, TypeVar

import typer
from omegaconf import DictConfig

import tomocpt.mainConfig
from tomocpt.mainConfig import MainConfig
from tomocpt.configManager import create_configurable_app, update_dataclass_from_config
from tomocpt.training.train import train

from functools import wraps

def app_register_command(func):
    @app.command()
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def set_global_config_fn(config:DictConfig):
    update_dataclass_from_config(tomocpt.mainConfig.mainConfig, config)

app = create_configurable_app(MainConfig, set_global_config_fn=set_global_config_fn)

app_register_command(train)


if __name__ == "__main__":
    app.run()

    """
python -m tomocpt.main  --config-file /home/sanchezg/sideProjects/tomocpt/externalConfExamples/externalConf.yaml
python -m tomocpt.main  --train-experiment-name kk --config-file /home/sanchezg/sideProjects/tomocpt/externalConfExamples/externalConf.yam
    """