# subcommand_app.py
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List

from omegaconf import DictConfig, OmegaConf
import typer

from tomocpt.configManager.configManager import create_configurable_app

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32

@dataclass
class EvalConfig:
    metrics: List[str] = ('accuracy', 'f1')

@dataclass
class Config:
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()

# Create the app
app = create_configurable_app(Config)

@app.command(config_key=)
def train(
    model_name: Annotated[str, typer.Option(help="Model name")],
    data_path: Annotated[Path, typer.Option(help="Data path")],
    config: DictConfig = None
):
    """Train a model."""
    print(f"Training {model_name}")
    print(f"data_path {data_path}")
    print(f"Configuration: {OmegaConf.to_yaml(config)}")

@app.command(config_key=)
def evaluate(
    model_path: Annotated[Path, typer.Option(help="Model path")],
    config: DictConfig = None
):
    """Evaluate a model."""
    print(f"Evaluating {model_path}")
    print(f"Configuration: {OmegaConf.to_yaml(config)}")

    # register_config(config)

if __name__ == "__main__":
    app.run()