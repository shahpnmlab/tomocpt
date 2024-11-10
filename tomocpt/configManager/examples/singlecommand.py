# simple_app.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from omegaconf import DictConfig, OmegaConf
import typer

from tomocpt.configManager.configManager import create_configurable_app

@dataclass
class InnerConfig:
    inner1: int = -2

@dataclass
class Config:
    value1: int = 42
    value2: str = "default"
    inner: InnerConfig = field(default_factory=InnerConfig)

# Create the app
config_app = create_configurable_app(Config)

def main(
    input_path: Annotated[Path, typer.Option(help="Input path")],
    verbose: Annotated[bool, typer.Option(help="Verbose output")] = False,
    config: DictConfig = None
):
    """Simple command with configuration."""
    print(f"Processing {input_path}")
    print(f"Verbose {verbose}")
    print(f">>Configuration:\n{OmegaConf.to_yaml(config)}")

if __name__ == "__main__":
    config_app.run_with_config(main)

    """
PYTHONPATH=/home/sanchezg/sideProjects/tomocpt python scratch_28.py --input-path kk.path --verbose value1=-1  value2=-2  inner.inner1=-999
    """