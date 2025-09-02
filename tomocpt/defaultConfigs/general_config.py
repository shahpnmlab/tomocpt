import typer
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, Optional


@dataclass
class GlobalPropertyConfig:
    training_data_dir: Annotated[
        Optional[Path], typer.Option(help="Path to where the volume label pairs should be stored")
    ] = None
    chunks_dir: Annotated[Optional[Path], typer.Option(help="The directory with chunks")] = None