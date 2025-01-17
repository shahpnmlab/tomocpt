from dataclasses import dataclass, field
from typing import Annotated, List
from omegaconf import MISSING
from pathlib import Path
import typer
from enum import Enum


class PrepareDataType(str, Enum):
    star = "star"
    imod = "imod"


@dataclass
class PrepdataConfig:
    raw_data_dir: Annotated[
        str, typer.Option(help="Comma-separated paths to tomogram directories")
    ] = MISSING

    prepared_data_dir: Annotated[
        Path, typer.Option(help="Path to where the volume label pairs should be stored")
    ] = Path("/tmp/inputs")

    particle_length_ang: Annotated[
        str, typer.Option(help="Comma-separated list of particle lengths in angstroms")
    ] = MISSING

    coordinate_files: Annotated[
        str, typer.Option(help="Comma-separated paths to coordinate files")
    ] = MISSING

    coordinate_file_type: Annotated[
        PrepareDataType, typer.Option(help="Coordinate file type (star or imod)?")
    ] = PrepareDataType.star

    class_id: Annotated[
        str, typer.Option(help="Comma-separated list of class IDs")
    ] = "all"

    desired_particle_pixel_size: Annotated[
        int, typer.Option(help="Resize the volume to dimensions that yield particle radius in pixels")
    ] = 20

    USE_CUDA_FOR_DATA: bool = True
    ALPHA_FOR_DROPPING_EMPTY_CUBES: float = 1

    def parse_lists(self):
        """Convert comma-separated strings to lists"""
        self.particle_length_ang = [float(x.strip()) for x in self.particle_length_ang.split(',')]
        self.raw_data_dir = [Path(x.strip()) for x in self.raw_data_dir.split(',')]
        self.coordinate_files = [Path(x.strip()) for x in self.coordinate_files.split(',')]
        self.class_id = [x.strip() for x in self.class_id.split(',')]
