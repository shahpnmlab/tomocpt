from dataclasses import dataclass
from typing import Annotated
from omegaconf import MISSING
from pathlib import Path
import typer
from enum import Enum


class PrepareDataType(str, Enum):
    star = "star"
    imod = "imod"

@dataclass
class PrepdataConfig:
    particle_length_ang: Annotated[
        float, typer.Option(help="Particle longest dimension in angstroms, if spherical use diameter.")] = MISSING
    raw_data_dir: Annotated[Path, typer.Option(help="Path to where the tomograms are stored")] = MISSING
    prepared_data_dir: Annotated[
        Path, typer.Option(help="Path to where the volume label pairs should be stored")] = Path("/tmp/inputs")
    coordinate_file_type: Annotated[
        PrepareDataType, typer.Option(help="Path to where the volume label pairs should be stored")] = PrepareDataType.star
    desired_particle_pixel_size: Annotated[
        int, typer.Option(help="Resize the volume to dimensions that yield particle size in pixels")] = 10
    input_file:Annotated[
        Path, typer.Option(help="Path to the coordinate file. Only .star and .mod files are supported")] = MISSING
    class_id: Annotated[
        str, typer.Option(help="which class id do you want to make labels from?")] = "all"