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
    particle_length_ang: Annotated[
        List[float], typer.Option(help="Particle longest dimension in angstroms, if spherical use diameter.")
    ] = field(default_factory=lambda: [MISSING])

    raw_data_dir: Annotated[Path, typer.Option(help="Path to where the tomograms are stored")] = field(default_factory=lambda: [MISSING])

    prepared_data_dir: Annotated[
        Path, typer.Option(help="Path to where the volume label pairs should be stored")
    ] = Path("/tmp/inputs")

    coordinate_file_type: Annotated[
        PrepareDataType, typer.Option(help="Path to where the volume label pairs should be stored")] = PrepareDataType.star

    desired_particle_pixel_size: Annotated[
        int, typer.Option(help="Resize the volume to dimensions that yield particle size in pixels")] = 10

    input_file:Annotated[
        List[Path], typer.Option(help="Path to the coordinate file. Only .star and .mod files are supported")
    ] = field(default_factory=lambda: [MISSING])

    class_id: Annotated[
        List[str | int], typer.Option(help="which class id do you want to make labels from?")] = field(default_factory=lambda: ["all"])

