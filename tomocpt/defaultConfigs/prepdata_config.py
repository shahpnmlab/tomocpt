from dataclasses import dataclass, field
from typing import Annotated, List, Optional
from omegaconf import MISSING
from pathlib import Path
import typer
from enum import Enum


class PrepareDataType(str, Enum):
    star = "star"
    imod = "imod"


class LabelBorderPolicy(str, Enum):
    skip = "skip"
    clip = "clip"


@dataclass
class PrepdataConfig:
    raw_data_dir: Annotated[
        str, typer.Option(help="Comma-separated paths to tomogram directories")
    ] = MISSING

    training_data_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to where the volume label pairs should be stored for subsequent training step"
        ),
    ] = MISSING

    

    particle_length_ang: Annotated[
        str, typer.Option(help="Comma-separated list of particle lengths in angstroms")
    ] = MISSING

    coordinate_files: Annotated[
        str, typer.Option(help="Comma-separated paths to coordinate files")
    ] = MISSING

    coordinate_file_type: Annotated[
        PrepareDataType, typer.Option(help="Coordinate file type (star or imod)?")
    ] = PrepareDataType.star

    class_id: Annotated[str, typer.Option(help="Comma-separated list of class IDs")] = (
        "all"
    )

    desired_particle_pixel_size: Annotated[
        int,
        typer.Option(
            help="Resize the volume so that the particle diameter is this many pixels"
        ),
    ] = 20

    apix_override: Annotated[
        Optional[float],
        typer.Option(help="Voxel size in A to use instead of the MRC header value (all tomograms)"),
    ] = None

    label_border_policy: Annotated[
        LabelBorderPolicy,
        typer.Option(
            help="Particles whose box touches the tomogram border: 'skip' (legacy, become negatives) "
                 "or 'clip' (draw the in-bounds part)"
        ),
    ] = LabelBorderPolicy.skip

    multiclass: Annotated[
        bool,
        typer.Option(
            help="One dataset holds every species: STAR files with rlnClassNumber, diameters from "
                 "class_diameters (pipeline v2)"
        ),
    ] = False

    class_diameters: Annotated[
        Optional[str],
        typer.Option(help="Multi-class diameters in A, per class number, e.g. '1:120,2:130'"),
    ] = None

    class_names: Annotated[
        Optional[str],
        typer.Option(help="Multi-class names per class number, e.g. '1:apo-ferritin,2:ribosome'"),
    ] = None

    class_annotated: Annotated[
        Optional[str],
        typer.Option(
            help="Classes annotated exhaustively (numbers or names, comma-separated); "
                 "default: every class present"
        ),
    ] = None

    USE_CUDA_FOR_DATA: bool = True
    ALPHA_FOR_DROPPING_EMPTY_CUBES: float = 1

    def parse_lists(self):
        """Convert comma-separated strings to lists"""
        particle_length_ang = [
            float(x.strip()) for x in self.particle_length_ang.split(",")
        ]
        raw_data_dir = [Path(x.strip()) for x in self.raw_data_dir.split(",")]
        coordinate_files = [Path(x.strip()) for x in self.coordinate_files.split(",")]
        class_id = [x.strip() for x in self.class_id.split(",")]

        return particle_length_ang, raw_data_dir, coordinate_files, class_id
