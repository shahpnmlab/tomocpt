"""Scale policy of data pipeline v2 (Gate 2).

Pipeline v2 resamples a tomogram only as far as needed to keep every class inside a band
of voxel diameters (CLAUDE.md Section 3). These settings are used by pipeline v2 only;
pipeline v1 (``train.data_pipeline_version = 1``) ignores them.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import typer


class ResampleMethod(str, Enum):
    trilinear = "trilinear"
    fourier = "fourier"


@dataclass
class ScaleConfig:
    band_min_vox: Annotated[
        float, typer.Option(help="Lower edge of the band of class diameters, in working voxels (pipeline v2)")
    ] = 10.0
    band_max_vox: Annotated[
        float, typer.Option(help="Upper edge of the band of class diameters, in working voxels (pipeline v2)")
    ] = 32.0
    scales_per_level: Annotated[
        int,
        typer.Option(help="Working voxel sizes per level: 1 = the level's size only; k > 1 adds k-1 jittered sizes"),
    ] = 1
    resample: Annotated[
        ResampleMethod,
        typer.Option(help="Resampling method of pipeline v2 (fourier = Fourier cropping, anti-aliased)"),
    ] = ResampleMethod.fourier
    zoom_aug_max: Annotated[
        float, typer.Option(help="Maximum zoom factor of the zoom augmentation (1.0 = off; suggestion 1.25)")
    ] = 1.0
    target_storage: Annotated[
        str, typer.Option(help="How chunk targets are stored: 'points' (default) or 'dense' (debugging)")
    ] = "points"
