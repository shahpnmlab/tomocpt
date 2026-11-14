"""On-the-fly crop sampling (Gate 8c).

``train.sampling = on_the_fly`` replaces the chunk files of pipeline v2 by one array per
tomogram and level (the "tomogram store", ``dataManager.tomostore``) and draws training crops
from it every epoch (``dataManager.otf``). These settings are read only in that mode. The
default ``train.sampling = auto`` picks this mode for pipeline v2 picking (human decision,
14 Nov 2026) and the chunk pipeline, unchanged, for pipeline v1 and SSL. Not CLI flags (YAML or
``key=value``).
"""

from dataclasses import dataclass
from typing import Annotated, Optional

import typer


@dataclass
class OnTheFlyConfig:
    volume_dtype: Annotated[
        str, typer.Option(help="dtype of the stored tomograms: 'float16' or 'float32' (targets are always float32)")
    ] = "float16"
    device: Annotated[
        str, typer.Option(help="Where tomograms live for the GPU data path: 'auto', 'cuda' or 'cpu'")
    ] = "auto"
    augment_device: Annotated[
        str, typer.Option(help="Where augmentation runs: 'auto' (cuda when training on a GPU and every enabled "
                               "augmentation has a GPU port, else cpu), 'cpu' (TorchIO in workers) or 'cuda'")
    ] = "auto"
    samples_per_epoch: Annotated[
        Optional[int], typer.Option(help="Training crops per epoch; None = expected number of grid chunks")
    ] = None
    jitter: Annotated[
        bool, typer.Option(help="Uniform sub-stride jitter of the crop corners")
    ] = True
    negative_mask_weight: Annotated[
        float, typer.Option(help="Extra sampling weight of crops with negative-mask voxels (0 = off)")
    ] = 0.0
    seed: Annotated[
        int, typer.Option(help="Seed of the training crop draws (with the epoch and the rank)")
    ] = 42
    val_seed: Annotated[
        int, typer.Option(help="Seed of the fixed validation crop set")
    ] = 1234
    val_samples: Annotated[
        Optional[int], typer.Option(help="Validation crops; None = expected number of grid chunks")
    ] = None
