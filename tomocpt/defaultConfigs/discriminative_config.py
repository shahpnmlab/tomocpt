from dataclasses import dataclass
from typing import Annotated

import typer


@dataclass
class DiscriminativeConfig:
    """Discriminative training of pipeline-v2 pickers (Gate 7): supervised dense contrastive
    loss with hard negatives, and a focal value term. Both are off by default."""

    contrastive_weight: Annotated[
        float, typer.Option(help="Weight of the supervised dense contrastive loss (0 = off, no head created)")
    ] = 0.0
    embed_dim: Annotated[int, typer.Option(help="Size of the contrastive embedding")] = 64
    temperature: Annotated[float, typer.Option(help="Temperature of the contrastive loss")] = 0.1
    anchors_per_class: Annotated[int, typer.Option(help="Maximum anchors per class per batch")] = 32
    anchor_min_target: Annotated[
        float, typer.Option(help="Minimum target value of an anchor (peak voxel of a particle / filament ridge)")
    ] = 0.5
    bank_size: Annotated[int, typer.Option(help="Memory bank size per class (and for negatives)")] = 1024
    hard_negatives: Annotated[int, typer.Option(help="Hard negatives per batch, ranked by the model's score")] = 64
    hard_negative_min_score: Annotated[float, typer.Option(help="Lowest score of a hard negative")] = 0.3
    hard_negative_max_score: Annotated[
        float, typer.Option(help="Highest score of a hard negative (near-certain picks are often unlabelled particles)")
    ] = 0.9
    exclusion_radius: Annotated[
        float, typer.Option(help="No negatives within this many class radii of a labelled object")
    ] = 1.0
    mask_negatives: Annotated[int, typer.Option(help="Negatives per batch from prepData.negative_masks")] = 64
    init_from_ssl: Annotated[
        bool, typer.Option(help="Copy an SSL v2 checkpoint's dense head into the contrastive head when fine-tuning")
    ] = True
    value_loss: Annotated[
        str, typer.Option(help="Value term of the v2 picking loss: 'huber' (Gate 6) or 'focal' (soft-target focal penalty with the Huber weights on particle channels; filament channels keep Huber)")
    ] = "huber"
    focal_alpha: Annotated[float, typer.Option(help="Exponent gamma of |y - p| in the soft focal penalty")] = 2.0
    focal_beta: Annotated[float, typer.Option(help="Unused (beta of the CenterNet focal loss, kept for config compatibility)")] = 4.0
