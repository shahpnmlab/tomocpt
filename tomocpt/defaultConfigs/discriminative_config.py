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
    # Gate 9
    contrastive_warmup_steps: Annotated[
        int, typer.Option(help="The contrastive weight ramps linearly from 0 to contrastive_weight over the first "
                               "N optimizer steps (0 = constant weight)")
    ] = 0
    bank_max_age: Annotated[
        int, typer.Option(help="Memory-bank entries older than this many steps are not used (0 = no limit)")
    ] = 0
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
    # Gate 9b: the class head uses the contrastive embedding (09b_AGENT_EMBED_HEAD.md)
    embed_ce_weight: Annotated[
        float, typer.Option(help="Weight of the cosine-softmax cross-entropy on the projection embedding at the "
                                 "contrastive anchors, over the particle classes and background (0 = off)")
    ] = 0.0
    prototype_head: Annotated[
        bool, typer.Option(help="Particle logits get a residual alpha * s * (cos(e, p_k) - cos(e, p_bg)) from the "
                                "projection embedding (alpha learned, starts at 0)")
    ] = False
    prototype_scale: Annotated[
        float, typer.Option(help="Scale s of the cosine prototype logits (embedding CE and prototype head)")
    ] = 10.0
