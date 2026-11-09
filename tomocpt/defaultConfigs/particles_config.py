from dataclasses import dataclass
from typing import Annotated, Optional

import typer


@dataclass
class ParticlesConfig:
    """Multi-class picking settings (pipeline v2, Gate 3)."""

    class_weights: Annotated[
        Optional[str],
        typer.Option(help="Per-class loss weights by class number, e.g. '1:1.0,2:2.0' (missing classes weigh 1.0)"),
    ] = None
    loss_checkpoint: Annotated[
        bool,
        typer.Option(help="Gradient checkpointing in the Hessian loss term (saves memory, slower)"),
    ] = False
    cross_class_neg_weight: Annotated[
        float,
        typer.Option(help="Extra negative weight (x positive weight) for voxels inside another class's particle; 0 = off"),
    ] = 0.0
    class_head: Annotated[
        str,
        typer.Option(help="Class head of pipeline-v2 pickers: 'sigmoid' (independent classes) or 'softmax' "
                          "(classes and an implicit background compete per voxel)"),
    ] = "sigmoid"
    core_weight: Annotated[
        float,
        typer.Option(help="Centroid weight (Gate 4f): particle-channel value weight x (1 + core_weight * y^core_power), "
                          "so the few core voxels of the target count; 0 = off"),
    ] = 0.0
    core_power: Annotated[float, typer.Option(help="Exponent p of the centroid weight")] = 4.0
    offset_weight: Annotated[
        float,
        typer.Option(help="Offset head (Gate 4f): weight of the L1 loss on the predicted sub-voxel offset to the "
                          "particle centre; 0 = no head"),
    ] = 0.0
