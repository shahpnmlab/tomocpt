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
        typer.Option(help="Compute the Hessian loss term one class at a time with gradient checkpointing"),
    ] = True
