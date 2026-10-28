"""Self-supervised pretraining settings (Gate 5).

``version = 1`` (default) is the legacy SSL (rotation, contrastive, inpainting on the
fixed-particle chunks), unchanged. ``version = 2`` is the wedge-aware, scale-aware SSL:
scale pairs of views, masked reconstruction of a clean view from a degraded one, global and
dense contrastive learning, and relative-scale prediction.

The ``None`` defaults depend on the version. They stay ``None`` in the stored config (so an
override of ``version`` alone gives a valid run) and are resolved by :meth:`SSLConfig.resolved`.
"""
import copy
from dataclasses import dataclass
from typing import Annotated, Optional

import typer

ROTATION_ILL_POSED = (
    "train.ssl.version=2 does not support use_rotation=True: with a missing wedge the 4-way "
    "rotation task about Z is ill-posed. Rotations by 90 and 270 degrees move the wedge from the "
    "kz-kx plane to the kz-ky plane while 0 and 180 degrees leave it in place, so the only reliable "
    "cue separates {0, 180} from {90, 270} (best accuracy ~50 %) and the head learns the wedge "
    "orientation (05_AGENT_SSL.md, background point 1)."
)


@dataclass
class SSLConfig:
    version: Annotated[int, typer.Option(help="Self-supervised pretraining: 1 = legacy, 2 = wedge- and scale-aware")] = 1
    use_rotation: Annotated[
        Optional[bool], typer.Option(help="4-way rotation task (None: True for v1, False for v2; v2 forbids it)")
    ] = None
    contrast_weight: Annotated[float, typer.Option(help="v2: weight of the global contrastive loss (0 = off)")] = 1.0
    dense_weight: Annotated[float, typer.Option(help="v2: weight of the dense contrastive loss (0 = off)")] = 1.0
    dense_dim: Annotated[int, typer.Option(help="v2: dense embedding size")] = 64
    dense_grad_scale: Annotated[
        float, typer.Option(help="v2: scale of the dense contrastive gradient into the network (0 = head only, 1 = full)")
    ] = 0.0
    n_anchors: Annotated[int, typer.Option(help="v2: anchor points per sample for the dense contrastive loss")] = 256
    recon_weight: Annotated[float, typer.Option(help="v2: weight of the masked reconstruction loss")] = 10.0
    recon_visible_weight: Annotated[
        Optional[float], typer.Option(help="Weight of the L1 on visible voxels (None: 1.0 for v1, 0.1 for v2)")
    ] = None
    scale_weight: Annotated[float, typer.Option(help="v2: weight of the relative-scale loss (0 = off)")] = 1.0
    scale_range_log2: Annotated[float, typer.Option(help="v2: view scale factors are log-uniform in [1, 2^this]")] = 1.0
    apix_ref_min: Annotated[float, typer.Option(help="v2: lower clip of the SSL reference voxel size (A)")] = 8.0
    apix_ref_max: Annotated[float, typer.Option(help="v2: upper clip of the SSL reference voxel size (A)")] = 16.0
    source_chunk: Annotated[int, typer.Option(help="v2: size of the SSL source chunks (stride = half of it)")] = 128
    use_mw_aug: Annotated[bool, typer.Option(help="v2: missing wedge in the degradation D")] = True
    use_fourier_aug: Annotated[bool, typer.Option(help="v2: amplitude-spectrum randomisation in D")] = True
    degradation_p: Annotated[float, typer.Option(help="v2: probability of each Fourier degradation in D")] = 0.8
    keep_angle_min: Annotated[float, typer.Option(help="v2: minimum kept wedge half-angle (deg) in D")] = 30.0
    keep_angle_max: Annotated[float, typer.Option(help="v2: maximum kept wedge half-angle (deg) in D")] = 60.0
    inversion_p: Annotated[float, typer.Option(help="v2: probability of contrast inversion of a view and its target")] = 0.5

    def resolved(self) -> "SSLConfig":
        """A copy with the version-dependent ``None`` defaults filled in.

        Raises ``ValueError`` for ``version = 2`` with ``use_rotation = True``."""
        version = int(self.version)
        if version not in (1, 2):
            raise ValueError(f"train.ssl.version must be 1 or 2, got {self.version}")
        out = copy.copy(self)
        out.version = version
        if out.use_rotation is None:
            out.use_rotation = version == 1
        if out.recon_visible_weight is None:
            out.recon_visible_weight = 1.0 if version == 1 else 0.1
        if version == 2 and out.use_rotation:
            raise ValueError(ROTATION_ILL_POSED)
        if version == 2 and not out.keep_angle_min <= out.keep_angle_max:
            raise ValueError("train.ssl.keep_angle_min must be <= keep_angle_max")
        return out
