from typing import Dict, Any, Optional, Tuple

import torch
from monai.transforms import MapTransform

from tomocpt.dataManager.preprocessing import preprocess_tomogram, process_label_to_match_tomogram


class PreprocessTomogramd(MapTransform):
    """
    A MONAI-compatible transform that ensures the label has the exact same
    spatial dimensions as the tomogram and correctly handles CPU tensor caching.
    """
    def __init__(self, tomo_key: str, label_key: str, particle_diameter_key: str, device: torch.device,
                 target_px: Optional[int] = None, chunk_size: Optional[int] = None):
        """``target_px`` (particle diameter in voxels after the rescale) and ``chunk_size``
        (minimum size per axis after padding) are passed by the caller. Chunking workers
        always pass them (T2.6): a Dask worker process has a fresh default ``mainConfig``.
        ``None`` falls back to the global config for other legacy callers."""
        super().__init__([tomo_key, label_key])
        self.tomo_key = tomo_key
        self.label_key = label_key
        self.particle_diameter_key = particle_diameter_key
        self.device = device
        if target_px is None or chunk_size is None:
            from tomocpt.mainConfig import mainConfig
            target_px = mainConfig.prepData.desired_particle_pixel_size if target_px is None else target_px
            chunk_size = mainConfig.train.CHUNK_SIZE if chunk_size is None else chunk_size
        self.target_px = target_px
        self.chunk_size = chunk_size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        diameter = d[self.particle_diameter_key]

        # This function now correctly returns a CPU tensor and its shape
        processed_tomo, final_tomo_shape, _ = preprocess_tomogram(
            mrc_path=d[self.tomo_key],
            particle_diameter_angst=diameter,
            target_particle_px=self.target_px,
            chunk_size=self.chunk_size,
            device=self.device,
            invert_contrast=False
        )
        d[self.tomo_key] = processed_tomo

        is_self_supervised = d[self.tomo_key] == d[self.label_key]
        if is_self_supervised:
            d[self.label_key] = processed_tomo.clone()
        else:
            # This function also correctly returns a CPU tensor
            processed_label = process_label_to_match_tomogram(
                mrc_path=d[self.label_key],
                final_tomogram_shape=final_tomo_shape,
                device=self.device,
            )
            d[self.label_key] = processed_label

        return d
