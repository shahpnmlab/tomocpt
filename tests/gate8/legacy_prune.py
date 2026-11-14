"""Frozen copy of ``tomocpt.predict.helpers._prune_refined_peaks_gpu`` at the Gate 8a commit
(e88922c), reference of the 8e.1 pruning test. Do not edit."""
from typing import Tuple

import torch


def _prune_refined_peaks_gpu(
    coords: torch.Tensor, scores: torch.Tensor, min_distance: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prunes refined, sub-pixel coordinates to enforce a minimum distance after refinement.

    Iterates through coordinates sorted by score, keeping the highest-scoring peak
    and removing any other lower-scoring peaks within the specified minimum distance.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 3) with sub-pixel coordinates.
        scores (torch.Tensor): Tensor of shape (N,) with corresponding peak scores.
        min_distance (float): The minimum distance to enforce between peaks in pixels.

    Returns:
        A tuple containing:
        - torch.Tensor: The pruned coordinates.
        - torch.Tensor: The scores of the pruned coordinates.
    """
    if coords.shape[0] < 2:
        return coords, scores

    device = coords.device
    min_distance_sq = min_distance**2

    # Sort peaks by score in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_coords = coords[sorted_indices]
    sorted_scores = scores[sorted_indices]

    # Mask to track which peaks to keep
    keep_mask = torch.ones(sorted_coords.shape[0], dtype=torch.bool, device=device)

    for i in range(sorted_coords.shape[0]):
        # If this peak has already been suppressed by a higher-scoring one, skip it
        if not keep_mask[i]:
            continue

        # This is a good peak. Now, suppress any subsequent, lower-scoring peaks that are too close.
        # Calculate squared distances to all subsequent peaks.
        distances_sq = torch.sum(
            (sorted_coords[i + 1 :] - sorted_coords[i]) ** 2, dim=1
        )

        # Find which of the *subsequent* peaks are too close
        too_close_mask = distances_sq < min_distance_sq

        # Update the 'keep_mask' for the subsequent peaks.
        # A peak is marked as False if it's too close to the current 'good' peak.
        if too_close_mask.any():
            keep_mask[i + 1 :][too_close_mask] = False

    final_coords = sorted_coords[keep_mask]
    final_scores = sorted_scores[keep_mask]

    return final_coords, final_scores


