"""Frozen copy of ``tomocpt.predict.helpers.peak_local_max_torch`` at the Gate 8a commit (e88922c),
the reference of G8e-C1 (identical peaks). Do not edit."""
import torch
import logging
logger = logging.getLogger(__name__)


def peak_local_max_torch(
    image: torch.Tensor,
    min_distance: int,
    threshold_abs: float,
    exclude_border: bool = True,
) -> torch.Tensor:
    """
    A corrected and vectorized PyTorch implementation of scikit-image's peak_local_max.
    Finds peaks in an image as coordinate list.
    """
    device = image.device
    max_pooled = (
        torch.nn.functional.max_pool3d(
            image.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
        )
        .squeeze(0)
        .squeeze(0)
    )
    local_maxima = image == max_pooled
    all_peaks = local_maxima & (image > threshold_abs)

    if exclude_border:
        requested_border = int(min_distance)
        for axis, dim in enumerate(image.shape):
            # Clamp per axis: a border of dim // 2 or more would mask the whole
            # volume along this axis and silently yield no peaks at all. This
            # happens routinely on binned tomograms, where the particle diameter
            # in pixels can exceed half the thickness in Z.
            border = min(requested_border, max(0, (dim - 1) // 2))
            if border < requested_border:
                logger.warning(
                    f"Border exclusion of {requested_border} px is too large for axis "
                    f"{axis} of size {dim}; reduced to {border} px. Peaks close to the "
                    "edge of this axis will be kept."
                )
            if border == 0:
                # A zero border must be skipped explicitly: `all_peaks[-0:]` is
                # the entire array, which would discard every peak.
                continue
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(None, border)
            upper[axis] = slice(-border, None)
            all_peaks[tuple(lower)] = False
            all_peaks[tuple(upper)] = False

    candidates = all_peaks.nonzero(as_tuple=False)
    if candidates.shape[0] == 0:
        return torch.empty((0, 3), device=device, dtype=torch.long)

    candidate_values = image[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    sorted_indices = torch.argsort(candidate_values, descending=True)
    candidates = candidates[sorted_indices]
    sorted_values = candidate_values[sorted_indices]

    # Every voxel of a saturated, flat-topped blob compares equal to its
    # max-pooled neighbourhood, so all of them become candidates. Taking an
    # arbitrary member puts the peak somewhere on the plateau instead of at its
    # centre, and sub-pixel refinement cannot recover it because a flat top has
    # no curvature. Group tied candidates so that each plateau collapses to its
    # centroid; ties are contiguous here because the candidates are value-sorted.
    tie_counts = torch.unique_consecutive(sorted_values, return_counts=True)[1]
    any_ties = bool((tie_counts > 1).any())
    if any_ties:
        tie_ends = torch.cumsum(tie_counts, dim=0)
        tie_starts = tie_ends - tie_counts
        tie_of_candidate = torch.repeat_interleave(
            torch.arange(tie_counts.shape[0], device=device), tie_counts
        )
        start_of_candidate = tie_starts[tie_of_candidate].tolist()
        end_of_candidate = tie_ends[tie_of_candidate].tolist()
        is_tied = (tie_counts > 1)[tie_of_candidate].tolist()

    suppress_grid = torch.zeros_like(image, dtype=torch.bool)
    final_peaks = []
    dist = min_distance
    z, y, x = torch.meshgrid(
        torch.arange(-dist, dist + 1, device=device),
        torch.arange(-dist, dist + 1, device=device),
        torch.arange(-dist, dist + 1, device=device),
        indexing="ij",
    )
    suppress_mask = (z**2 + y**2 + x**2) <= dist**2

    for i in range(candidates.shape[0]):
        coord = candidates[i]
        if suppress_grid[coord[0], coord[1], coord[2]]:
            continue
        if any_ties and is_tied[i]:
            # Replace this arbitrary plateau member with the centroid of the
            # tied voxels belonging to the same particle. Restricting to the
            # suppression radius keeps two equally saturated particles apart.
            tied = candidates[start_of_candidate[i] : end_of_candidate[i]]
            # Two passes: the arbitrary starting member can sit on the rim of the
            # plateau, where the inclusion sphere clips off the far side and
            # biases the centroid towards that rim.
            for _ in range(2):
                same_blob = ((tied - coord) ** 2).sum(dim=1) <= dist**2
                coord = tied[same_blob].float().mean(dim=0).round().long()
        final_peaks.append(coord)
        z_min, z_max = max(0, coord[0] - dist), min(image.shape[0], coord[0] + dist + 1)
        y_min, y_max = max(0, coord[1] - dist), min(image.shape[1], coord[1] + dist + 1)
        x_min, x_max = max(0, coord[2] - dist), min(image.shape[2], coord[2] + dist + 1)
        mask_z_min, mask_z_max = dist - (coord[0] - z_min), dist + (
            z_max - coord[0] - 1
        )
        mask_y_min, mask_y_max = dist - (coord[1] - y_min), dist + (
            y_max - coord[1] - 1
        )
        mask_x_min, mask_x_max = dist - (coord[2] - x_min), dist + (
            x_max - coord[2] - 1
        )
        suppress_grid[z_min:z_max, y_min:y_max, x_min:x_max] |= suppress_mask[
            mask_z_min : mask_z_max + 1,
            mask_y_min : mask_y_max + 1,
            mask_x_min : mask_x_max + 1,
        ]

    if not final_peaks:
        return torch.empty((0, 3), device=device, dtype=torch.long)
    return torch.stack(final_peaks)
