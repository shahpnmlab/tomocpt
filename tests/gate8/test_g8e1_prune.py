"""Gate 8e.1: ``_prune_refined_peaks_gpu`` keeps exactly the same peaks, in the same order, as the 8a
version (``legacy_prune.py``): random sub-voxel clusters, pairs placed at the distance boundary,
equal scores, N up to several thousand; CPU and (slow) GPU."""
import pytest
import torch

from gate8.legacy_prune import _prune_refined_peaks_gpu as legacy

CUDA = torch.cuda.is_available()


def _peaks(seed, n=3000, extent=200.0, min_distance=6.5):
    g = torch.Generator().manual_seed(seed)
    centres = torch.rand(n // 3, 3, generator=g) * extent
    jitter = torch.randn(n // 3, 2, 3, generator=g) * min_distance * 0.6  # clusters to prune
    coords = torch.cat([centres, (centres[:, None] + jitter).reshape(-1, 3)])
    # pairs exactly at / just inside / just outside the distance
    base = torch.rand(50, 3, generator=g) * extent
    d = torch.tensor([min_distance, min_distance * (1 - 1e-7), min_distance * (1 + 1e-7)])
    for k in range(3):
        coords = torch.cat([coords, base, base + torch.tensor([d[k], 0.0, 0.0])])
    scores = (torch.rand(coords.shape[0], generator=g) * 20).round() / 20  # many equal scores
    return coords.float(), scores.float(), min_distance


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [1, 2, 50, 3000])
def test_G8e_C1_prune_identical_cpu(seed, n):
    from tomocpt.predict.helpers import _prune_refined_peaks_gpu

    c, s, md = _peaks(seed, n=max(n, 3))
    c, s = c[:n], s[:n]
    rc, rs = legacy(c, s, md)
    gc, gs = _prune_refined_peaks_gpu(c, s, md)
    assert torch.equal(gc, rc) and torch.equal(gs, rs)
    assert gc.device == rc.device


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_G8e_C1_prune_identical_gpu(seed):
    from tomocpt.predict.helpers import _prune_refined_peaks_gpu

    c, s, md = _peaks(seed, n=6000)
    c, s = c.cuda(), s.cuda()
    rc, rs = legacy(c, s, md)
    gc, gs = _prune_refined_peaks_gpu(c, s, md)
    assert torch.equal(gc, rc) and torch.equal(gs, rs)
    assert gc.device == rc.device
