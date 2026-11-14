"""Gate 8e.1, G8e-C1: ``peak_local_max_torch`` returns identical peaks, in the same order, to the
version at the 8a commit (``legacy_peaks.py``) on random maps with plateaus, ties, borders and
thousands of candidates, on the CPU and (slow) on the GPU."""
import pytest
import torch

from gate8.legacy_peaks import peak_local_max_torch as legacy

CUDA = torch.cuda.is_available()


def _map(seed, shape=(40, 70, 64), n_blobs=60, plateaus=True, noise=0.05):
    g = torch.Generator().manual_seed(seed)
    D, H, W = shape
    zz, yy, xx = torch.meshgrid(*(torch.arange(s, dtype=torch.float32) for s in shape), indexing="ij")
    img = noise * torch.rand(shape, generator=g)
    for i in range(n_blobs):
        c = torch.rand(3, generator=g) * torch.tensor([D, H, W], dtype=torch.float32)
        s = 1.0 + 3.0 * float(torch.rand(1, generator=g))
        blob = torch.exp(-((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (2 * s * s))
        if plateaus and i % 3 == 0:
            blob = blob.clamp(max=0.6) / 0.6  # flat-topped (saturated) blob: tied candidates
        img = torch.maximum(img, blob)
    if plateaus:
        img = (img * 50).round() / 50  # quantised: many exact ties
    return img


CASES = [
    dict(seed=0, min_distance=3, threshold_abs=0.1),
    dict(seed=1, min_distance=5, threshold_abs=0.02),           # thousands of candidates (noise)
    dict(seed=2, min_distance=2, threshold_abs=0.5, exclude_border=False),
    dict(seed=3, min_distance=30, threshold_abs=0.2),           # border clamp on the short axis
    dict(seed=4, min_distance=4, threshold_abs=0.9),
    dict(seed=5, min_distance=1, threshold_abs=0.0),
    dict(seed=6, min_distance=6, threshold_abs=2.0),            # no candidates
]


def _run(fn, img, case):
    kw = {k: v for k, v in case.items() if k != "seed"}
    return fn(img, **kw)


@pytest.mark.parametrize("case", CASES, ids=[str(c["seed"]) for c in CASES])
def test_G8e_C1_peaks_identical_cpu(case):
    from tomocpt.predict.helpers import peak_local_max_torch

    img = _map(case["seed"])
    ref = _run(legacy, img, case)
    got = _run(peak_local_max_torch, img, case)
    assert got.device == ref.device and got.dtype == ref.dtype
    assert torch.equal(got, ref), (got.shape, ref.shape)
    if case["seed"] == 1:
        assert ref.shape[0] > 200


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("case", CASES, ids=[str(c["seed"]) for c in CASES])
def test_G8e_C1_peaks_identical_gpu(case):
    from tomocpt.predict.helpers import peak_local_max_torch

    img = _map(case["seed"], shape=(60, 120, 110), n_blobs=200).cuda()
    ref = _run(legacy, img, case)
    got = _run(peak_local_max_torch, img, case)
    assert got.device == ref.device and got.dtype == ref.dtype
    assert torch.equal(got, ref), (got.shape, ref.shape)
