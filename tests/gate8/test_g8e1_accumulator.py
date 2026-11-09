"""Gate 8e.1, accumulator device (8a finding 1; human decision 9 Nov 2026: pulled forward before 8b).

``infer.accumulate_device=auto`` must count the caching allocator's reserved-but-unallocated memory
as free, so tomogram 2 of a run gets the same decision as tomogram 1. Maps accumulated on the CPU
and on the GPU agree within ``atol 1e-6`` (E1).
"""
import pytest
import torch

from tomocpt.predict import multiclass as mc

CUDA = torch.cuda.is_available()


def _fake_cuda(monkeypatch, free, reserved, allocated):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (int(free), int(80e9)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: int(reserved))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: int(allocated))


def test_G8e1_auto_counts_cached_memory_as_free(monkeypatch):
    dev = torch.device("cuda", 0)
    shape, n_maps = (200, 400, 400), 6
    needed = 200 * 400 * 400 * (n_maps * 4 + 4)  # ~0.9 GB
    # Driver sees little free memory, but the allocator caches 20 GB it can reuse.
    _fake_cuda(monkeypatch, free=needed, reserved=20e9 + 1e9, allocated=1e9)
    assert mc._accumulator_device("auto", dev, n_maps, shape, torch.float32) == dev
    # Nothing cached and too little free: CPU, as before.
    _fake_cuda(monkeypatch, free=needed, reserved=1e9, allocated=1e9)
    assert mc._accumulator_device("auto", dev, n_maps, shape, torch.float32) == torch.device("cpu")
    # Plenty free: GPU, as before.
    _fake_cuda(monkeypatch, free=10 * needed, reserved=0, allocated=0)
    assert mc._accumulator_device("auto", dev, n_maps, shape, torch.float32) == dev


def test_G8e1_explicit_settings_unchanged():
    cpu = torch.device("cpu")
    assert mc._accumulator_device("cpu", torch.device("cuda", 0), 2, (8, 8, 8), torch.float32) == cpu
    assert mc._accumulator_device("auto", cpu, 2, (8, 8, 8), torch.float32) == cpu
    assert mc._accumulator_device("cuda", cpu, 2, (8, 8, 8), torch.float32) == cpu
    assert mc._accumulator_device("cuda", torch.device("cuda", 0), 2, (8, 8, 8), torch.float32) == torch.device("cuda", 0)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
def test_G8e1_same_decision_after_cache_fill():
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.empty_cache()
    free0, _ = torch.cuda.mem_get_info(dev)
    # Maps needing ~30 % of the free memory (below the 0.5 fraction).
    n_vox = int(0.3 * free0 / (6 * 4 + 4))
    shape = (n_vox // (256 * 256), 256, 256)
    first = mc._accumulator_device("auto", dev, 6, shape, torch.float32)
    x = torch.empty(int(0.6 * free0) // 4, dtype=torch.float32, device=dev)  # tomogram 1's buffers
    del x  # cached by the allocator, not returned to the driver
    second = mc._accumulator_device("auto", dev, 6, shape, torch.float32)
    torch.cuda.empty_cache()
    assert first == dev
    assert second == first


class _ConvModel(torch.nn.Module):
    """Deterministic stand-in for ``InferenceModel`` (``patch_size``, ``predict_step``)."""

    patch_size = 32

    def __init__(self, n_classes=3):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv3d(1, n_classes, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.copy_(torch.randn(self.conv.bias.shape, generator=g))

    def predict_step(self, x, batch_idx, apix=None, classes=None):
        return torch.sigmoid(self.conv(x.float()))


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("tta", [False, True])
def test_G8e1_cpu_vs_gpu_accumulation_maps(tta):
    from tomocpt.defaultConfigs.infer_config import InferConfig

    dev = torch.device("cuda")
    model = _ConvModel().to(dev).eval()
    g = torch.Generator().manual_seed(1)
    vol = torch.randn((70, 81, 64), generator=g)
    out = {}
    for setting in ("cpu", "cuda"):
        cfg = InferConfig()
        cfg.accumulate_device = setting
        cfg.accumulate_dtype = "float32"
        cfg.use_tta = tta
        cfg.use_cuda = False  # no autocast: compare accumulation only
        cfg.predictions_batch_size = 2
        out[setting] = mc._predict_level(model, vol, [0, 1, 2], 10.0, 8, cfg, dev, None, "t")
    assert out["cpu"].shape == (3, 70, 81, 64)
    torch.testing.assert_close(out["cuda"], out["cpu"], atol=1e-6, rtol=0)
