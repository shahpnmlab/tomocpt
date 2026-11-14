"""Gate 8e.1, G8e-C2: GPU patch grid for predict (on by default on CUDA, no config field).

* ``tomocpt.predict.multiclass.grid_locations(shape, patch_size, overlap) -> np.ndarray [N, 6]``
  (``i0, j0, k0, i1, j1, k1``) equals TorchIO's ``GridSampler`` locations, in the same order.
* On CUDA, ``_predict_level`` slices the patches from the volume on the GPU; with the same batch
  size the model sees the same inputs, so the maps (and offsets) are bit-identical to the TorchIO
  path (``multiclass._GPU_PATCH_GRID = False``), with and without flip TTA.
"""
import numpy as np
import pytest
import torch
import torchio as tio

CUDA = torch.cuda.is_available()


@pytest.mark.parametrize("shape", [(64, 64, 64), (70, 81, 64), (65, 129, 200), (100, 64, 97)])
@pytest.mark.parametrize("overlap", [0, 16, 32, 30])  # TorchIO rejects odd overlaps (ruling)
def test_G8e_C2_locations_equal_torchio(shape, overlap):
    from tomocpt.predict.multiclass import grid_locations

    sampler = tio.GridSampler(tio.Subject(v=tio.ScalarImage(tensor=torch.zeros((1, *shape)))),
                              patch_size=64, patch_overlap=overlap)
    got = grid_locations(shape, 64, overlap)
    assert isinstance(got, np.ndarray) and got.shape[1] == 6
    assert np.array_equal(got, np.asarray(sampler.locations))


class _OffsetModel(torch.nn.Module):
    patch_size = 32

    def __init__(self, n_classes=3):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv3d(1, n_classes + 3, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.copy_(torch.randn(self.conv.bias.shape, generator=g))
        self.n = n_classes

    def predict_step(self, x, batch_idx, apix=None, classes=None):
        return torch.sigmoid(self.conv(x.float())[:, : self.n])

    def predict_step_with_offsets(self, x, apix=None, classes=None):
        y = self.conv(x.float())
        return torch.sigmoid(y[:, : self.n]), torch.tanh(y[:, self.n:])


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs a CUDA GPU")
@pytest.mark.parametrize("tta", [False, True])
@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("acc_dtype", ["float32", "float16"])
def test_G8e_C2_gpu_grid_maps_bit_identical(monkeypatch, tta, offsets, acc_dtype):
    from tomocpt.defaultConfigs.infer_config import InferConfig
    from tomocpt.predict import multiclass as mc

    dev = torch.device("cuda")
    model = _OffsetModel().to(dev).eval()
    vol = torch.randn((70, 81, 65), generator=torch.Generator().manual_seed(1))
    cfg = InferConfig()
    cfg.accumulate_device, cfg.accumulate_dtype, cfg.use_tta = "cuda", acc_dtype, tta
    cfg.use_cuda, cfg.predictions_batch_size = True, 3
    out = {}
    for gpu_grid in (False, True):
        monkeypatch.setattr(mc, "_GPU_PATCH_GRID", gpu_grid)
        out[gpu_grid] = mc._predict_level(model, vol, [0, 1, 2], 10.0, 8, cfg, dev, None, "t", with_offsets=offsets)
    if offsets:
        assert torch.equal(out[True][0], out[False][0]) and torch.equal(out[True][1], out[False][1])
    else:
        assert torch.equal(out[True], out[False])


def test_G8e_C2_gpu_grid_is_the_default():
    from tomocpt.predict import multiclass as mc

    assert mc._GPU_PATCH_GRID is True
