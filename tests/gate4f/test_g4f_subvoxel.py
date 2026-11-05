"""G4f-C6 (parabolic refinement) and G4f-C9 (predict with offsets)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from gate4 import g4_helpers as H4
from gate4b import g4b_helpers as HB
from gate4f import g4f_helpers as H
from utils.tiny import applied_main_config


@pytest.mark.parametrize("centre", [(20.3, 31.7, 25.45), (20.0, 31.0, 25.0), (19.6, 30.55, 24.9)])
def test_G4f_C6_parabolic_refinement(centre):
    from tomocpt.predict.multiclass import refine_parabolic

    m = H.gaussian_map((40, 64, 50), centre, sigma=2.0)
    v = np.round(centre).astype(int)
    peaks = np.array([[v[0], v[1], v[2], float(m[tuple(v)])]], float)
    out = refine_parabolic(torch.from_numpy(m), peaks)
    assert out.shape == (1, 4) and out[0, 3] == pytest.approx(peaks[0, 3])
    assert np.abs(out[0, :3] - np.array(centre)).max() < 0.1
    # a peak on the border gets no shift on that axis
    edge = np.array([[0, v[1], v[2], 0.5]], float)
    assert refine_parabolic(torch.from_numpy(m), edge)[0, 0] == 0


def _cond_classes():
    return ((1, "s100", 100.0), (2, "l500", 500.0))


@pytest.fixture(scope="module")
def offset_ckpt(tmp_path_factory):
    """A tiny v2 checkpoint with the offset head (unconditioned, softmax)."""
    from tomocpt.networks.pickingModel import BasePickingModel
    from gate3 import g3_helpers as G3

    root = tmp_path_factory.mktemp("g4f_off")
    cfg = G3.v2_config(root, _cond_classes(), cond=False)
    cfg.train.particles.class_head = "softmax"
    cfg.train.particles.offset_weight = 1.0
    items = [G3.v2_item(600 + i, [10.0, 50.0], apix=10.0) for i in range(2)]
    with applied_main_config(cfg):
        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        trainer = G3.cpu_trainer(root / "tb")
        trainer.fit(model, datamodule=G3.datamodule(items))
        path = root / "off.ckpt"
        trainer.save_checkpoint(path)
    return path


SHIFT = (0.3, -0.2, 0.4)  # (dz, dy, dx) in working voxels


def _constant_offsets(monkeypatch):
    """Every model's predict_step_with_offsets returns its own probabilities and a constant offset."""
    from tomocpt.networks.pickingModel import BasePickingModel

    def _with(self, x, apix=None, classes=None):
        probs = self.predict_step(x, 0, apix=apix, classes=classes)
        off = torch.tensor(SHIFT, dtype=probs.dtype, device=probs.device).view(1, 3, 1, 1, 1)
        return probs, off.expand(probs.shape[0], 3, *probs.shape[2:]).clone()

    monkeypatch.setattr(BasePickingModel, "predict_step_with_offsets", _with)


def _picks(tmp_path, monkeypatch, weights, refine, tomo):
    d, _ = H4.run_predict(tmp_path, monkeypatch, dict(weights=weights, tomogram_file=tomo, subvoxel_refine=refine,
                                                      confidence_threshold=0.3))
    return {n: HB.sort_rows(HB.class_picks(d, n)) for _, n, _ in _cond_classes()}


def test_G4f_C9_constant_offset_shifts_picks(tmp_path, monkeypatch, offset_ckpt):
    v = HB.blob_val_tomogram(tmp_path / "tomos", "TS_off", _cond_classes())
    with H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=False)):
        base = _picks(tmp_path / "a", monkeypatch, offset_ckpt, "none", v["path"])
        _constant_offsets(monkeypatch)
        shifted = _picks(tmp_path / "b", monkeypatch, offset_ckpt, "offset", v["path"])
    n = 0
    for name in base:
        assert base[name].shape == shifted[name].shape, name
        if len(base[name]):
            n += len(base[name])
            # star rows are x, y, z, score; the tomogram is predicted at its native voxel size
            want = base[name][:, :3] + np.array(SHIFT[::-1])
            np.testing.assert_allclose(HB.sort_rows(np.c_[want, base[name][:, 3]])[:, :3], shifted[name][:, :3],
                                       atol=1e-3)
    assert n > 0, "no picks (test is vacuous)"


def test_G4f_C9_offset_without_head_raises(tmp_path, monkeypatch):
    from gate4 import g4_helpers as H4b  # noqa: F401

    root = tmp_path / "plain"
    ck = H4.build_v2_ckpt(root, _cond_classes(), cond=False)["path"]
    v = HB.blob_val_tomogram(tmp_path / "tomos", "TS_plain", _cond_classes())
    with pytest.raises(Exception, match="(?i)offset"):
        with H4.oracle_models(monkeypatch, H4.neg_oracle_factory(conditioned=False)):
            _picks(tmp_path / "x", monkeypatch, ck, "offset", v["path"])


def test_G4f_C9_ensemble_averages_offsets(tmp_path, offset_ckpt):
    from tomocpt.predict.ensemble import load_ensemble

    with applied_main_config(None):
        ens = load_ensemble(HB.write_ensemble(tmp_path / "e.json", [offset_ckpt, offset_ckpt]))
        m0 = ens.members[0]
        with torch.no_grad():
            for p in m0.offset_head.parameters():
                p.add_(0.05)
        assert ens.has_offset_head()
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            pe, oe = ens.predict_step_with_offsets(x)
            outs = [m.predict_step_with_offsets(x) for m in ens.members]
    torch.testing.assert_close(oe.float(), torch.stack([o for _, o in outs]).float().mean(0), rtol=0, atol=1e-6)
    torch.testing.assert_close(pe.float(), torch.stack([p for p, _ in outs]).float().mean(0), rtol=0, atol=1e-6)
