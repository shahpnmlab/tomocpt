"""G4f-C7 (offset targets) and G4f-C8 (offset head)."""

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as G3
from gate4f import g4f_helpers as H
from utils.tiny import applied_main_config


def _render(points, C=2, shape=(64, 64, 64)):
    from tomocpt.labels.helpers import render_targets

    return torch.from_numpy(render_targets(np.asarray(points, float).reshape(-1, 5), shape, C, "clip"))


def test_G4f_C7_offset_targets_point_to_the_exact_centres():
    from tomocpt.training.offsets import offset_targets

    pts = [(20.3, 30.7, 25.45, 0, 12.0), (44.6, 18.2, 40.9, 1, 24.0), (40.1, 47.75, 15.3, 0, 12.0)]
    y = _render(pts).unsqueeze(0)
    d = torch.tensor([[12.0, 24.0]])
    avail = torch.tensor([[True, True]])
    tgt, mask = offset_targets(y, d, avail)
    assert tgt.shape == (1, 3, 64, 64, 64) and mask.shape == (1, 1, 64, 64, 64) and mask.dtype == torch.bool
    zz, yy, xx = torch.meshgrid(*[torch.arange(64, dtype=torch.float32)] * 3, indexing="ij")
    n_core = 0
    for z, yv, x, c, dv in pts:
        core = (y[0, int(c)] > 0.5)
        near = ((zz - z) ** 2 + (yy - yv) ** 2 + (xx - x) ** 2).sqrt() < dv / 2
        sel = core & near
        assert sel.any()
        assert mask[0, 0][sel].all()
        n_core += int(sel.sum())
        for i, comp in enumerate((z - zz, yv - yy, x - xx)):
            err = (tgt[0, i][sel] - comp[sel]).abs().max()
            assert float(err) < 0.05, (c, i, float(err))
    assert int(mask.sum()) == n_core
    # no particle: empty mask
    tgt0, mask0 = offset_targets(torch.zeros(1, 2, 64, 64, 64), d, avail)
    assert not mask0.any()
    # an unavailable class gives no target
    _, m_un = offset_targets(y, d, torch.tensor([[False, True]]))
    assert int(m_un.sum()) < n_core and m_un.any()


def _model(offset_weight):
    from tomocpt.networks.pickingModel import BasePickingModel

    torch.manual_seed(0)
    return BasePickingModel(lr=1e-3, train_continue=None)


def test_G4f_C8_off_creates_nothing(tmp_path):
    with applied_main_config(H.v2_cfg(tmp_path)):
        m = _model(0.0)
    assert m.offset_head is None and not m.has_offset_head()
    assert not any(k.startswith("offset_head") for k in m.state_dict())


def test_G4f_C8_head_zero_init_gradients_and_round_trip(tmp_path):
    from tomocpt.networks.inference import InferenceModel

    with applied_main_config(H.v2_cfg(tmp_path, **{"particles.offset_weight": 1.0})):
        m = _model(1.0)
        assert m.has_offset_head()
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            logits_ref = m(x)
            probs, off = m.predict_step_with_offsets(x)
        assert off.shape == (1, 3, 64, 64, 64) and torch.count_nonzero(off) == 0
        torch.testing.assert_close(probs, m.predict_step(x), rtol=0, atol=0)

        for empty in (False, True):
            its = H.items(2)
            if empty:
                for it in its:
                    it["target_data"] = torch.zeros_like(it["target_data"])
            loss, grads, _ = G3.first_step_loss_and_grads(m, its, tmp_path / f"tb{empty}")
            head = {n: g for n, g in grads.items() if n.startswith("offset_head.")}
            assert head and all(g is not None for g in head.values()), empty
            assert torch.isfinite(torch.tensor(loss))
        trainer = G3.cpu_trainer(tmp_path / "tb_fit")
        trainer.fit(m, datamodule=G3.datamodule(H.items(2)))
        path = tmp_path / "off.ckpt"
        trainer.save_checkpoint(path)
        with torch.no_grad():
            assert torch.equal(m(x), m(x))
    with applied_main_config(None):
        loaded = InferenceModel.load_from_checkpoint(str(path), map_location="cpu", strict=True)
    assert loaded.has_offset_head()


def test_G4f_C8_loss_includes_offset_term(tmp_path, monkeypatch):
    """With a non-zero head, the training loss is the picking loss plus offset_weight x masked L1."""
    from tomocpt.training.offsets import offset_targets

    with applied_main_config(H.v2_cfg(tmp_path, **{"particles.offset_weight": 2.0})):
        m = _model(2.0)
        with torch.no_grad():
            for p in m.offset_head.parameters():
                p.normal_(0, 0.05)
        batch = G3.collate(H.items(2))
        logged = {}
        monkeypatch.setattr(m, "log", lambda name, value, *a, **k: logged.__setitem__(name, float(value)))
        m.train()
        total = float(m.training_step(batch, 0))
    assert "loss_offset" in logged
    assert logged["loss_offset"] > 0
    assert total == pytest.approx(logged["loss"], rel=1e-6)
