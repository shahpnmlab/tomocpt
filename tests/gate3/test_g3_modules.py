"""G3-C2, G3-C3, G3-C7 (module level): ScaleEmbedding, FiLM, ScaleConditionedSwinUNETR,
load_legacy_state_dict, load_single_class_into_multiclass."""

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H
from utils.tiny import tiny_swin_kwargs

ATOL = 1e-6
FS = tiny_swin_kwargs()["feature_size"]
#: FiLM channels in the decoder path: encoder10, decoder5, decoder4, decoder3, decoder2, decoder1
FILM_CHANNELS = [16 * FS, 8 * FS, 4 * FS, 2 * FS, FS, FS]


def _legacy_net(out_channels=1, seed=3):
    from tomocpt.networks.swinunetr import MySwinUNETR

    kw = dict(tiny_swin_kwargs(), out_channels=out_channels)
    net = MySwinUNETR(**kw)
    H.load_common().reinit_(net, seed=seed)
    return net.eval()


def _cond_net(out_channels=1):
    from tomocpt.networks.swinunetr import ScaleConditionedSwinUNETR

    return ScaleConditionedSwinUNETR(**dict(tiny_swin_kwargs(), out_channels=out_channels)).eval()


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_G3_C2_structure_scale_embedding_and_film_zero_init():
    from tomocpt.networks.conditioning import FiLM, ScaleEmbedding

    emb = ScaleEmbedding(dim=64)
    # [t, sin(2^k t), cos(2^k t)] for k = 0..7 -> 17 features; Linear(17->64) SiLU Linear(64->64)
    assert _n_params(emb) == 17 * 64 + 64 + 64 * 64 + 64
    e = emb(H.apix_tensor(10.0, B=3))
    assert e.shape[0] == 3 and e.shape[-1] == 64 and e.numel() == 3 * 64
    assert torch.isfinite(e).all()

    film = FiLM(24, 64)
    assert _n_params(film) == 64 * 2 * 24 + 2 * 24  # Linear(cond_dim -> 2C)
    assert all(torch.count_nonzero(p) == 0 for p in film.parameters()), "FiLM must start at zero"

    net = _cond_net()
    assert isinstance(net.cond.scale_embed, ScaleEmbedding)
    assert isinstance(net.cond.film, torch.nn.ModuleList)
    assert len(net.cond.film) == 6
    assert all(isinstance(f, FiLM) for f in net.cond.film)
    assert [_n_params(f) for f in net.cond.film] == [64 * 2 * c + 2 * c for c in FILM_CHANNELS]
    for f in net.cond.film:
        assert all(torch.count_nonzero(p) == 0 for p in f.parameters()), "FiLM must start at zero"


def test_G3_C2_zero_init_matches_legacy_logits():
    """With legacy weights loaded, the conditioned model gives the legacy logits (and
    hidden[4]) for apix 5, 10, 20 and None."""
    from tomocpt.networks.swinunetr import load_legacy_state_dict

    legacy = _legacy_net()
    cond = _cond_net()
    missing, unexpected = load_legacy_state_dict(cond, legacy.state_dict())
    assert list(unexpected) == []
    cond_keys = {k for k in cond.state_dict() if k.startswith("cond.")}
    assert cond_keys and set(missing) == cond_keys

    x = H.fixed_input(2, seed=5)
    with torch.no_grad():
        ref_logits, ref_hidden = legacy(x)
        out_none = cond(x)
        out_none_kw = cond(x, apix=None)
        assert isinstance(out_none, tuple) and len(out_none) == 2
        for out in (out_none, out_none_kw):
            np.testing.assert_allclose(out[0].numpy(), ref_logits.numpy(), rtol=0, atol=ATOL)
            np.testing.assert_allclose(out[1].numpy(), ref_hidden.numpy(), rtol=0, atol=ATOL)
        for apix in (5.0, 10.0, 20.0):
            logits, hidden = cond(x, apix=H.apix_tensor(apix, B=2))
            np.testing.assert_allclose(logits.numpy(), ref_logits.numpy(), rtol=0, atol=ATOL, err_msg=f"apix={apix}")
            np.testing.assert_allclose(hidden.numpy(), ref_hidden.numpy(), rtol=0, atol=ATOL)


def test_G3_C2_apix_none_is_exactly_unconditioned_even_with_trained_film():
    """``apix=None`` gives exactly ``MySwinUNETR.forward`` (CLAUDE.md 5.3), also when the FiLM
    layers are non-zero; with non-zero FiLM the voxel size changes the output."""
    from tomocpt.networks.swinunetr import load_legacy_state_dict

    legacy = _legacy_net()
    cond = _cond_net()
    load_legacy_state_dict(cond, legacy.state_dict())
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in cond.cond.parameters():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    x = H.fixed_input(1, seed=6)
    with torch.no_grad():
        ref = legacy(x)[0]
        np.testing.assert_allclose(cond(x)[0].numpy(), ref.numpy(), rtol=0, atol=ATOL)
        a5 = cond(x, apix=H.apix_tensor(5.0))[0]
        a20 = cond(x, apix=H.apix_tensor(20.0))[0]
    assert float((a5 - ref).abs().max()) > 1e-3
    assert float((a5 - a20).abs().max()) > 1e-3


def test_G3_C2_scalar_apix_matches_tensor_apix():
    """Gate 4 calls ``forward(x, apix=level.apix_work)`` with a float; it must equal the
    ``[B, 1]`` tensor form of the training batch."""
    cond = _cond_net()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in cond.cond.parameters():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    x = H.fixed_input(1, seed=7)
    with torch.no_grad():
        a = cond(x, apix=H.apix_tensor(14.0))[0]
        b = cond(x, apix=14.0)[0]
    np.testing.assert_allclose(b.numpy(), a.numpy(), rtol=0, atol=ATOL)


def test_G3_C2_load_legacy_state_dict_raises_on_bad_keys():
    from tomocpt.networks.swinunetr import load_legacy_state_dict

    sd = _legacy_net().state_dict()
    extra = dict(sd)
    extra["not_a_module.weight"] = torch.zeros(1)
    with pytest.raises(Exception):
        load_legacy_state_dict(_cond_net(), extra)
    lacking = dict(sd)
    del lacking["decoder1.conv_block.conv1.conv.weight"]  # a non-cond key missing
    with pytest.raises(Exception):
        load_legacy_state_dict(_cond_net(), lacking)


def test_G3_C3_every_parameter_gets_a_gradient_with_conditioning():
    """One backward pass with conditioning: every parameter has ``grad is not None`` and at
    least one FiLM weight gradient is non-zero (zero-initialised FiLM still learns)."""
    from tomocpt.networks.swinunetr import load_legacy_state_dict
    from tomocpt.training.multiclass_loss import multiclass_picking_loss

    legacy = _legacy_net(out_channels=1)
    cond = _cond_net(out_channels=1)
    load_legacy_state_dict(cond, legacy.state_dict())
    cond.train()
    b = H.v2_batch([15.0], B=2, seed=3, apix=[7.0, 20.0])
    logits, _ = cond(b["input_data"], apix=b["apix_work"])
    loss = multiclass_picking_loss(torch.sigmoid(logits), b["target_data"], b["class_available"],
                                   b["class_d_vox"], checkpoint=False)
    loss.backward()
    no_grad = [n for n, p in cond.named_parameters() if p.requires_grad and p.grad is None]
    assert no_grad == []
    film_w = [p.grad for n, p in cond.cond.film.named_parameters() if n.endswith("weight")]
    assert film_w and any(torch.count_nonzero(g) > 0 for g in film_w)


def test_G3_C7_single_class_warm_start_every_channel_equals_old_picker():
    """``load_single_class_into_multiclass``: each of C channels gives the single-class logits,
    for a plain multi-class model and for a conditioned one (any apix: FiLM starts at zero)."""
    from tomocpt.networks.swinunetr import MySwinUNETR, load_single_class_into_multiclass

    single = _legacy_net(out_channels=1)
    x = H.fixed_input(1, seed=8)
    with torch.no_grad():
        ref_logits, ref_hidden = single(x)

    multi = MySwinUNETR(**dict(tiny_swin_kwargs(), out_channels=3)).eval()
    assert load_single_class_into_multiclass(multi, single.state_dict()) is None
    with torch.no_grad():
        logits, hidden = multi(x)
    assert tuple(logits.shape) == (1, 3, 64, 64, 64)
    for c in range(3):
        np.testing.assert_allclose(logits[:, c].numpy(), ref_logits[:, 0].numpy(), rtol=0, atol=ATOL)
    np.testing.assert_allclose(hidden.numpy(), ref_hidden.numpy(), rtol=0, atol=ATOL)

    cond = _cond_net(out_channels=4)
    load_single_class_into_multiclass(cond, single.state_dict())
    with torch.no_grad():
        for apix in (None, 8.0, 20.0):
            logits = cond(x, apix=None if apix is None else H.apix_tensor(apix))[0]
            assert logits.shape[1] == 4
            for c in range(4):
                np.testing.assert_allclose(logits[:, c].numpy(), ref_logits[:, 0].numpy(), rtol=0, atol=ATOL)
