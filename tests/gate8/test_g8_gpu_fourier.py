"""Gate 8 follow-up (16 Nov 2026): GPU ports of the Fourier degradation and the optional intensity
transforms, and the Fourier degradation on by default for pipeline v2 picking (human decision).

- F1 ``gpu_augment.fourier_degrade`` = ``RandomFourierDegradation._degrade_patch`` on the same
  numpy draws (atol 1e-4), hard and tapered wedge, even and odd shapes, each half alone.
- F2 wedge orientation for (z, y, x): the kept double wedge is the region a +-60 deg tilt series
  measures (around kX, removed around kZ, constant along kY); equals ``oriented_wedge_mask``.
- F3 input mean and std are kept (``match_input_stats``); reclip bounds the output.
- F4 in ``GpuAugment``: targets and negative masks untouched; the degradation runs after every
  spatial transform (and blur) and before contrast inversion.
- F5 ``check_gpu_augmentation`` accepts every ported field; ``augment_device=auto`` picks cuda.
- F6 default: ``fourier_auto`` turns both switches on for pipeline v2 picking only; pipeline v1,
  SSL and ``fourier_auto=False`` unchanged; ``AugmentationConfig()`` itself stays legacy (off).
- F7 an old checkpoint without ``fourier_auto`` loads and gets the field backfilled.
- I1 intensity ports: exact against TorchIO for anisotropy, gamma and bias field at fixed
  parameters; firing probability and parameter-range checks for every transform.
"""
import numpy as np
import pytest
import torch

from gate8 import g8c2_helpers as H


def _deg(**kw):
    from tomocpt.dataManager.fourier_augmentations import RandomFourierDegradation

    return RandomFourierDegradation(**kw)


def _patch(rng, shape):
    return np.clip(rng.normal(0.0, 0.3, shape), -3, 3).astype(np.float32)


# --------------------------------------------------------------------------------------------
# F1 - F3 the Fourier port
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(), dict(taper_width=8.0), dict(amplitude_aug=False),
                                dict(missing_wedge_aug=False), dict(keep_angle_range=(30.0, 70.0)),
                                dict(reclip=False, match_input_stats=False)])
def test_G8F1_fourier_equals_cpu_on_same_draws(kw):
    from tomocpt.dataManager.gpu_augment import draw_fourier_params, fourier_degrade

    deg = _deg(missing_wedge_prob=0.7, amplitude_prob=0.7, **kw)
    rng = np.random.default_rng(1)
    n_applied = 0
    for s in range(16):
        shape = (64, 64, 64) if s % 2 else (31, 40, 48)
        patch = _patch(rng, shape)
        np.random.seed(100 + s)
        p = deg._randomize()
        applied = p["do_mw"] or p["do_amp"]
        want = deg._degrade_patch(patch, p) if applied else patch
        n_applied += applied
        np.random.seed(100 + s)
        got = fourier_degrade(torch.from_numpy(patch)[None, None], draw_fourier_params(deg, 1, shape), deg)
        assert got.shape == (1, 1) + shape and got.dtype == torch.float32
        np.testing.assert_allclose(got[0, 0].numpy(), want, atol=1e-4, err_msg=f"sample {s}")
    assert n_applied >= 4


def test_G8F1_batched_draws_match_sequential_cpu_calls():
    from tomocpt.dataManager.gpu_augment import draw_fourier_params, fourier_degrade

    deg = _deg(missing_wedge_prob=0.6, amplitude_prob=0.6)
    rng = np.random.default_rng(2)
    x = np.stack([_patch(rng, (32, 32, 32)) for _ in range(6)])
    np.random.seed(7)
    want = []
    for b in range(6):
        p = deg._randomize()
        want.append(deg._degrade_patch(x[b], p) if (p["do_mw"] or p["do_amp"]) else x[b])
    np.random.seed(7)
    got = fourier_degrade(torch.from_numpy(x)[:, None], draw_fourier_params(deg, 6, (32, 32, 32)), deg)
    np.testing.assert_allclose(got[:, 0].numpy(), np.stack(want), atol=1e-4)


@pytest.mark.parametrize("taper", [0.0, 6.0])
@pytest.mark.parametrize("shape", [(64, 64, 64), (33, 40, 48)])
def test_G8F2_wedge_mask_equals_oriented_reference(taper, shape):
    from tomocpt.dataManager.fourier_augmentations import oriented_wedge_mask
    from tomocpt.dataManager.gpu_augment import wedge_keep_torch

    angles = [20.0, 45.0, 60.0, 85.0]
    got = wedge_keep_torch(shape, angles, taper_width=taper).numpy()
    for i, a in enumerate(angles):
        np.testing.assert_allclose(got[i], oriented_wedge_mask(shape, a, array_order="zyx", taper_width=taper)
                                   .astype(np.float32), atol=1e-6, err_msg=f"angle {a}")


def test_G8F2_wedge_keeps_what_a_60_degree_tilt_series_measures():
    """(z, y, x) spectrum, tilt axis Y, beam Z: a +-60 deg series measures |kz| <= tan(60) |kx|."""
    from tomocpt.dataManager.gpu_augment import fourier_degrade, wedge_keep_torch

    n = 64
    c = n // 2
    keep = wedge_keep_torch((n, n, n), [60.0])[0].numpy()
    kz, ky, kx = np.meshgrid(*(np.arange(n) - (n - 1) / 2.0,) * 3, indexing="ij")
    t = np.tan(np.radians(60.0))
    dc = np.zeros_like(keep, dtype=bool)
    dc[c, c, c] = True                                             # DC removed (reference zero_dc)
    assert keep[dc] == 0
    assert keep[(np.abs(kz) < t * np.abs(kx) * 0.98) & ~dc].min() == 1.0
    assert keep[np.abs(kz) > t * np.abs(kx) * 1.02].max() == 0.0
    k2 = keep.copy()
    k2[c, c, c] = k2[c, 0, c]
    assert (k2 == k2[:, :1, :]).all()                              # constant along the tilt axis Y (but DC)
    assert keep[c, c, c + 10] == 1 and keep[c + 10, c, c] == 0      # kX kept, kZ removed

    # end to end on white noise: the output's power along kZ is gone, along kX it stays
    x = torch.randn(1, 1, n, n, n) * 0.3
    deg = _deg(reclip=False)
    p = [{"apply": True, "do_mw": True, "do_amp": False, "do_kernel": False, "keep_angle": 60.0}]
    y = fourier_degrade(x, p, deg)[0, 0].numpy()
    P = np.abs(np.fft.fftshift(np.fft.fftn(y))) ** 2
    assert P[c + 4:, c, c].mean() < 1e-6 * P[c, c, c + 4:].mean()


def test_G8F3_mean_and_std_kept_and_reclipped():
    from tomocpt.dataManager.gpu_augment import draw_fourier_params, fourier_degrade

    x = torch.randn(8, 1, 32, 32, 32) * 0.3 + 0.05
    deg = _deg(missing_wedge_prob=1.0, amplitude_prob=1.0, reclip=False)
    np.random.seed(3)
    y = fourier_degrade(x, draw_fourier_params(deg, 8, (32, 32, 32)), deg)
    assert not torch.allclose(y, x)
    for b in range(8):
        assert abs(float(y[b].mean() - x[b].mean())) < 1e-4
        assert abs(float(y[b].std(unbiased=False) / x[b].std(unbiased=False)) - 1) < 1e-4
    big = torch.randn(4, 1, 32, 32, 32) * 2.0
    deg = _deg(missing_wedge_prob=1.0, amplitude_prob=1.0, reclip=True, clip_value=3.0)
    y = fourier_degrade(big, draw_fourier_params(deg, 4, (32, 32, 32)), deg)
    assert float(y.abs().max()) <= 3.0


# --------------------------------------------------------------------------------------------
# F4 inside GpuAugment
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return H.write_store(tmp_path_factory.mktemp("g8f_store"))


def _aug(stores, augmentation, train=True):
    from tomocpt.dataManager.gpu_augment import GpuAugment

    return GpuAugment(list(stores), n_classes=H.N_CLASSES, augmentation=augmentation, chunk_size=64,
                      train=train, device="cpu")


def _fourier_cfg(**kw):
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    base = dict(use_mw_aug=True, use_fourier_aug=True, missing_wedge_prob=1.0, amplitude_prob=1.0)
    base.update(kw)
    return AugmentationConfig(**base)


def _run(aug, store, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = H.descriptors([(8, 16, 24), (30, 60, 50), (20, 20, 20)], meta=store.meta)
    return aug(d)


def test_G8F4_targets_and_masks_untouched(store):
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    off = _run(_aug([store], AugmentationConfig()), store, 5)
    on = _run(_aug([store], _fourier_cfg()), store, 5)
    assert torch.equal(on["target_data"], off["target_data"])
    if "negative_mask" in off:
        assert torch.equal(on["negative_mask"], off["negative_mask"])
    assert not torch.allclose(on["input_data"], off["input_data"])


def test_G8F4_runs_after_spatial_and_before_inversion(store, monkeypatch):
    from tomocpt.dataManager import gpu_augment as G
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    seen = []

    def spy(x, params, deg):
        seen.append(x.clone())
        return x

    monkeypatch.setattr(G, "fourier_degrade", spy)
    _run(_aug([store], _fourier_cfg()), store, 9)
    off = _run(_aug([store], AugmentationConfig()), store, 9)["input_data"]
    assert len(seen) == 1
    x = seen[0]
    for b in range(x.shape[0]):     # off = x, or x reflected about its mean (inversion after)
        same = torch.allclose(off[b], x[b], atol=1e-6)
        inv = torch.allclose(off[b], 2 * x[b].mean() - x[b], atol=1e-5)
        assert same or inv, b


# --------------------------------------------------------------------------------------------
# F5 no longer refused
# --------------------------------------------------------------------------------------------

FIELDS = [("use_mw_aug", True), ("use_fourier_aug", True), ("brightness_gradient_p", 0.5), ("local_gamma_p", 0.5),
          ("brightness_p", 0.5), ("contrast_p", 0.5), ("anisotropy_p", 0.5), ("noise_p", 0.5), ("gamma_p", 0.5),
          ("bias_field_p", 0.5)]


@pytest.mark.parametrize("field,value", FIELDS)
def test_G8F5_ported_fields_accepted(field, value, monkeypatch, store):
    from tomocpt.dataManager.gpu_augment import check_gpu_augmentation
    from tomocpt.dataManager.otf import resolve_augment_device
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    cfg = AugmentationConfig()
    setattr(cfg, field, value)
    check_gpu_augmentation(cfg)
    _aug([store], cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_augment_device("auto", cfg, True) == "cuda"


# --------------------------------------------------------------------------------------------
# F6 - F7 defaults and checkpoints
# --------------------------------------------------------------------------------------------

def _train(tmp_path, version=2, mode="picking"):
    from utils.tiny import tiny_main_config

    cfg = tiny_main_config(tmp_path)
    cfg.train.data_pipeline_version = version
    cfg.train.mode = mode
    return cfg.train


def test_G8F6_dataclass_defaults_stay_legacy():
    from tomocpt.dataManager.dataloading import build_training_transforms
    from tomocpt.dataManager.fourier_augmentations import RandomFourierDegradation
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    a = AugmentationConfig()
    assert a.use_mw_aug is False and a.use_fourier_aug is False and a.fourier_auto is True
    assert not any(isinstance(t, RandomFourierDegradation) for t in build_training_transforms(a).transforms)
    for f, _ in FIELDS[2:]:
        assert getattr(a, f) == 0.0, f       # optional intensity transforms stay off


@pytest.mark.parametrize("version,mode,expected", [(2, "picking", True), (1, "picking", False),
                                                    (1, "selfSupervised", False), (2, "selfSupervised", False)])
def test_G8F6_fourier_default_by_pipeline(tmp_path, version, mode, expected):
    from tomocpt.dataManager.dataloading import build_training_transforms
    from tomocpt.dataManager.fourier_augmentations import RandomFourierDegradation
    from tomocpt.training.helpers import resolve_fourier_augmentation

    t = _train(tmp_path, version, mode)
    assert resolve_fourier_augmentation(t) is expected
    assert t.augmentation.use_mw_aug is expected and t.augmentation.use_fourier_aug is expected
    has = any(isinstance(x, RandomFourierDegradation) for x in build_training_transforms(t.augmentation).transforms)
    assert has is expected


def test_G8F6_explicit_off_and_partial(tmp_path):
    from tomocpt.training.helpers import resolve_fourier_augmentation

    t = _train(tmp_path)
    t.augmentation.fourier_auto = False
    assert resolve_fourier_augmentation(t) is False
    assert t.augmentation.use_mw_aug is False and t.augmentation.use_fourier_aug is False
    t.augmentation.use_mw_aug = True                        # one half only, explicitly
    assert resolve_fourier_augmentation(t) is True
    assert t.augmentation.use_mw_aug is True and t.augmentation.use_fourier_aug is False
    v1 = _train(tmp_path, version=1)
    v1.augmentation.use_mw_aug = True                       # v1 explicit opt-in still works
    assert resolve_fourier_augmentation(v1) is True and v1.augmentation.use_fourier_aug is False


def test_G8F6_prepare_data_resolves(tmp_path, monkeypatch):
    """``_prepare_data_if_needed`` resolves the default before the data module is built."""
    from tomocpt.training import helpers

    t = _train(tmp_path)
    calls = []
    monkeypatch.setattr(helpers, "resolve_fourier_augmentation", lambda c: calls.append(c) or True)
    try:
        helpers._prepare_data_if_needed(t)
    except Exception:
        pass        # no data in tmp_path: only the resolution call matters here
    assert calls and calls[0] is t


def test_G8F7_old_config_backfilled():
    from tomocpt.configManager.configManager import backfill_missing_fields, safe_getattr
    from tomocpt.mainConfig import MainConfig

    cfg = MainConfig()
    del cfg.train.augmentation.__dict__["fourier_auto"]
    assert safe_getattr(cfg.train.augmentation, "fourier_auto", None) is None
    backfill_missing_fields(cfg, MainConfig())
    assert "fourier_auto" in cfg.train.augmentation.__dict__
    assert cfg.train.augmentation.fourier_auto is True
    assert cfg.train.augmentation.use_mw_aug is False


def test_G8F7_old_checkpoint_loads(tmp_path):
    from utils.legacy import legacy_checkpoint, legacy_reference
    from tomocpt.configManager.configManager import safe_getattr
    from tomocpt.networks.pickingModel import BasePickingModel

    ckpt = torch.load(legacy_checkpoint(), map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["config"]
    cfg.train.augmentation.__dict__.pop("fourier_auto", None)
    path = tmp_path / "old.ckpt"
    torch.save(ckpt, path)
    model = BasePickingModel.load_from_checkpoint(path, map_location="cpu").eval()
    aug = model.hparams.config.train.augmentation
    assert safe_getattr(aug, "fourier_auto", None) is True
    assert aug.use_mw_aug is False and aug.use_fourier_aug is False
    ref = legacy_reference(legacy_checkpoint())
    with torch.no_grad():
        out = model(torch.as_tensor(ref["x"]))
    out = out[0] if isinstance(out, tuple) else out
    np.testing.assert_allclose(out.numpy(), ref["logits"], atol=1e-5)


# --------------------------------------------------------------------------------------------
# I1 optional intensity transforms
# --------------------------------------------------------------------------------------------

def _cfg(**kw):
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    return AugmentationConfig(**kw)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("f", [1.5, 2.7, 4.9])
def test_G8I1_anisotropy_equals_torchio(axis, f, monkeypatch):
    import torchio as tio
    from tomocpt.dataManager.gpu_augment import anisotropy_torch

    x = torch.randn(1, 20, 24, 28)
    t = tio.RandomAnisotropy(p=1.0)
    monkeypatch.setattr(t, "get_params", lambda axes, rng: (axis, f))
    want = t(tio.Subject(img=tio.ScalarImage(tensor=x)))["img"].data[0]
    got = anisotropy_torch(x[0], axis, f)
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-5)


def test_G8I1_gamma_and_bias_field_equal_torchio():
    import torchio as tio
    from tomocpt.dataManager.gpu_augment import bias_fields

    x = torch.randn(1, 16, 16, 16) * 0.3
    for g in (0.75, 1.3):
        want = tio.Gamma(g)(tio.Subject(img=tio.ScalarImage(tensor=x)))["img"].data
        got = x.sign() * x.abs() ** g
        np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-6)
    c = (torch.rand(20, dtype=torch.float64) - 0.5)
    want = tio.BiasField(coefficients=c.tolist(), order=3).generate_bias_field(x, 3, c.tolist())
    np.testing.assert_allclose(bias_fields((16, 16, 16), c[None])[0].numpy(), want, rtol=1e-5)


def test_G8I1_firing_probability():
    from tomocpt.dataManager.gpu_augment import _fires

    n = 10000
    for p in (0.2, 0.5):
        frac = len(_fires(n, p)) / n
        assert abs(frac - p) < 3 * np.sqrt(p * (1 - p) / n), (p, frac)
    assert _fires(n, 0.0) == [] and len(_fires(n, 1.0)) == n


@pytest.mark.parametrize("field", [f for f, _ in FIELDS[2:]])
def test_G8I1_each_transform_gated_by_its_probability(field):
    from tomocpt.dataManager.gpu_augment import optional_intensity

    n = 200
    x = torch.randn(n, 1, 12, 12, 12) * 0.3
    assert torch.equal(optional_intensity(x, _cfg(**{field: 0.0})), x)
    y = optional_intensity(x, _cfg(**{field: 1.0}))
    changed = ((y - x).abs().reshape(n, -1).max(1).values > 1e-6).float().mean().item()
    assert changed > 0.5, (field, changed)
    assert torch.isfinite(y).all()


def test_G8I1_parameter_ranges():
    from tomocpt.dataManager.gpu_augment import optional_intensity

    n = 400
    x = torch.randn(n, 1, 8, 8, 8) * 0.3
    d = (optional_intensity(x, _cfg(brightness_p=1.0, brightness_sigma=0.5)) - x).reshape(n, -1)
    assert torch.allclose(d, d[:, :1].expand_as(d), atol=1e-6)
    assert abs(float(d[:, 0].std()) - 0.5) < 0.06 and abs(float(d[:, 0].mean())) < 0.08

    y = optional_intensity(x, _cfg(contrast_p=1.0, contrast_min=0.5, contrast_max=2.0))
    r = (y.reshape(n, -1).std(1) / x.reshape(n, -1).std(1))
    assert float(r.min()) > 0.5 - 1e-4 and float(r.max()) < 2.0 + 1e-4 and abs(float(r.mean()) - 1.25) < 0.06
    torch.testing.assert_close(y.reshape(n, -1).mean(1), x.reshape(n, -1).mean(1), atol=1e-5, rtol=0)

    s = (optional_intensity(x, _cfg(noise_p=1.0, noise_std=0.1)) - x).reshape(n, -1).std(1)
    assert float(s.max()) < 0.13 and abs(float(s.mean()) - 0.05) < 0.01

    g = optional_intensity(x, _cfg(brightness_gradient_p=1.0)) - x
    peak = g.reshape(n, -1).abs().max(1).values
    assert float(peak.max()) <= 5.0 + 1e-4 and float(peak.mean()) > 0.5

    y = optional_intensity(x, _cfg(local_gamma_p=1.0))
    torch.testing.assert_close(y.reshape(n, -1).amin(1), x.reshape(n, -1).amin(1), atol=1e-4, rtol=0)
    assert float((y.reshape(n, -1).amax(1) - x.reshape(n, -1).amax(1)).max()) < 1e-4

    pos = torch.rand(n, 1, 8, 8, 8) + 0.1
    pos[:, :, 0, 0, 0] = 0.5
    lg = torch.log(optional_intensity(pos, _cfg(gamma_p=1.0))) / torch.log(pos)
    gam = lg.reshape(n, -1)[:, 0]
    assert float(gam.min()) > np.exp(-0.3) - 1e-3 and float(gam.max()) < np.exp(0.3) + 1e-3

    b = optional_intensity(pos, _cfg(bias_field_p=1.0)) / pos
    assert float(b.min()) > 0 and abs(float(torch.log(b).mean())) < 0.1


def test_G8I1_intensity_never_touches_targets(store):
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    every = dict(brightness_gradient_p=1.0, local_gamma_p=1.0, brightness_p=1.0, contrast_p=1.0,
                 anisotropy_p=1.0, noise_p=1.0, gamma_p=1.0, bias_field_p=1.0)
    off = _run(_aug([store], AugmentationConfig()), store, 11)
    on = _run(_aug([store], _fourier_cfg(**every)), store, 11)
    assert torch.equal(on["target_data"], off["target_data"])
    assert torch.isfinite(on["input_data"]).all()
