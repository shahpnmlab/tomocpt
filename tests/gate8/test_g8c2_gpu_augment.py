"""Gate 8c.2: augmentation and target rendering on the device of the batch
(``train.otf.augment_device=cuda``, run in ``on_after_batch_transfer``). The code is device
agnostic, so these tests run on CPU tensors; ``test_G8c2_cuda_matches_cpu`` repeats one case on
the GPU when there is one.

- render_targets_torch / render_tubes_torch = the CPU renderers (atol 1e-5).
- G8c-C6 spatial: identity and flips give the stored crop exactly; for 100 random affines the
  target equals the CPU render of the mapped points and segments (atol 1e-5), particle maxima sit
  at the mapped positions (<= 0.5 voxel per axis), and the input equals a trilinear read of the
  whole tomogram, padded with its median (atol 1e-4). A constant elastic field is a shift.
- G8c-C7 intensity: contrast inversion at p 0.5 +- 3 sigma over 1e4 samples; intensity transforms
  never change the target; elastic displacement statistics match TorchIO's within 10 %.
- G8c-C10 negative masks warped (nearest) with the same grid; one training step with contrastive
  anchors and mask negatives on a GPU-path batch.
"""
import numpy as np
import pytest
import torch

from gate8 import g8c2_helpers as H


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return H.write_store(tmp_path_factory.mktemp("g8c2_store"))


def _aug(stores, train=False, augmentation=None, **kw):
    from tomocpt.dataManager.gpu_augment import GpuAugment
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    return GpuAugment(list(stores), n_classes=H.N_CLASSES,
                      augmentation=augmentation if augmentation is not None else AugmentationConfig(),
                      chunk_size=64, train=train, device="cpu", **kw)


def _cpu_target(points, segs, n=64):
    from tomocpt.labels.filaments import render_tubes
    from tomocpt.labels.helpers import render_targets

    t = render_targets(points, (n,) * 3, H.N_CLASSES, border_policy="clip")
    if segs is not None and len(segs):
        t = np.maximum(t, render_tubes(segs, (n,) * 3, H.N_CLASSES))
    return t


# --------------------------------------------------------------------------------------------
# torch renderers
# --------------------------------------------------------------------------------------------

def test_G8c2_render_targets_torch_equals_cpu():
    from tomocpt.labels.helpers import render_targets
    from tomocpt.labels.render_torch import render_targets_torch

    rng = np.random.default_rng(3)
    shape = (40, 48, 56)
    for trial in range(6):
        n = int(rng.integers(0, 30))
        pts = np.stack([rng.uniform(-6, shape[0] + 6, n), rng.uniform(-6, shape[1] + 6, n),
                        rng.uniform(-6, shape[2] + 6, n), rng.integers(0, 3, n), rng.uniform(2.5, 30, n)],
                       axis=1).astype(np.float32)
        want = render_targets(pts, shape, 3, border_policy="clip")
        got = render_targets_torch(torch.as_tensor(pts), shape, 3)
        assert got.shape == (3,) + shape and got.dtype == torch.float32
        np.testing.assert_allclose(got.numpy(), want, atol=1e-5, err_msg=f"trial {trial}")


def test_G8c2_render_targets_torch_batched():
    from tomocpt.labels.helpers import render_targets
    from tomocpt.labels.render_torch import render_targets_torch

    rng = np.random.default_rng(4)
    shape = (32, 32, 32)
    per = [np.stack([rng.uniform(0, 32, k), rng.uniform(0, 32, k), rng.uniform(0, 32, k),
                     rng.integers(0, 2, k), rng.uniform(3, 14, k)], axis=1).astype(np.float32) for k in (5, 0, 9)]
    pts = torch.as_tensor(np.concatenate(per))
    bi = torch.as_tensor(np.concatenate([np.full(len(p), b) for b, p in enumerate(per)]).astype(np.int64))
    got = render_targets_torch(pts, shape, 2, batch_index=bi, batch_size=3)
    assert got.shape == (3, 2) + shape
    for b, p in enumerate(per):
        np.testing.assert_allclose(got[b].numpy(), render_targets(p, shape, 2), atol=1e-5)


def test_G8c2_render_tubes_torch_equals_cpu():
    from tomocpt.labels.filaments import render_tubes
    from tomocpt.labels.render_torch import render_tubes_torch

    rng = np.random.default_rng(5)
    shape = (40, 48, 56)
    rows = []
    for f in range(5):
        a = rng.uniform(-5, 50, 3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        cls, dv = int(rng.integers(0, 2)), float(rng.uniform(4, 20))
        for k in range(int(rng.integers(1, 6))):
            b = a + rng.uniform(2, 12) * d
            rows.append([*a, *b, cls, dv])
            a = b
    rows.append([10, 10, 10, 10, 10, 10, 1, 8.0])  # zero-length segment: a ball
    segs = np.asarray(rows, np.float32)
    want = render_tubes(segs, shape, 2)
    got = render_tubes_torch(torch.as_tensor(segs), shape, 2)
    np.testing.assert_allclose(got.numpy(), want, atol=1e-5)
    bi = torch.as_tensor((np.arange(len(segs)) % 2).astype(np.int64))
    got2 = render_tubes_torch(torch.as_tensor(segs), shape, 2, batch_index=bi, batch_size=2)
    for b in range(2):
        np.testing.assert_allclose(got2[b].numpy(), render_tubes(segs[bi.numpy() == b], shape, 2), atol=1e-5)


# --------------------------------------------------------------------------------------------
# G8c-C6 spatial transform
# --------------------------------------------------------------------------------------------

def test_G8c_C6_identity_gives_the_stored_crop(store):
    aug = _aug([store])
    corners = [(0, 0, 0), (32, 64, 32), (32, 64, 64)]
    out = aug(H.descriptors(corners, meta=store.meta))
    assert out["input_data"].shape == (3, 1, 64, 64, 64) and out["input_data"].dtype == torch.float32
    assert out["target_data"].shape == (3, H.N_CLASSES, 64, 64, 64)
    assert out["negative_mask"].shape == (3, 1, 64, 64, 64) and out["negative_mask"].dtype == torch.bool
    for b, c in enumerate(corners):
        np.testing.assert_array_equal(out["input_data"][b, 0].numpy(), store.crop(c))
        want = _cpu_target(store.points_in_box(c), store.segments_in_box(c))
        np.testing.assert_allclose(out["target_data"][b].numpy(), want, atol=1e-5)
        np.testing.assert_array_equal(out["negative_mask"][b, 0].numpy(), store.negative_crop(c))
    for k in ("class_available", "class_d_vox", "apix_work", "dataset_id"):
        assert k in out
    assert "store_index" not in out and "corner" not in out


def test_G8c_C6_flips_are_exact(store):
    aug = _aug([store])
    c = (16, 40, 50)
    for flips in [(1, 1, -1), (-1, 1, 1), (-1, -1, -1)]:
        A = torch.diag(torch.tensor(flips, dtype=torch.float64))[None]
        out = aug(H.descriptors([c], meta=store.meta), affine=A)
        ref = store.crop(c)
        ref = np.flip(ref, axis=tuple(a for a in range(3) if flips[a] < 0))
        np.testing.assert_array_equal(out["input_data"][0, 0].numpy(), ref)
        want = _cpu_target(H.map_points(store.points, c, A[0].numpy()), H.map_segments(store.segments, c, A[0].numpy()))
        np.testing.assert_allclose(out["target_data"][0].numpy(), want, atol=1e-5)


def test_G8c_C6_random_affines_targets_at_mapped_points(store):
    from scipy.ndimage import map_coordinates

    aug = _aug([store])
    rng = np.random.default_rng(11)
    vol = np.asarray(store.volume, np.float32)
    med = float(np.median(vol))
    assert abs(float(aug.pad_values[0]) - med) < 1e-3
    n_checked_max = 0
    for t in range(100):
        A = H.random_affine_np(rng)
        c = tuple(int(rng.integers(-8, s - 56)) for s in store.shape)  # corners near and past the border
        out = aug(H.descriptors([c], meta=store.meta), affine=torch.as_tensor(A)[None])
        pts = H.map_points(store.points, c, A)
        segs = H.map_segments(store.segments, c, A)
        tgt = out["target_data"][0].numpy()
        np.testing.assert_allclose(tgt, _cpu_target(pts, segs), atol=1e-5, err_msg=f"affine {t}")
        for z, y, x, ci, d in pts:  # maxima of the particle channel at the mapped centres
            R = d / 2
            if not all(R + 1 <= v <= 62 - R for v in (z, y, x)):
                continue
            lo = [int(np.floor(v - R)) for v in (z, y, x)]
            win = tgt[0, lo[0]:lo[0] + int(2 * R) + 2, lo[1]:lo[1] + int(2 * R) + 2, lo[2]:lo[2] + int(2 * R) + 2]
            arg = np.array(np.unravel_index(np.argmax(win), win.shape)) + lo
            assert np.all(np.abs(arg - np.array([z, y, x])) <= 0.5 + 1e-6), (t, arg, (z, y, x))
            n_checked_max += 1
        if t < 10:
            src = H.source_coords(c, A)
            ref = map_coordinates(vol, src.reshape(3, -1), order=1, mode="grid-constant", cval=med).reshape(64, 64, 64)
            np.testing.assert_allclose(out["input_data"][0, 0].numpy(), ref, atol=1e-4, err_msg=f"input {t}")
    assert n_checked_max >= 50


def test_G8c_C6_constant_elastic_field_is_a_shift(store):
    from scipy.ndimage import map_coordinates

    aug = _aug([store])
    c = (16, 32, 32)
    e = np.array([1.5, -2.25, 0.5])
    field = torch.as_tensor(np.broadcast_to(e[:, None, None, None], (3, 64, 64, 64)).copy())[None]
    out = aug(H.descriptors([c], meta=store.meta), affine=torch.eye(3, dtype=torch.float64)[None], elastic=field)
    vol = np.asarray(store.volume, np.float32)
    src = H.source_coords(c, np.eye(3), field=field[0].numpy())
    ref = map_coordinates(vol, src.reshape(3, -1), order=1, mode="grid-constant", cval=float(np.median(vol)))
    np.testing.assert_allclose(out["input_data"][0, 0].numpy(), ref.reshape(64, 64, 64), atol=1e-4)
    # target: rendered in the affine frame, warped with the same grid (linear): the particle peaks move by -e
    pts = H.map_points(store.points, np.asarray(c) + e, np.eye(3))
    tgt = out["target_data"][0].numpy()
    shifted = _cpu_target(pts, None)
    wrong = _cpu_target(H.map_points(store.points, np.asarray(c) - e, np.eye(3)), None)
    inner = (slice(4, 60),) * 3
    err = np.abs(tgt[0][inner] - shifted[0][inner]).max()  # linear warp of a sharp peak at a half-voxel shift
    assert err < 0.35, err
    assert np.abs(tgt[0][inner] - wrong[0][inner]).max() > 2.5 * err  # the shift has the documented sign


def test_G8c_C6_sampled_affines(store):
    """The random spatial draw: rotation_mode=affine (affine_p, degrees, TorchIO's +-10 % scale)
    and rotation_mode=flips (flip_p per axis)."""
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    torch.manual_seed(0)
    aug = _aug([store], train=True)
    A = aug.sample_affines(4000)
    assert A.shape == (4000, 3, 3)
    moved = (A - torch.eye(3, dtype=A.dtype)).abs().amax(dim=(1, 2)) > 1e-6
    p, sd = 0.8, (0.8 * 0.2 / 4000) ** 0.5
    assert abs(moved.double().mean().item() - p) < 3 * sd
    sv = torch.linalg.svdvals(A[moved].double())
    assert sv.min() >= 0.9 - 1e-4 and sv.max() <= 1.1 + 1e-4

    cfg = AugmentationConfig()
    cfg.rotation_mode = "flips"
    cfg.flip_p = 0.5
    augf = _aug([store], train=True, augmentation=cfg)
    F = augf.sample_affines(4000)
    off = F - torch.diag_embed(torch.diagonal(F, dim1=1, dim2=2))
    assert off.abs().max() == 0
    diag = torch.diagonal(F, dim1=1, dim2=2)
    assert set(diag.unique().tolist()) <= {-1.0, 1.0}
    frac = (diag < 0).double().mean(dim=0)
    assert torch.all((frac - 0.5).abs() < 3 * (0.25 / 4000) ** 0.5), frac
    assert augf.sample_elastic(8) is None  # flips mode has no elastic deformation


# --------------------------------------------------------------------------------------------
# G8c-C7 intensity and elastic statistics
# --------------------------------------------------------------------------------------------

def test_G8c_C7_contrast_inversion_rate(store):
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    cfg = AugmentationConfig()
    cfg.blur_weight = 0.0
    cfg.elastic_weight = 0.0
    aug = _aug([store], train=True, augmentation=cfg)
    torch.manual_seed(1)
    x = torch.randn(10000, 1, 4, 4, 4) + 0.3
    y = aug.intensity(x.clone())
    inv = torch.isclose(y, 2 * x.mean(dim=(1, 2, 3, 4), keepdim=True) - x, atol=1e-5).flatten(1).all(1)
    same = torch.equal(y[~inv], x[~inv])
    assert same
    sd = (0.25 / 10000) ** 0.5
    assert abs(inv.double().mean().item() - 0.5) < 3 * sd


def test_G8c_C7_intensity_never_touches_the_target(store):
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    cfg = AugmentationConfig()
    cfg.rotation_mode = "flips"
    cfg.flip_p = 0.0
    cfg.elastic_blur_p = 1.0
    cfg.blur_weight = 1.0
    c = [(16, 32, 32), (0, 64, 64)]
    ref = _aug([store])(H.descriptors(c, meta=store.meta))
    torch.manual_seed(2)
    out = _aug([store], train=True, augmentation=cfg)(H.descriptors(c, meta=store.meta))
    assert torch.equal(out["target_data"], ref["target_data"])
    assert torch.equal(out["negative_mask"], ref["negative_mask"])
    assert not torch.equal(out["input_data"], ref["input_data"])  # blur (and inversion) did run


def _tio_field_stats(n_draws, size, margin):
    import torchio as tio

    g = np.stack(np.meshgrid(*(np.arange(size, dtype=np.float32),) * 3, indexing="ij"))
    inner = (slice(None),) + (slice(margin, size - margin),) * 3
    means, maxes = [], []
    tr = tio.RandomElasticDeformation()
    for _ in range(n_draws):
        out = tr(tio.ScalarImage(tensor=torch.as_tensor(g))).data.numpy()
        d = np.linalg.norm((out - g)[inner], axis=0)
        means.append(d.mean())
        maxes.append(d.max())
    return float(np.mean(means)), float(np.mean(maxes))


def test_G8c_C7_elastic_statistics_match_torchio():
    from tomocpt.dataManager.gpu_augment import elastic_field

    size, margin, n = 32, 8, 1000
    torch.manual_seed(3)
    np.random.seed(3)
    f = elastic_field(n, size)
    assert f.shape == (n, 3, size, size, size)
    inner = (slice(None), slice(None)) + (slice(margin, size - margin),) * 3
    d = torch.linalg.vector_norm(f[inner].double(), dim=1)
    g_mean, g_max = d.mean().item(), d.flatten(1).amax(1).mean().item()
    t_mean, t_max = _tio_field_stats(n, size, margin)
    assert abs(g_mean / t_mean - 1) < 0.10, (g_mean, t_mean)
    assert abs(g_max / t_max - 1) < 0.10, (g_max, t_max)


def test_G8c_C7_formerly_unported_options_are_accepted(store):
    """Re-frozen 17 Nov 2026 (human-approved dispute): the Fourier and intensity options have GPU ports."""
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    for field, value in [("use_mw_aug", True), ("use_fourier_aug", True), ("brightness_p", 0.5),
                         ("noise_p", 0.5)]:
        cfg = AugmentationConfig()
        setattr(cfg, field, value)
        _aug([store], train=True, augmentation=cfg)


# --------------------------------------------------------------------------------------------
# G8c-C10 negative masks and contrastive anchors on GPU-path batches
# --------------------------------------------------------------------------------------------

def test_G8c_C10_negative_mask_follows_the_grid(store):
    from scipy.ndimage import map_coordinates

    aug = _aug([store])
    rng = np.random.default_rng(12)
    mask = np.unpackbits(np.load(f"{store.path}/negative_mask.npy"), axis=-1)[..., :store.shape[2]].astype(np.float32)
    for t in range(5):
        A = H.random_affine_np(rng)
        c = tuple(int(rng.integers(0, s - 64)) for s in store.shape)
        out = aug(H.descriptors([c], meta=store.meta), affine=torch.as_tensor(A)[None])
        src = H.source_coords(c, A)
        ref = map_coordinates(mask, src.reshape(3, -1), order=0, mode="constant", cval=0).reshape(64, 64, 64) > 0.5
        agree = (out["negative_mask"][0, 0].numpy() == ref).mean()
        assert agree > 0.999, (t, agree)


def test_G8c_C10_contrastive_and_mask_negatives_one_step(tmp_path, store):
    import importlib.util
    import sys
    from pathlib import Path

    from gate3 import g3_helpers as G3
    from gate7 import g7_helpers as H7
    from tomocpt.networks.pickingModel import BasePickingModel
    from utils.tiny import applied_main_config

    golden = Path(__file__).resolve().parents[1] / "golden"
    if str(golden) not in sys.path:
        sys.path.insert(0, str(golden))
    spec = importlib.util.spec_from_file_location("_gate7_common_g8c2", golden / "gate7_common.py")
    g7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g7)

    torch.manual_seed(5)
    batch = _aug([store], train=True)(H.descriptors([(16, 32, 32), (32, 40, 60)], meta=store.meta))
    assert batch["negative_mask"].any()
    items = [{k: v[b] for k, v in batch.items()} for b in range(2)]
    cfg = H7.disc_config(tmp_path, g7.PARTICLES, g7.FILAMENTS, class_head="softmax", cond=True,
                         contrastive_weight=0.1)
    with applied_main_config(cfg):
        torch.manual_seed(0)
        model = BasePickingModel(lr=1e-3, train_continue=None)
        loss, grads, _ = G3.first_step_loss_and_grads(model, items, tmp_path / "tb")
    assert np.isfinite(loss)
    missing = sorted(n for n, g in grads.items() if g is None)
    assert not missing, missing[:10]


# --------------------------------------------------------------------------------------------
# wiring: OnTheFlyData with augment_device=cuda
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phantom_store(tmp_path_factory):
    from gate8 import g8c_helpers as G

    root = tmp_path_factory.mktemp("g8c2_phantom")
    df = G.write_inputs(root / "in")
    return G.build(df, root / "store", storage="tomogram")


def test_G8c2_datamodule_yields_descriptors_and_transfers(phantom_store):
    from gate8 import g8c_helpers as G

    gpu = G.otf_data(phantom_store, augment_device="cuda", batch_size=2)
    cpu = G.otf_data(phantom_store, augment_device="cpu", batch_size=2)
    tb = next(iter(gpu.train_dataloader()))
    assert set(tb) >= {"store_index", "corner", "otf_train"} and "input_data" not in tb
    assert tb["otf_train"].all()
    out = gpu.on_after_batch_transfer(tb, 0)
    assert out["input_data"].shape == (2, 1, 64, 64, 64)
    assert out["target_data"].shape[:2] == (2, 2)
    assert out["class_available"].shape == (2, 2) and out["apix_work"].shape == (2, 1)
    # validation: no augmentation, the same crops and contents as the CPU path
    vb = next(iter(gpu.val_dataloader()))
    assert not vb["otf_train"].any()
    vg = gpu.on_after_batch_transfer(vb, 0)
    vc = next(iter(cpu.val_dataloader()))
    np.testing.assert_array_equal(vg["input_data"].numpy(), vc["input_data"].numpy())
    np.testing.assert_allclose(vg["target_data"].numpy(), vc["target_data"].numpy(), atol=1e-5)
    for k in ("class_available", "class_d_vox", "apix_work", "dataset_id"):
        assert torch.equal(vg[k], vc[k]), k


def test_G8c2_datamodule_accepts_fourier_on_gpu_path(phantom_store):
    """Re-frozen 17 Nov 2026 (human-approved dispute): the Fourier degradation has a GPU port."""
    from gate2 import g2_helpers as G2
    from gate8 import g8c_helpers as G

    aug = G2.no_spatial_aug()
    aug.use_mw_aug = True
    G.otf_data(phantom_store, augmentation=aug, augment_device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_G8c2_cuda_matches_cpu(store):
    from tomocpt.dataManager.gpu_augment import GpuAugment
    from tomocpt.defaultConfigs.train_config import AugmentationConfig

    rng = np.random.default_rng(13)
    A = torch.as_tensor(np.stack([H.random_affine_np(rng) for _ in range(2)]))
    d = H.descriptors([(8, 16, 24), (30, 60, 50)], meta=store.meta)
    a_cpu = GpuAugment([store], n_classes=H.N_CLASSES, augmentation=AugmentationConfig(), train=False, device="cpu")
    a_gpu = GpuAugment([store], n_classes=H.N_CLASSES, augmentation=AugmentationConfig(), train=False, device="auto")
    o_cpu = a_cpu(d, affine=A)
    o_gpu = a_gpu({k: v.cuda() for k, v in d.items()}, affine=A.cuda())
    assert o_gpu["input_data"].is_cuda
    for k in ("input_data", "target_data"):
        np.testing.assert_allclose(o_gpu[k].cpu().numpy(), o_cpu[k].numpy(), atol=1e-4, err_msg=k)
    assert (o_gpu["negative_mask"].cpu() == o_cpu["negative_mask"]).float().mean() > 0.999
