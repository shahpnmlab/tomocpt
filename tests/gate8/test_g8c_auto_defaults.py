"""Gate 8 defaults (human decisions 14 Nov 2026).

* G8c-D1 ``train.sampling = auto`` (default): on_the_fly for pipeline v2 picking, chunks for
  pipeline v1, SSL and a chunks_dir that already holds chunk files; explicit values unchanged.
* G8c-D2 ``train.otf.augment_device = auto`` (default): cuda only when training on a GPU with
  every enabled augmentation ported, else cpu.
* G8-S1 Swin checkpointing off when memory allows (``training.swin_memory``): the switch reaches
  every MONAI Swin module, keeps the output and gradients identical, and the memory rule.
"""

import pytest
import torch


def _train(tmp_path, version=2, mode="picking", sampling="auto"):
    from utils.tiny import tiny_main_config

    cfg = tiny_main_config(tmp_path)
    cfg.train.data_pipeline_version = version
    cfg.train.mode = mode
    cfg.train.sampling = sampling
    cfg.train.chunks_dir = str(tmp_path / "chunks")
    return cfg.train


def _prepared(tmp_path, storage=None):
    from tomocpt.dataManager.chunking import get_chunking_name_done
    from tomocpt.dataManager.metadata import write_chunk_layout

    d = tmp_path / "chunks"
    d.mkdir(exist_ok=True)
    info = {"layout": 2, "pipeline_version": 2}
    if storage:
        info["storage"] = storage
    write_chunk_layout(d, info)
    open(get_chunking_name_done(str(d), require_labels=True), "w").close()


def test_G8c_D1_defaults_are_auto():
    from tomocpt.defaultConfigs.otf_config import OnTheFlyConfig
    from tomocpt.defaultConfigs.train_config import TrainConfig

    assert TrainConfig().sampling == "auto"
    assert OnTheFlyConfig().augment_device == "auto"


@pytest.mark.parametrize("version,mode,expected", [(2, "picking", "on_the_fly"), (1, "picking", "chunks"),
                                                    (1, "selfSupervised", "chunks"),
                                                    (2, "selfSupervised", "chunks")])
def test_G8c_D1_auto_resolution(tmp_path, version, mode, expected):
    from tomocpt.training.helpers import _requested_storage, resolve_sampling

    t = _train(tmp_path, version, mode)
    assert resolve_sampling(t) == expected and t.sampling == expected
    assert _requested_storage(_train(tmp_path, version, mode)) == ("tomogram" if expected == "on_the_fly"
                                                                  else "chunks")


@pytest.mark.parametrize("storage,expected", [(None, "chunks"), ("chunks", "chunks"), ("tomogram", "on_the_fly")])
def test_G8c_D1_auto_reuses_existing_data(tmp_path, storage, expected):
    from tomocpt.training.helpers import resolve_sampling

    _prepared(tmp_path, storage)
    assert resolve_sampling(_train(tmp_path)) == expected


@pytest.mark.parametrize("explicit", ["chunks", "on_the_fly"])
def test_G8c_D1_explicit_unchanged(tmp_path, explicit):
    from tomocpt.training.helpers import resolve_sampling

    _prepared(tmp_path, "tomogram" if explicit == "chunks" else None)  # opposite data: still no override
    t = _train(tmp_path, sampling=explicit)
    assert resolve_sampling(t) == explicit and t.sampling == explicit


def test_G8c_D1_bad_value(tmp_path):
    from tomocpt.training.helpers import resolve_sampling

    with pytest.raises(ValueError, match="train.sampling"):
        resolve_sampling(_train(tmp_path, sampling="sometimes"))


def test_G8c_D2_augment_device(monkeypatch):
    from tomocpt.defaultConfigs.train_config import AugmentationConfig
    from tomocpt.dataManager.otf import resolve_augment_device

    aug = AugmentationConfig()
    assert resolve_augment_device("cpu", aug, True) == "cpu"
    assert resolve_augment_device("cuda", aug, False) == "cuda"
    assert resolve_augment_device("auto", aug, False) == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_augment_device("auto", aug, True) == "cuda"
    aug.noise_p = 0.5  # GPU port since 17 Nov 2026 (re-frozen, human-approved dispute)
    assert resolve_augment_device("auto", aug, True) == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_augment_device("auto", AugmentationConfig(), True) == "cpu"
    with pytest.raises(ValueError, match="augment_device"):
        resolve_augment_device("tpu", aug, True)


def test_G8_S1_switch_is_exact():
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # bitwise reductions, as G8b-C2
    try:
        _switch_is_exact()
    finally:
        torch.set_num_threads(n_threads)


def _switch_is_exact():
    from tomocpt.networks.swinunetr import MySwinUNETR
    from tomocpt.training.swin_memory import set_swin_checkpoint, swin_modules
    from utils.tiny import tiny_swin_kwargs

    torch.manual_seed(0)
    m = MySwinUNETR(**{**tiny_swin_kwargs(), "use_checkpoint": True}).train()
    assert len(swin_modules(m)) >= 8 and all(x.use_checkpoint for x in swin_modules(m))
    x = torch.randn(1, 1, 64, 64, 64)
    out = []
    for on in (True, False):
        assert set_swin_checkpoint(m, on) == len(swin_modules(m))
        m.zero_grad()
        y = m(x)[0]
        y.square().mean().backward()
        out.append((y.detach().clone(), [p.grad.clone() for p in m.parameters() if p.grad is not None]))
    assert not any(x.use_checkpoint for x in swin_modules(m))
    assert torch.equal(out[0][0], out[1][0])
    assert all(torch.equal(a, b) for a, b in zip(out[0][1], out[1][1]))


def test_G8_S1_memory_rule():
    from tomocpt.training.swin_memory import extra_gb, fits_without_checkpoint

    e = extra_gb(16, 36, 64, "bf16-mixed")
    assert e == pytest.approx(0.47 * 16 * 1.2)
    assert extra_gb(16, 36, 64, "32") == pytest.approx(2 * e)
    assert extra_gb(16, 72, 64, "bf16-mixed") == pytest.approx(2 * e)
    assert fits_without_checkpoint(9.9, e, 48.0)          # A6000, CZII B=16 (measured 17.4 GB off)
    assert fits_without_checkpoint(9.9, e, 24.0)          # 24 GB: 18.9 <= 20.4
    assert not fits_without_checkpoint(9.9, e, 16.0)      # 16 GB GPU: keep on


def test_G8_S1_callback_respects_explicit_off():
    from types import SimpleNamespace

    from tomocpt.networks.swinunetr import MySwinUNETR
    from tomocpt.training.swin_memory import SwinCheckpointAuto, swin_modules
    from utils.tiny import tiny_swin_kwargs

    m = MySwinUNETR(**tiny_swin_kwargs())  # use_checkpoint=False
    cb = SwinCheckpointAuto(36, 64, "bf16-mixed", 16)
    cb.on_train_batch_end(SimpleNamespace(), SimpleNamespace(device=torch.device("cpu"), modules=m.modules),
                          None, {}, 0)
    assert cb.decided and not any(x.use_checkpoint for x in swin_modules(m))
