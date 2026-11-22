"""Regression: settings a run resolves in place survive a warm start (``fine_tune_from``).

``_prepare_data_if_needed`` resolves ``train.sampling=auto`` (Gate 8c) and the Fourier switches
of ``augmentation.fourier_auto`` (Gate 8 follow-up) in the global ``mainConfig.train``. Loading
the source checkpoint in ``_create_model`` (``BaseModel.__init__``) then copied the checkpoint's
train settings into ``mainConfig.train``, except the YAML/CLI changes. A resolved ``auto`` is not
such a change, so a pipeline-v2 run warm-started from a v1 picker went back to
``sampling=chunks`` (or ``auto``) and the Fourier switches off, and ``data.setup()`` refused the
tomogram store it had just prepared (G4-C9 arm d: "storage mismatch").
"""

import copy

import pytest
import torch

from gate3 import g3_helpers as H
from utils.legacy import legacy_checkpoint
from utils.tiny import applied_main_config, tiny_main_config

CLASSES = ((1, "A", 150.0), (2, "B", 300.0))


@pytest.fixture(autouse=True)
def _clear_snapshot():
    from tomocpt.configManager.configManager import clear_cli_snapshot

    clear_cli_snapshot()
    yield
    clear_cli_snapshot()


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """Legacy picker checkpoints: as written by ``v2-gate-0-base`` (no ``sampling`` field), and
    as written by a later v1 run (``sampling=chunks``, Fourier switches off)."""
    legacy = legacy_checkpoint()
    ckpt = torch.load(legacy, map_location="cpu", weights_only=False)
    train = ckpt["hyper_parameters"]["config"].train
    train.sampling = "chunks"
    train.augmentation.use_mw_aug = False
    train.augmentation.use_fourier_aug = False
    path = tmp_path_factory.mktemp("g8_warm_start") / "v1_chunks.ckpt"
    torch.save(ckpt, path)
    return {"gate0": str(legacy), "v1_chunks": str(path)}


def _warm_start(g, source):
    """The CLI flow of ``train``: snapshot of YAML + CLI, in-place resolution, model creation."""
    from tomocpt.configManager.configManager import set_cli_snapshot
    from tomocpt.training.helpers import _create_model, resolve_fourier_augmentation, resolve_sampling

    g.train.fine_tune_from = source
    g.train.resume_from = None
    g.train.use_distillation = False
    set_cli_snapshot(g)
    resolve_sampling(g.train)
    resolve_fourier_augmentation(g.train)
    before = copy.deepcopy(g.train)
    model, resume, _ = _create_model(g.train)
    return model, resume, before


@pytest.mark.parametrize("source", ["gate0", "v1_chunks"])
@pytest.mark.parametrize("cond", [False, True])
def test_G8_warm_start_v2_keeps_resolved_sampling_and_fourier(tmp_path, sources, source, cond):
    cfg = H.v2_config(tmp_path, CLASSES, cond=cond)
    assert cfg.train.sampling == "auto" and cfg.train.augmentation.fourier_auto
    with applied_main_config(cfg) as g:
        model, resume, before = _warm_start(g, sources[source])
        assert before.sampling == "on_the_fly"
        assert resume is None
        saved = model.hparams.config.train
        for train in (g.train, saved):
            assert train.sampling == "on_the_fly"
            assert train.augmentation.use_mw_aug is True
            assert train.augmentation.use_fourier_aug is True


def test_G8_warm_start_v2_explicit_values_win(tmp_path, sources):
    cfg = H.v2_config(tmp_path, CLASSES, cond=True)
    cfg.train.sampling = "chunks"
    cfg.train.augmentation.fourier_auto = False
    with applied_main_config(cfg) as g:
        model, _, _ = _warm_start(g, sources["v1_chunks"])
        for train in (g.train, model.hparams.config.train):
            assert train.sampling == "chunks"
            assert train.augmentation.use_mw_aug is False
            assert train.augmentation.use_fourier_aug is False


def test_G8_warm_start_v1_keeps_legacy_inheritance(tmp_path, sources):
    """Pipeline v1 resolves ``chunks`` and never touches the Fourier switches: a v1 fine-tune
    still inherits the source checkpoint's augmentation settings (invariant 1)."""
    legacy = torch.load(sources["gate0"], map_location="cpu", weights_only=False)
    legacy["hyper_parameters"]["config"].train.augmentation.use_mw_aug = True
    path = tmp_path / "v1_mw.ckpt"
    torch.save(legacy, path)
    with applied_main_config(tiny_main_config(tmp_path)) as g:
        model, _, _ = _warm_start(g, str(path))
        for train in (g.train, model.hparams.config.train):
            assert train.sampling == "chunks"
            assert train.augmentation.use_mw_aug is True
            assert train.augmentation.use_fourier_aug is False
