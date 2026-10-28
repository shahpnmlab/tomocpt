"""G5-C10: SSLConfig version rules.

Where the ``None`` defaults are resolved is left to the implementer: at construction, or in a
``resolve()`` (or ``resolved()``) method that returns the resolved config (README.md).
"""

import pytest

from gate5 import g5_helpers as H
from utils.tiny import applied_main_config


def test_G5_C10_defaults_reproduce_v1():
    from tomocpt.mainConfig import MainConfig

    cfg = H.resolved(H.ssl_cfg())
    assert cfg.version == 1
    assert cfg.use_rotation is True
    assert cfg.recon_visible_weight == pytest.approx(1.0)
    assert MainConfig().train.ssl.version == 1


def test_G5_C10_version_2_alone_resolves():
    cfg = H.resolved(H.ssl_cfg(version=2))
    assert cfg.use_rotation is False
    assert cfg.recon_visible_weight == pytest.approx(0.1)


def test_G5_C10_version_2_with_rotation_raises(tmp_path):
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel

    try:
        cfg = H.resolved(H.ssl_cfg(version=2, use_rotation=True))
    except ValueError:
        return
    main = H.ssl_v2_main_config(tmp_path)
    main.train.ssl = cfg
    with applied_main_config(main):
        with pytest.raises(ValueError):
            SelfSupervisedModel(lr=1e-3, train_continue=None)


def test_G5_C10_version_set_after_construction_builds_a_v2_model(tmp_path):
    """``train.ssl.version=2`` given as an override on a default config gives a v2 model."""
    from tomocpt.networks.selfSupervisedModel import SelfSupervisedModel
    from tomocpt.networks.swinunetr import ScaleConditionedSwinUNETR

    main = H.ssl_v2_main_config(tmp_path)
    main.train.ssl = H.ssl_cfg()
    main.train.ssl.version = 2
    with applied_main_config(main):
        model = SelfSupervisedModel(lr=1e-3, train_continue=None)
    assert getattr(model, "rotation_head", None) is None
    assert isinstance(model.model, ScaleConditionedSwinUNETR)
