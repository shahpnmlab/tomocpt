"""G1-C7 (added after review): backfill must see fields missing from the *instance*.

An unpickled old checkpoint config lacks newer fields in its ``__dict__``. For a field
with a plain default, the class attribute still exists, so ``hasattr`` is true and plain
``getattr`` silently returns the class default. ``backfill_missing_fields`` must still
fill such a field, so that ``safe_getattr`` and plain ``getattr`` agree afterwards.

The sweep covers every field of the current ``MainConfig`` tree, so fields added by later
gates are checked automatically.
"""

from dataclasses import fields, is_dataclass

import pytest
import torch

from tomocpt.configManager.configManager import (
    backfill_missing_fields,
    safe_getattr,
    update_config,
)
from tomocpt.mainConfig import MainConfig

_ABSENT = object()


def _field_paths(obj, prefix=()):
    for f in fields(obj):
        yield prefix + (f.name,)
        value = getattr(obj, f.name)
        if is_dataclass(value):
            yield from _field_paths(value, prefix + (f.name,))


def _resolve(obj, path):
    for name in path:
        obj = getattr(obj, name)
    return obj


ALL_PATHS = list(_field_paths(MainConfig()))


@pytest.mark.parametrize("path", ALL_PATHS, ids=[".".join(p) for p in ALL_PATHS])
def test_G1_C7_backfill_fills_every_field_missing_from_the_instance(path):
    config, defaults = MainConfig(), MainConfig()
    parent = _resolve(config, path[:-1])
    del parent.__dict__[path[-1]]
    assert safe_getattr(parent, path[-1], _ABSENT) is _ABSENT

    backfill_missing_fields(config, defaults)

    assert path[-1] in parent.__dict__
    want = _resolve(defaults, path)
    got = safe_getattr(parent, path[-1], _ABSENT)
    assert got == want
    assert got == getattr(parent, path[-1])
    if is_dataclass(want):
        assert got is not want  # a copy, not an alias of the defaults object


def test_G1_C7_backfill_uses_the_linked_shared_value_not_the_class_default():
    # Section fields such as train.chunks_dir default to None on the class and are
    # filled from `shared` in MainConfig.__post_init__; backfill must use the latter.
    defaults = MainConfig()
    defaults.shared.chunks_dir = "/from/shared"
    defaults.train.chunks_dir = "/from/shared"
    config = MainConfig()
    del config.train.__dict__["chunks_dir"]

    backfill_missing_fields(config, defaults)

    assert config.train.chunks_dir == "/from/shared"


def test_G1_C7_update_config_keeps_target_when_source_instance_lacks_a_plain_field():
    target, source = MainConfig(), MainConfig()
    target.train.batch_size = 7
    del source.train.__dict__["batch_size"]  # class default (16) would leak through getattr

    update_config(target.train, source=source.train)

    assert target.train.batch_size == 7


def test_G1_C7_checkpoint_missing_a_plain_field_is_backfilled_on_load(legacy_ckpt, tmp_path):
    from tomocpt.mainConfig import _mainConfigNoChanges
    from tomocpt.networks.pickingModel import BasePickingModel

    ckpt = torch.load(legacy_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["config"]
    del cfg.train.__dict__["n_epochs"]
    del cfg.train.augmentation.__dict__["use_mw_aug"]
    stripped = tmp_path / "stripped.ckpt"
    torch.save(ckpt, stripped)

    model = BasePickingModel.load_from_checkpoint(stripped, map_location="cpu")
    loaded = model.hparams.config

    assert "n_epochs" in loaded.train.__dict__
    assert safe_getattr(loaded.train, "n_epochs", _ABSENT) == _mainConfigNoChanges.train.n_epochs
    assert safe_getattr(loaded.train.augmentation, "use_mw_aug", _ABSENT) is False
