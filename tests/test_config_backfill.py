"""Old checkpoints must stay loadable as the config schema grows.

Checkpoints pickle the whole ``MainConfig``. Unpickling a dataclass restores
``__dict__`` directly and never calls ``__init__``, so a checkpoint written
before a field existed yields an instance of the *current* class that is simply
missing that attribute. Anything walking the field list with a bare ``getattr``
then raises ``AttributeError``.

This is what broke when ``AugmentationConfig`` was added, and it would break
again for the next field added to ``TrainConfig``, so these tests are written
against the mechanism rather than against ``augmentation`` specifically.
"""

import copy

import pytest

from tomocpt.configManager.configManager import (
    backfill_missing_fields,
    update_config,
    update_config_with_changed_values,
)
from tomocpt.defaultConfigs.train_config import AugmentationConfig
from tomocpt.mainConfig import MainConfig


@pytest.fixture
def legacy_config():
    """A config as an older checkpoint would unpickle it: no ``augmentation``."""
    config = MainConfig()
    delattr(config.train, "augmentation")
    assert not hasattr(config.train, "augmentation")
    return config


def test_backfill_restores_a_missing_nested_field(legacy_config):
    """A field absent from the checkpoint gets the current default."""
    backfill_missing_fields(legacy_config, MainConfig())

    assert hasattr(legacy_config.train, "augmentation")
    assert legacy_config.train.augmentation == AugmentationConfig()


def test_backfill_does_not_overwrite_values_the_checkpoint_does_have():
    """Only missing fields are filled; everything else is left alone."""
    config = MainConfig()
    config.train.n_epochs = 999
    config.train.augmentation.use_mw_aug = True

    backfill_missing_fields(config, MainConfig())

    assert config.train.n_epochs == 999
    assert config.train.augmentation.use_mw_aug is True


def test_backfill_reaches_fields_missing_inside_a_nested_dataclass():
    """A field added to an existing nested config is backfilled too."""
    config = MainConfig()
    delattr(config.train.augmentation, "use_mw_aug")

    backfill_missing_fields(config, MainConfig())

    assert config.train.augmentation.use_mw_aug is False


def test_backfilled_copies_are_independent():
    """Backfilled defaults must not alias the defaults object."""
    config = MainConfig()
    delattr(config.train, "augmentation")
    defaults = MainConfig()

    backfill_missing_fields(config, defaults)
    config.train.augmentation.use_mw_aug = True

    assert defaults.train.augmentation.use_mw_aug is False


def test_update_config_tolerates_a_source_missing_a_field(legacy_config):
    """``update_config`` keeps the target's value when the source lacks the field.

    This is the call that crashed outright: ``BaseModel.__init__`` merges the
    checkpoint's config into the live one on every load.
    """
    live = MainConfig()
    live.train.augmentation.use_mw_aug = True

    update_config(live.train, source=legacy_config.train)

    assert live.train.augmentation.use_mw_aug is True


def test_cli_overrides_of_new_fields_apply_to_an_old_checkpoint(legacy_config):
    """Setting a new option on the CLI must work against an old checkpoint.

    The second failure mode: ``update_config_with_changed_values`` walks the
    difference path ``train.augmentation.<field>`` into the checkpoint's config.
    """
    defaults = MainConfig()
    after_cli = copy.deepcopy(defaults)
    after_cli.train.augmentation.use_mw_aug = True
    after_cli.train.augmentation.missing_wedge_prob = 0.7

    backfill_missing_fields(legacy_config, defaults)
    result = update_config_with_changed_values(
        target=legacy_config, originaConfig=defaults, configAfterCli=after_cli
    )

    assert result.train.augmentation.use_mw_aug is True
    assert result.train.augmentation.missing_wedge_prob == 0.7


def test_backfill_is_a_no_op_for_a_current_config():
    """A config written by the current code is returned unchanged."""
    config = MainConfig()
    before = copy.deepcopy(config)

    backfill_missing_fields(config, MainConfig())

    assert config == before
