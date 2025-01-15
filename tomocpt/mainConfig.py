from dataclasses import dataclass, field, fields
from typing import Any

from tomocpt.defaultConfigs.general_config import GlobalPropertyConfig
from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
from tomocpt.defaultConfigs.train_config import TrainConfig

def _create_linked_config_class(primary_class):
    # Get the fields from the primary class
    primary_fields = {field.name: field for field in fields(primary_class)}

    # Add shared field to annotations
    annotations = {**primary_class.__annotations__}
    annotations['shared'] = Any

    # Create class with both primary fields and shared
    @dataclass
    class LinkedConfig(primary_class):
        shared: Any = None

        def __init__(self, primary, shared):
            self._primary = primary
            self._shared = shared

        def __getattr__(self, name):
            return getattr(self._primary, name)

        def __setattr__(self, name, value):
            if name in ('_primary', '_shared', 'shared'):
                super().__setattr__(name, value)
            else:
                setattr(self._primary, name, value)

        @property
        def shared(self):
            return self._shared

    # Copy over all annotations and field metadata
    LinkedConfig.__annotations__ = annotations

    return LinkedConfig

def create_linked_config(primary_config, shared_config):
    LinkedConfig = _create_linked_config_class(type(primary_config))
    config_for_ = LinkedConfig(primary_config, shared_config)
    return config_for_


@dataclass
class MainConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    prepData: PrepdataConfig = field(default_factory=PrepdataConfig)
    shared: GlobalPropertyConfig = field(default_factory=GlobalPropertyConfig)

_mainConfigNoChanges = MainConfig()
mainConfig = MainConfig()


