from dataclasses import dataclass, field, fields
from typing import Any
from omegaconf import MISSING

from tomocpt.defaultConfigs.general_config import GlobalPropertyConfig
from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
from tomocpt.defaultConfigs.train_config import TrainConfig

@dataclass
class MainConfig:
    prepData: PrepdataConfig = field(default_factory=PrepdataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    shared: GlobalPropertyConfig = field(default_factory=GlobalPropertyConfig)

    def __post_init__(self):
        # Link shared configs
        for config in [self.prepData, self.train, self.infer]:
            for f in fields(self.shared):
                if hasattr(config, f.name):
                    if getattr(config, f.name) is None or getattr(config, f.name) == MISSING:
                        setattr(config, f.name, getattr(self.shared, f.name))

_mainConfigNoChanges = MainConfig()
mainConfig = MainConfig()
