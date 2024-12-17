from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from tomocpt.defaultConfigs.general_config import GlobalPropertyConfig
from tomocpt.defaultConfigs.infer_config import InferConfig
from tomocpt.defaultConfigs.prepdata_config import PrepdataConfig
from tomocpt.defaultConfigs.train_config import TrainConfig


@dataclass
class MainConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    prepData: PrepdataConfig = field(default_factory=PrepdataConfig)
    shared: GlobalPropertyConfig = field(default_factory=PrepdataConfig)


# cs = ConfigStore.instance()
# cs.store(name="main", node=MainConfig)
# cs.store(group="train", name="default", node=TrainConfig)
# cs.store(group="infer", name="default", node=InferConfig)

_mainConfigNoChanges = MainConfig()
mainConfig = MainConfig()
