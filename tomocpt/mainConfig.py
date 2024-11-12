from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from tomocpt.defaultConfigs.network_config import NetworkConfig
from tomocpt.defaultConfigs.train_config import TrainConfig


@dataclass
class MainConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)


cs = ConfigStore.instance()
cs.store(name="main", node=MainConfig)
cs.store(group="train", name="default", node=TrainConfig)
cs.store(group="network", name="default", node=NetworkConfig)

mainConfig = MainConfig()
