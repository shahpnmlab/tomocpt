import hydra
from omegaconf import OmegaConf

from tomocpt.config import MainConfig


@hydra.main(version_base=None, config_name="main", config_path=None)#"config", config_path="../config")
def my_app(cfg: MainConfig) -> None:

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    print(cfg)
    print(cfg.run.BATCH_SIZE)

if __name__ == "__main__":
    my_app()

    """
    
python -m tomocpt.main run.MODEL_PATH=kk net.KERNEL_SIZE=2
python -m tomocpt.main  --config-dir /tmp/model/other_config --config-name mainConf

    """