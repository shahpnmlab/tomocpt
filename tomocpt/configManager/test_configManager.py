import pytest
from dataclasses import dataclass, field
from typing import Annotated, Optional
import typer
import subprocess
import sys
from pathlib import Path
import tempfile
import json

@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.Adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)

@dataclass
class TrainConfig:
    chunks_dir: Annotated[Optional[str], typer.Option(help="Path to chunks")] = None
    model_dir: Annotated[Optional[str], typer.Option(help="Path to save model")] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    n_epochs: Annotated[int, typer.Option(help="Number of epochs")] = 10

@dataclass
class MainConfig:
    train: TrainConfig = field(default_factory=TrainConfig)

@pytest.fixture
def script_path(tmp_path):
        script_content = '''
    from tomocpt.configManager.configManager import create_app
    from dataclasses import dataclass, field
    from typing import Annotated, Optional, Tuple
    import typer
    import json
    
    @dataclass
    class OptimizerConfig:
        _target_: str = "torch.optim.Adam"
        lr: float = 0.001
        weight_decay: float = 0.0
        betas: tuple = (0.9, 0.999)
    
    @dataclass
    class TrainConfig:
        chunks_dir: Annotated[Optional[str], typer.Option(help="Path to chunks")] = None
        model_dir: Annotated[Optional[str], typer.Option(help="Path to save model")] = None
        optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
        n_epochs: Annotated[int, typer.Option(help="Number of epochs")] = 10
    
    @dataclass
    class MainConfig:
        train: TrainConfig = field(default_factory=TrainConfig)
    
    app = create_app()
    config = MainConfig()
    
    @app.command(config=config)
    def main():
        # Print config as JSON for easy parsing in tests
        from dataclasses import asdict
        print(json.dumps(asdict(config)))
    
    if __name__ == "__main__":
        app.run()
    '''     path = tmp_path / "test_script.py"
    path.write_text(script_content)
    return path

@pytest.fixture
def yaml_config_file(tmp_path):
    content = """
train:
  chunks_dir: "data/chunks"
  model_dir: "models/test"
  optimizer:
    _target_: "torch.optim.SGD"
    lr: 0.01
    weight_decay: 0.1
  n_epochs: 20
"""
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path

def run_script(script_path, args):
    result = subprocess.run(
        [sys.executable, str(script_path)] + args,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {result.stderr}")
    print(result.stdout)
    return json.loads(result.stdout.strip().split("\n")[-1])

def test_basic_cli(script_path):
    result = run_script(script_path, [
        "--train.chunks_dir", "data/test",
        "--train.n_epochs", "30"
    ])
    assert result["train"]["chunks_dir"] == "data/test"
    assert result["train"]["n_epochs"] == 30

def test_yaml_config(script_path, yaml_config_file):
    result = run_script(script_path, ["--config_file", str(yaml_config_file), "--config_merge_preference", "configFile"])
    assert result["train"]["chunks_dir"] == "data/chunks"
    assert result["train"]["optimizer"]["_target_"] == "torch.optim.SGD"

def test_hydra_style_override(script_path):
    result = run_script(script_path, [
        "train.optimizer.lr=0.1",
        "train.optimizer._target_=torch.optim.SGD"
    ])
    assert result["train"]["optimizer"]["lr"] == 0.1
    assert result["train"]["optimizer"]["_target_"] == "torch.optim.SGD"

def test_merge_preference(script_path, yaml_config_file):
    # Test CLI preference
    result = run_script(script_path, [
        "--config_file", str(yaml_config_file),
        "--config_merge_preference", "command",
        "--train.n_epochs", "50",
        "train.optimizer.lr=0.2"
    ])
    assert result["train"]["n_epochs"] == 50
    assert result["train"]["optimizer"]["lr"] == 0.2

    # Test YAML preference
    result = run_script(script_path, [
        "--config_file", str(yaml_config_file),
        "--config_merge_preference", "configFile",
        "--train.n_epochs", "50",
        "train.optimizer.lr=0.2"
    ])
    assert result["train"]["n_epochs"] == 20
    assert result["train"]["optimizer"]["lr"] == 0.01

def test_invalid_path(script_path):
    #TODO: Fix this one!
    with pytest.raises(RuntimeError):
        run_script(script_path, ["invalid.path=123"])



def test_missing_required(script_path, yaml_config_file):
    with pytest.raises(RuntimeError):
        run_script(script_path, ["--train.chunks_dir"])

def test_mixed_cli_and_hydra(script_path):
    result = run_script(script_path, [
        "--train.chunks_dir", "data/test",
        "train.optimizer.lr=0.1",
        "--train.n_epochs", "30"
    ])
    assert result["train"]["chunks_dir"] == "data/test"
    assert result["train"]["optimizer"]["lr"] == 0.1
    assert result["train"]["n_epochs"] == 30