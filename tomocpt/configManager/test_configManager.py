import pytest
from dataclasses import dataclass, field
from typing import Annotated, List, Optional, Any
import typer
from pathlib import Path
import tempfile
from omegaconf import OmegaConf, DictConfig

from tomocpt.configManager.configManager import create_app, MergePreference


# Test Configurations
@dataclass
class TrainConfig:
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 0.001
    epochs: Annotated[int, typer.Option(help="Number of epochs")] = 10
    advanced_parameter: int = -1


@dataclass
class MainConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment_name: Annotated[str, typer.Option(help="experiment_name")] = "default"
    otro: int = 9


@pytest.fixture
def yaml_config_file():
    content = """
train:
  learning_rate: 0.999999
  epochs: 32
  advanced_parameter: 100
experiment_name: "from_yaml"
otro: 42
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(content)
    yield Path(f.name)
    Path(f.name).unlink()


def test_basic_cli():
    """Test basic CLI parameter handling"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit) as exit_info:
        app.app(["--train.learning-rate", "0.1"])

        assert result.get('config') is not None
        if result.get('config'):
            assert result['config'].train.learning_rate == 0.1
            assert result['config'].train.epochs == 10
            assert result['config'].experiment_name == "default"


def test_help_shows_parameters(capsys):
    """Test that help shows the correct parameters"""
    app = create_app()
    config = MainConfig()

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        pass

    with pytest.raises(SystemExit):
        app.app(["--help"])

    captured = capsys.readouterr()
    assert "--train.learning-rate" in captured.out
    assert "--train.epochs" in captured.out
    assert "--experiment-name" in captured.out
    assert "advanced_parameter" not in captured.out


def test_yaml_config_loading(yaml_config_file):
    """Test loading configuration from YAML file"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit):
        app.app(["--config_file", str(yaml_config_file)])

        assert result['config'].train.learning_rate == 0.999999
        assert result['config'].train.epochs == 32
        assert result['config'].train.advanced_parameter == 100
        assert result['config'].experiment_name == "from_yaml"
        assert result['config'].otro == 42


def test_hydra_style_updates():
    """Test updating values through hydra-style arguments"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit):
        app.app(["train.advanced_parameter=100", "otro=42"])

        assert result['config'].train.advanced_parameter == 100
        assert result['config'].otro == 42
        # Annotated params should keep defaults
        assert result['config'].train.learning_rate == 0.001
        assert result['config'].train.epochs == 10


def test_merge_preference(yaml_config_file):
    """Test config merge preferences"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    # Test CLI preference
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "command",
            "--train.learning-rate", "0.1",
            "train.advanced_parameter=200"
        ])

        assert result['config'].train.learning_rate == 0.1
        assert result['config'].train.advanced_parameter == 200

    # Test YAML preference
    result = {}
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "configFile",
            "--train.learning-rate", "0.1",
            "train.advanced_parameter=200"
        ])

        assert result['config'].train.learning_rate == 0.999999
        assert result['config'].train.advanced_parameter == 100


def test_invalid_values():
    """Test handling of invalid parameter values"""
    app = create_app()
    config = MainConfig()

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        pass

    with pytest.raises(SystemExit):
        app.app(["--train.learning-rate", "not_a_float"])

    with pytest.raises(SystemExit):
        app.app(["--train.epochs", "-10"])  # Assuming negative epochs should fail


def test_missing_yaml_file():
    """Test handling of missing YAML file"""
    app = create_app()
    config = MainConfig()

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        pass

    with pytest.raises(FileNotFoundError):
        app.app(["--config_file", "nonexistent.yaml"])


def test_multiple_cli_parameters():
    """Test setting multiple CLI parameters at once"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit):
        app.app([
            "--train.learning-rate", "0.1",
            "--train.epochs", "20",
            "--experiment-name", "test_run"
        ])

        assert result['config'].train.learning_rate == 0.1
        assert result['config'].train.epochs == 20
        assert result['config'].experiment_name == "test_run"


def test_mixed_cli_and_hydra():
    """Test mixing CLI parameters and hydra-style updates"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit):
        app.app([
            "--train.learning-rate", "0.1",
            "train.advanced_parameter=100",
            "--experiment-name", "test_run",
            "otro=42"
        ])

        assert result['config'].train.learning_rate == 0.1
        assert result['config'].train.advanced_parameter == 100
        assert result['config'].experiment_name == "test_run"
        assert result['config'].otro == 42


def test_override_with_yaml_then_cli(yaml_config_file):
    """Test loading YAML and then overriding with CLI"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--train.learning-rate", "0.1",
            "train.advanced_parameter=200"
        ])

        # Should use CLI values by default
        assert result['config'].train.learning_rate == 0.1
        assert result['config'].train.advanced_parameter == 200
        # Should keep other YAML values
        assert result['config'].train.epochs == 32
        assert result['config'].experiment_name == "from_yaml"


def test_type_conversion():
    """Test type conversion for different parameter types"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit):
        app.app([
            "--train.learning-rate", "1e-4",  # Scientific notation
            "--train.epochs", "0x14",  # Hex number
            "train.advanced_parameter=0o10"  # Octal number
        ])

        assert result['config'].train.learning_rate == 0.0001
        assert result['config'].train.epochs == 20
        assert result['config'].train.advanced_parameter == 8


def test_invalid_yaml_content(tmp_path):
    """Test handling of valid YAML file with invalid content"""
    invalid_config = tmp_path / "invalid_config.yaml"
    invalid_config.write_text("""
train:
    learning_rate: not_a_float
    epochs: true  # bool instead of int
    advanced_parameter: [1, 2, 3]  # list instead of int
""")

    app = create_app()
    config = MainConfig()

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        pass

    with pytest.raises(SystemExit):
        app.app(["--config_file", str(invalid_config)])


def test_empty_yaml_file(tmp_path):
    """Test handling of empty YAML file"""
    empty_config = tmp_path / "empty_config.yaml"
    empty_config.write_text("")

    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    with pytest.raises(SystemExit):
        app.app(["--config_file", str(empty_config)])

        # Should maintain default values
        assert result['config'].train.learning_rate == 0.001
        assert result['config'].train.epochs == 10
        assert result['config'].train.advanced_parameter == -1


def test_additional_cli_args():
    """Test handling CLI arguments not in config"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(
            extra_param: Annotated[str, typer.Option(help="Extra parameter")] = "default",
            configs: Optional[List[Any]] = None
    ):
        result['config'] = configs[0]
        result['extra'] = extra_param
        return result['config']

    with pytest.raises(SystemExit):
        app.app([
            "--train.learning-rate", "0.1",
            "--extra-param", "custom_value"
        ])

        assert result['config'].train.learning_rate == 0.1
        assert result['extra'] == "custom_value"


def test_conflicting_cli_args():
    """Test handling of conflicting CLI arguments"""
    app = create_app()
    config = MainConfig()

    @app.command(config=config)
    def test_func(
            learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 0.01,
            configs: Optional[List[Any]] = None
    ):
        pass

    # Should raise error due to duplicate learning rate parameter
    with pytest.raises((SystemExit, typer.BadParameter)):
        app.app([
            "--train.learning-rate", "0.1",
            "--learning-rate", "0.2"
        ])


def test_nested_yaml_override(yaml_config_file):
    """Test deep overriding of nested config values"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(config: Optional[Any] = None):
        result['config'] = config
        return result['config']

    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "command",
            "train.advanced_parameter=200",
            "--train.epochs", "50"
        ])

        # Check that all levels of nesting are properly handled
        assert result['config'].train.advanced_parameter == 200
        assert result['config'].train.epochs == 50
        assert result['config'].train.learning_rate == 0.999999  # From YAML


def test_merge_preference_conflicts(yaml_config_file):
    """Test merge preference handling when conflicts exist"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    # Test without merge preference specified (should raise error)
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--train.learning-rate", "0.1",
            "--train.epochs", "20"
        ])

    # Test with command preference
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "command",
            "--train.learning-rate", "0.1",
            "--train.epochs", "20"
        ])

        assert result['config'].train.learning_rate == 0.1  # CLI value should win
        assert result['config'].train.epochs == 20  # CLI value should win
        assert result['config'].train.advanced_parameter == 100  # YAML value remains

    # Test with config file preference
    result = {}
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "configFile",
            "--train.learning-rate", "0.1",
            "--train.epochs", "20"
        ])

        assert result['config'].train.learning_rate == 0.999999  # YAML value should win
        assert result['config'].train.epochs == 32  # YAML value should win
        assert result['config'].train.advanced_parameter == 100  # YAML value remains


def test_merge_preference_hydra_style_conflicts(yaml_config_file):
    """Test merge preference handling with hydra-style arguments"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    # Test without merge preference specified (should raise error)
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "train.advanced_parameter=200",
            "otro=42"
        ])

    # Test with command preference
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "command",
            "train.advanced_parameter=200",
            "otro=42"
        ])

        assert result['config'].train.advanced_parameter == 200  # CLI value should win
        assert result['config'].otro == 42  # CLI value should win
        assert result['config'].train.learning_rate == 0.999999  # YAML value remains

    # Test with config file preference
    result = {}
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "configFile",
            "train.advanced_parameter=200",
            "otro=42"
        ])

        assert result['config'].train.advanced_parameter == 100  # YAML value should win
        assert result['config'].otro == 9  # YAML value should win
        assert result['config'].train.learning_rate == 0.999999  # YAML value remains


def test_merge_preference_mixed_conflicts(yaml_config_file):
    """Test merge preference with both CLI and hydra-style arguments"""
    app = create_app()
    config = MainConfig()
    result = {}

    @app.command(config=config)
    def test_func(configs: Optional[List[Any]] = None):
        result['config'] = configs[0]
        return result['config']

    # Test without merge preference specified (should raise error)
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--train.learning-rate", "0.1",
            "train.advanced_parameter=200",
            "--experiment-name", "test",
            "otro=42"
        ])

    # Test with command preference
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "command",
            "--train.learning-rate", "0.1",
            "train.advanced_parameter=200",
            "--experiment-name", "test",
            "otro=42"
        ])

        assert result['config'].train.learning_rate == 0.1  # CLI value should win
        assert result['config'].train.advanced_parameter == 200  # CLI value should win
        assert result['config'].experiment_name == "test"  # CLI value should win
        assert result['config'].otro == 42  # CLI value should win

    # Test with config file preference
    result = {}
    with pytest.raises(SystemExit):
        app.app([
            "--config_file", str(yaml_config_file),
            "--config_merge_preference", "configFile",
            "--train.learning-rate", "0.1",
            "train.advanced_parameter=200",
            "--experiment-name", "test",
            "otro=42"
        ])

        assert result['config'].train.learning_rate == 0.999999  # YAML value should win
        assert result['config'].train.advanced_parameter == 100  # YAML value should win
        assert result['config'].experiment_name == "from_yaml"  # YAML value should win
        assert result['config'].otro == 42  # YAML value should win