# test_config_manager.py
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List
import sys
from unittest.mock import patch
from io import StringIO

import pytest
import typer
from omegaconf import DictConfig, OmegaConf

from tomocpt.configManager import create_configurable_app, MergePreference


# Test configurations
@dataclass
class InnerTestConfig:
    nested_value: str = "default"
    number: int = 10


@dataclass
class TestConfig:
    value1: int = 42
    value2: float = 0.001
    inner: InnerTestConfig = field(default_factory=InnerTestConfig)


# Helper function to capture both stdout and stderr and run command
def run_with_args(func, args):
    """Run the function with given args and capture both stdout and stderr"""
    stdout = StringIO()
    stderr = StringIO()
    with patch('sys.stdout', stdout), \
            patch('sys.stderr', stderr), \
            patch('sys.argv', ['script.py'] + args):
        try:
            config_app.run_with_config(func)
            return 0, stdout.getvalue(), stderr.getvalue()
        except SystemExit as e:
            return e.code, stdout.getvalue(), stderr.getvalue()


# Fixtures
@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing"""
    content = """
value1: 100
value2: 0.5
inner:
  nested_value: "from_file"
  number: 20
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(content)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def conflicting_config_file():
    """Create a config file that conflicts with typical command line args"""
    content = """
value1: 999
value2: 0.999
inner:
  nested_value: "conflict"
  number: 999
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(content)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


# Tests
def test_basic_execution():
    """Test basic execution without any config options"""
    result_dict = {}

    def test_func(
            input_path: Annotated[Path, typer.Option(help="Input path")],
            config: DictConfig = None
    ):
        result_dict['input_path'] = str(input_path)
        result_dict['config'] = OmegaConf.to_container(config)

    exit_code, stdout, stderr = run_with_args(test_func, ['--input-path', 'input.txt'])

    assert exit_code == 0
    assert result_dict['input_path'] == 'input.txt'
    assert result_dict['config']['value1'] == 42  # default value
    assert result_dict['config']['value2'] == 0.001  # default value


def test_config_file(temp_config_file):
    """Test loading configuration from file"""
    result_dict = {}

    def test_func(
            input_path: Annotated[Path, typer.Option(help="Input path")],
            config: DictConfig = None
    ):
        result_dict['config'] = OmegaConf.to_container(config)

    exit_code, stdout, stderr = run_with_args(test_func, [
        '--input-path', 'input.txt',
        '--config-file', temp_config_file
    ])

    assert exit_code == 0
    assert result_dict['config']['value1'] == 100
    assert result_dict['config']['value2'] == 0.5
    assert result_dict['config']['inner']['nested_value'] == 'from_file'


def test_command_line_override():
    """Test command line overrides"""
    result_dict = {}

    def test_func(
            input_path: Annotated[Path, typer.Option(help="Input path")],
            config: DictConfig = None
    ):
        result_dict['config'] = OmegaConf.to_container(config)

    exit_code, stdout, stderr = run_with_args(test_func, [
        '--input-path', 'input.txt',
        'value1=200',
        'inner.nested_value=from_cli'
    ])

    assert exit_code == 0
    assert result_dict['config']['value1'] == 200
    assert result_dict['config']['inner']['nested_value'] == 'from_cli'


def test_multiple_overrides():
    """Test multiple command line overrides"""
    result_dict = {}

    def test_func(
            input_path: Annotated[Path, typer.Option(help="Input path")],
            config: DictConfig = None
    ):
        result_dict['config'] = OmegaConf.to_container(config)

    exit_code, stdout, stderr = run_with_args(test_func, [
        '--input-path', 'input.txt',
        'value1=100',
        'value2=0.5',
        'inner.nested_value=new_value',
        'inner.number=42'
    ])

    assert exit_code == 0
    assert result_dict['config']['value1'] == 100
    assert result_dict['config']['value2'] == 0.5
    assert result_dict['config']['inner']['nested_value'] == 'new_value'
    assert result_dict['config']['inner']['number'] == 42


def test_conflict_detection(conflicting_config_file):
    """Test conflict detection between file and command line args"""
    result_dict = {}

    def test_func(
            input_path: Annotated[Path, typer.Option(help="Input path")],
            config: DictConfig = None
    ):
        result_dict['config'] = OmegaConf.to_container(config)

    exit_code, stdout, stderr = run_with_args(test_func, [
        '--input-path', 'input.txt',
        '--config-file', conflicting_config_file,
        'value1=200'
    ])

    assert exit_code != 0
    assert "Conflicts found between config file and command arguments" in (stdout + stderr)


def test_invalid_config_file():
    """Test handling of non-existent config file"""

    def test_func(
            input_path: Annotated[Path, typer.Option(help="Input path")],
            config: DictConfig = None
    ):
        pass

    exit_code, stdout, stderr = run_with_args(test_func, [
        '--input-path', 'input.txt',
        '--config-file', 'nonexistent.yaml'
    ])

    assert exit_code != 0
    assert "Config file not found" in (stdout + stderr)


def test_type_conversion():
    """Test proper type conversion in config values"""
    result_dict = {}

    def test_func(
            input_path: Annotated[Path, typer.Option(help="Input path")],
            config: DictConfig = None
    ):
        result_dict['config'] = OmegaConf.to_container(config)
        result_dict['raw_config'] = config

    exit_code, stdout, stderr = run_with_args(test_func, [
        '--input-path', 'input.txt',
        'value1=123',
        'value2=0.123'
    ])

    assert exit_code == 0
    assert isinstance(result_dict['raw_config'].value1, int)
    assert isinstance(result_dict['raw_config'].value2, float)
    assert result_dict['config']['value1'] == 123
    assert result_dict['config']['value2'] == 0.123


def test_argument_validation():
    """Test that using typer.Argument raises an error"""
    with pytest.raises(typer.BadParameter):
        def test_func(
                input_path: Annotated[Path, typer.Argument(help="Input path")],  # This should raise an error
                config: DictConfig = None
        ):
            pass

        config_app.run_with_config(test_func)


# Initialize config_app for all tests
config_app = create_configurable_app(TestConfig)