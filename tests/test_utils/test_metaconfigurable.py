# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for metaconfigurable module."""

import tempfile

import pytest
import traitlets as tr
from traitlets.config import Config
from traitlets.config.configurable import Configurable

from spectrochempy.utils.metaconfigurable import MetaConfigurable


class MockConfigManager:
    """Mock config manager for testing."""

    def __init__(self):
        self.config_dir = tempfile.mkdtemp()
        self.configs = {}

    def update(self, name, config):
        """Mock update method that stores configs in memory."""
        if name not in self.configs:
            self.configs[name] = {}
        self.configs[name].update(config)


class MockParent(Configurable):
    """Mock parent with config and config_manager."""

    def __init__(self):
        self.config_manager = MockConfigManager()
        self.config = Config()
        super().__init__()  # Initialize the parent Configurable class


class A_Configurable(MetaConfigurable):
    """Test implementation of MetaConfigurable."""

    name = tr.Unicode("A_Configurable")
    test_param = tr.Int(42, config=True, help="Test parameter")
    another_param = tr.Unicode("default", config=True, help="Another test parameter")


@pytest.fixture
def test_configurable():
    """Create a test configurable instance."""
    parent = MockParent()
    return A_Configurable(parent=parent)


def test_init():
    """Test initialization of MetaConfigurable."""
    parent = MockParent()
    test_config = {"A_Configurable": {"test_param": 100}}
    parent.config = Config(test_config)

    tc = A_Configurable(parent=parent)

    assert tc.name == "A_Configurable"
    assert tc.test_param == 100  # Should use config value
    assert tc.another_param == "default"  # Should use default value


def test_to_dict(test_configurable):
    """Test to_dict method."""
    config_dict = test_configurable.to_dict()

    assert "test_param" in config_dict
    assert config_dict["test_param"] == 42
    assert "another_param" in config_dict
    assert config_dict["another_param"] == "default"


def test_params(test_configurable):
    """Test params method."""
    # Test current params
    params = test_configurable.params()
    assert params.test_param == 42
    assert params.another_param == "default"

    # Change a parameter
    test_configurable.test_param = 100
    params = test_configurable.params()
    assert params.test_param == 100

    # Test default params
    default_params = test_configurable.params(default=True)
    assert default_params.test_param == 42


def test_parameters_deprecated(test_configurable):
    """Test that parameters method is deprecated but works."""
    with pytest.warns(DeprecationWarning):
        params = test_configurable.parameters()

    assert params.test_param == 42
    assert params.another_param == "default"


def test_reset(test_configurable):
    """Test reset method."""
    test_configurable.test_param = 100
    test_configurable.another_param = "changed"

    assert test_configurable.test_param == 100
    assert test_configurable.another_param == "changed"

    test_configurable.reset()

    assert test_configurable.test_param == 42
    assert test_configurable.another_param == "default"


def test_trait_change_updates_config(test_configurable):
    """Test that changing a trait updates the config."""
    test_configurable.test_param = 100

    # Verify that update was called on config_manager
    assert "A_Configurable" in test_configurable.cfg.configs
    assert "A_Configurable" in test_configurable.cfg.configs["A_Configurable"]
    assert (
        "test_param"
        in test_configurable.cfg.configs["A_Configurable"]["A_Configurable"]
    )
    assert (
        test_configurable.cfg.configs["A_Configurable"]["A_Configurable"]["test_param"]
        == 100
    )
