# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from spectrochempy.plugins.base import SpectroChemPyPlugin
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.manager import PluginManager


def test_plugin_manager_singleton():
    pm = PluginManager()
    assert pm is not None
    assert pm._discovered is False


def test_discover_plugins():
    pm = PluginManager()
    pm.discover()
    assert pm._discovered is True
    # At minimum, discovery runs without error and returns a list
    assert isinstance(pm.list_plugins(), list)


def test_available_plugins_empty():
    pm = PluginManager()
    pm.discover()
    # plugins may or may not be installed; nonexistent should always be False
    assert pm.has_plugin("nonexistent") is False


def test_register_dummy_plugin():
    pm = PluginManager()
    plugin = DummyPlugin()
    pm.register(plugin)
    assert pm.has_plugin("dummy")


def test_get_plugin():
    pm = PluginManager()
    plugin = DummyPlugin()
    pm.register(plugin)
    assert pm.get_plugin("dummy") is plugin


class DummyPlugin:
    name = "dummy"
    version = "0.1.0"
    api_version = "1.0"

    def register(self, registry):
        registry.register_reader("dummy", lambda x: x)


class TestPluginProtocol:
    def test_protocol_check(self):
        assert isinstance(DummyPlugin(), SpectroChemPyPlugin)

    def test_protocol_without_name(self):
        class InvalidPlugin:
            pass

        assert not isinstance(InvalidPlugin(), SpectroChemPyPlugin)


class TestMissingPluginError:
    def test_default_message(self):
        err = MissingPluginError("topspin reader")
        msg = str(err)
        assert "topspin reader" in msg
        assert "spectrochempy-nmr" in msg

    def test_custom_hint(self):
        err = MissingPluginError(
            "test", plugin_name="mypkg", install_hint="pip install mypkg"
        )
        assert "pip install mypkg" in str(err)
