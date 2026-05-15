# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for PluginManager."""


from spectrochempy.plugins.base import SpectroChemPyPlugin
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.registry import PluginRegistry

# ------------------------------------------------------------------
# Dummy plugins for testing
# ------------------------------------------------------------------


class DummyPlugin:
    name = "dummy"
    version = "0.1.0"
    api_version = "1.0"

    def register(self, registry):
        registry.register_reader("dummy", lambda x: x)


class DeclarativeReaderPlugin:
    name = "decl-reader"
    version = "0.2.0"
    api_version = "1.0"

    def register_readers(self) -> list[dict]:
        return [
            {
                "name": "myformat",
                "func": lambda path: "data",
                "description": "Read MyFormat",
                "extensions": [".myf"],
            }
        ]


class DeclarativeWriterPlugin:
    name = "decl-writer"
    version = "0.2.0"
    api_version = "1.0"

    def register_writers(self) -> list[dict]:
        return [
            {
                "name": "myformat",
                "func": lambda data, path: None,
                "description": "Write MyFormat",
            }
        ]


class DeclarativeProcessorPlugin:
    name = "decl-proc"
    version = "0.2.0"
    api_version = "1.0"

    def register_processors(self) -> list[dict]:
        return [
            {
                "name": "smooth",
                "func": lambda data: data,
                "description": "Smooth data",
            }
        ]


class HybridPlugin:
    """Plugin that implements both imperative register() and declarative hooks."""

    name = "hybrid"
    version = "0.3.0"
    api_version = "1.0"

    def register(self, registry):
        registry.register_reader("legacy_reader", lambda x: x)

    def register_writers(self) -> list[dict]:
        return [
            {
                "name": "decl_writer",
                "func": lambda data, path: None,
                "description": "Declarative writer",
            }
        ]


class BrokenDeclarativePlugin:
    """Plugin whose declarative hook raises."""

    name = "broken"
    version = "0.1.0"
    api_version = "1.0"

    def register_readers(self) -> list[dict]:
        msg = "Internal error"
        raise RuntimeError(msg)


# ------------------------------------------------------------------
# PluginManager — construction
# ------------------------------------------------------------------


def test_plugin_manager_creation():
    pm = PluginManager()
    assert pm is not None
    assert pm._discovered is False


def test_plugin_manager_injected_registry():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    assert pm.registry is registry


def test_plugin_manager_fallback_registry():
    pm = PluginManager()
    from spectrochempy.plugins.registry import registry as global_registry

    assert pm.registry is global_registry


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------


def test_discover_plugins():
    pm = PluginManager()
    pm.discover()
    assert pm._discovered is True
    assert isinstance(pm.list_plugins(), list)


def test_available_plugins_empty():
    pm = PluginManager()
    pm.discover()
    assert pm.has_plugin("nonexistent") is False


# ------------------------------------------------------------------
# Imperative registration (backward compatible)
# ------------------------------------------------------------------


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


def test_register_with_injected_registry():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DummyPlugin()
    pm.register(plugin)
    assert pm.has_plugin("dummy")
    assert registry.get_reader("dummy") is not None


# ------------------------------------------------------------------
# Declarative hook collection
# ------------------------------------------------------------------


def test_declarative_readers():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeReaderPlugin()
    pm.register(plugin)

    reader = registry.get_reader("myformat")
    assert reader is not None
    assert reader["description"] == "Read MyFormat"
    assert reader["extensions"] == [".myf"]


def test_declarative_writers():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeWriterPlugin()
    pm.register(plugin)

    writer = registry.get_writer("myformat")
    assert writer is not None
    assert writer["description"] == "Write MyFormat"


def test_declarative_processors():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeProcessorPlugin()
    pm.register(plugin)

    proc = registry.get_processor("smooth")
    assert proc is not None
    assert proc["description"] == "Smooth data"


# ------------------------------------------------------------------
# Coexistence: imperative + declarative
# ------------------------------------------------------------------


def test_imperative_and_declarative_coexist():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = HybridPlugin()
    pm.register(plugin)

    assert registry.get_reader("legacy_reader") is not None
    assert registry.get_writer("decl_writer") is not None


# ------------------------------------------------------------------
# Error handling in declarative hooks
# ------------------------------------------------------------------


def test_broken_declarative_hook_does_not_crash():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = BrokenDeclarativePlugin()
    # Must not raise despite the RuntimeError in register_readers
    pm.register(plugin)
    assert pm.has_plugin("broken")


# ------------------------------------------------------------------
# Isolation
# ------------------------------------------------------------------


def test_isolated_plugin_managers():
    """Two managers with separate registries must not share state."""
    r1 = PluginRegistry()
    r2 = PluginRegistry()
    pm1 = PluginManager(registry=r1)
    pm2 = PluginManager(registry=r2)

    d1 = DummyPlugin()
    d1.name = "d1"
    d2 = DummyPlugin()
    d2.name = "d2"

    pm1.register(d1)
    pm2.register(d2)

    assert pm1.has_plugin("d1")
    assert not pm1.has_plugin("d2")
    assert pm2.has_plugin("d2")
    assert not pm2.has_plugin("d1")


def test_registry_not_leaked_across_registrations():
    """Each plugin's contributions go to its manager's registry only."""
    r1 = PluginRegistry()
    r2 = PluginRegistry()
    pm1 = PluginManager(registry=r1)
    pm2 = PluginManager(registry=r2)

    p1 = DeclarativeReaderPlugin()
    p2 = DeclarativeReaderPlugin()
    p2.name = "decl-reader-2"  # type: ignore[assignment]

    pm1.register(p1)
    pm2.register(p2)

    assert r1.get_reader("myformat") is not None
    assert r2.get_reader("myformat") is not None
    # verify they are different func objects
    assert r1.get_reader("myformat")["func"] is not r2.get_reader("myformat")["func"]


# ------------------------------------------------------------------
# Protocol checks
# ------------------------------------------------------------------


class TestPluginProtocol:
    def test_protocol_check(self):
        assert isinstance(DummyPlugin(), SpectroChemPyPlugin)

    def test_protocol_without_name(self):
        class InvalidPlugin:
            pass

        assert not isinstance(InvalidPlugin(), SpectroChemPyPlugin)


# ------------------------------------------------------------------
# MissingPluginError
# ------------------------------------------------------------------


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
