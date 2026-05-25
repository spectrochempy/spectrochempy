# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for PluginManager."""


from operator import attrgetter

import pytest

from spectrochempy.plugins.base import SpectroChemPyPluginProtocol
from spectrochempy.plugins.deps import MissingPluginError
from spectrochempy.plugins.lifecycle import PluginState
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.namespace import DatasetPluginAccessor
from spectrochempy.plugins.namespace import PluginNamespace
from spectrochempy.plugins.registry import PluginRegistry

# ------------------------------------------------------------------
# Dummy plugins for testing
# ------------------------------------------------------------------


class DummyPlugin:
    name = "dummy"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register(self, registry):
        registry.register_reader("dummy", lambda x: x)


class DeclarativeReaderPlugin:
    name = "decl-reader"
    version = "0.2.0"
    PLUGIN_API_VERSION = "1.0"

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
    PLUGIN_API_VERSION = "1.0"

    def register_writers(self) -> list[dict]:
        return [
            {
                "name": "myformat",
                "func": lambda data, path: None,
                "description": "Write MyFormat",
            }
        ]


class DeclarativeVisualizerPlugin:
    name = "decl-viz"
    version = "0.2.0"
    PLUGIN_API_VERSION = "1.0"

    def register_visualizers(self) -> list[dict]:
        return [
            {
                "name": "myplot",
                "func": lambda data: None,
                "description": "Plot data",
            }
        ]


class DeclarativeProcessorPlugin:
    name = "decl-proc"
    version = "0.2.0"
    PLUGIN_API_VERSION = "1.0"

    def register_processors(self) -> list[dict]:
        return [
            {
                "name": "smooth",
                "func": lambda data: data,
                "description": "Smooth data",
            }
        ]


class DeclarativeUnitContextPlugin:
    name = "decl-unit-context"
    version = "0.2.0"
    PLUGIN_API_VERSION = "1.0"

    def register_unit_contexts(self) -> list[dict]:
        return [
            {
                "name": "my-context",
                "func": lambda value: value,
                "predicate": lambda obj: getattr(obj, "use_context", False),
                "argument_extractor": lambda obj: (obj.value,),
                "description": "Set up a custom unit context",
            }
        ]


class DeclarativeAccessorPlugin:
    name = "decl-accessor"
    version = "0.2.0"
    PLUGIN_API_VERSION = "1.0"

    def register_accessors(self) -> list[dict]:
        return [
            {
                "name": "plugin_mean",
                "func": lambda dataset: dataset.mean(),
                "description": "Mean via plugin accessor",
            }
        ]


class NamespacedAccessorPlugin:
    name = "domain"
    version = "0.2.0"
    PLUGIN_API_VERSION = "1.0"

    def register_accessors(self) -> list[dict]:
        return [
            {
                "namespace": "domain",
                "name": "scale",
                "legacy_names": ["domain_scale"],
                "func": lambda dataset, factor=2: dataset * factor,
                "description": "Scale via plugin namespace",
            }
        ]


class HybridPlugin:
    """Plugin that implements both imperative register() and declarative hooks."""

    name = "hybrid"
    version = "0.3.0"
    PLUGIN_API_VERSION = "1.0"

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
    PLUGIN_API_VERSION = "1.0"

    def register_readers(self) -> list[dict]:
        msg = "Internal error"
        raise RuntimeError(msg)


# ------------------------------------------------------------------
# PluginManager — construction
# ------------------------------------------------------------------


def test_plugin_manager_creation():
    pm = PluginManager()
    assert pm is not None
    assert pm._discovery_state == "not_discovered"


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
    assert pm._discovery_state == "discovered"
    assert isinstance(pm.list_plugins(), list)


def test_available_plugins_empty():
    pm = PluginManager()
    pm.discover()
    assert pm.has_plugin("nonexistent") is False


def test_available_plugins_uses_registered_plugin_names():
    pm = PluginManager()
    plugin = DummyPlugin()

    pm.register(plugin)

    assert pm.available_plugins["dummy"] is plugin
    assert all(isinstance(name, str) for name in pm.available_plugins)


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


def test_declarative_unit_contexts():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeUnitContextPlugin()
    pm.register(plugin)

    setup = registry.get_unit_context("my-context")
    assert setup is not None
    assert setup("value") == "value"

    class Target:
        use_context = True
        value = "target-value"

    context = registry.get_applicable_unit_context(Target())
    assert context is not None
    assert context["name"] == "my-context"
    assert context["argument_extractor"](Target()) == ("target-value",)


def test_declarative_visualizers():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeVisualizerPlugin()
    pm.register(plugin)

    vis = registry.visualization.get_visualizer("myplot")
    assert vis is not None
    assert vis["description"] == "Plot data"


def test_declarative_accessors():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeAccessorPlugin()
    pm.register(plugin)

    accessor = registry.get_accessor("plugin_mean")
    assert accessor is not None
    assert accessor["description"] == "Mean via plugin accessor"


def test_declarative_namespaced_accessors():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = NamespacedAccessorPlugin()
    pm.register(plugin)

    accessor = registry.get_accessor("domain.scale")
    assert accessor is not None
    assert accessor["metadata"]["namespace"] == "domain"
    assert registry.get_accessor("domain_scale") is not None


def test_plugin_namespace_clear_error_when_api_missing():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    namespace = PluginNamespace("domain", pm, registry)

    with pytest.raises(AttributeError, match="plugin namespace 'domain'"):
        attrgetter("missing")(namespace)


def test_dataset_namespace_clear_error_when_api_missing():
    registry = PluginRegistry()
    namespace = DatasetPluginAccessor(object(), "domain", registry)

    with pytest.raises(AttributeError, match="dataset plugin accessor 'domain'"):
        attrgetter("missing")(namespace)


# ------------------------------------------------------------------
# Declarative hook routing to specialised sub-registries
# ------------------------------------------------------------------


def test_declarative_readers_routed_to_io():
    """register_readers targets registry.io."""
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeReaderPlugin()
    pm.register(plugin)

    assert registry.io.get_reader("myformat") is not None


def test_declarative_writers_routed_to_io():
    """register_writers targets registry.io."""
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeWriterPlugin()
    pm.register(plugin)

    assert registry.io.get_writer("myformat") is not None


def test_declarative_processors_routed_to_processing():
    """register_processors targets registry.processing."""
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeProcessorPlugin()
    pm.register(plugin)

    assert registry.processing.get_processor("smooth") is not None


def test_declarative_unit_contexts_routed_to_processing():
    """register_unit_contexts targets registry.processing."""
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeUnitContextPlugin()
    pm.register(plugin)

    assert registry.processing.get_unit_context("my-context") is not None


def test_declarative_visualizers_routed_to_visualization():
    """register_visualizers targets registry.visualization."""
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeVisualizerPlugin()
    pm.register(plugin)

    assert registry.visualization.get_visualizer("myplot") is not None


def test_declarative_accessors_routed_to_extensions():
    """register_accessors targets registry.extensions."""
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    plugin = DeclarativeAccessorPlugin()
    pm.register(plugin)

    assert registry.extensions.get("accessor", "plugin_mean") is not None


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
    assert pm.get_plugin_state("broken") is PluginState.FAILED
    assert all(
        info.get("plugin") != "broken" for info in registry.available_readers.values()
    )


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
        assert isinstance(DummyPlugin(), SpectroChemPyPluginProtocol)

        # An InvalidPlugin without name/version/api_version is NOT a protocol match
        class InvalidPlugin:
            pass

        assert not isinstance(InvalidPlugin(), SpectroChemPyPluginProtocol)


# ------------------------------------------------------------------
# MissingPluginError
# ------------------------------------------------------------------


class TestMissingPluginError:
    def test_default_message_with_plugin_name(self):
        err = MissingPluginError("topspin reader", plugin_name="spectrochempy-nmr")
        msg = str(err)
        assert "topspin reader" in msg
        assert "spectrochempy-nmr" in msg

    def test_default_message_without_plugin_name(self):
        err = MissingPluginError("topspin reader")
        msg = str(err)
        assert "topspin reader" in msg
        assert "optional plugin" in msg

    def test_custom_hint(self):
        err = MissingPluginError(
            "test", plugin_name="mypkg", install_hint="pip install mypkg"
        )
        assert "pip install mypkg" in str(err)

    def test_notebook_traceback_rendering_is_compact(self):
        err = MissingPluginError(
            "test", plugin_name="mypkg", install_hint="pip install mypkg"
        )
        rendered = err._render_traceback_()
        assert rendered == [f"MissingPluginError: {err}\n"]
        assert "Traceback" not in rendered[0]


# ------------------------------------------------------------------
# Test plugins for lifecycle
# ------------------------------------------------------------------


class FailingConstructorPlugin:
    """Plugin whose constructor raises (simulates missing optional dep)."""

    name = "fail-ctor"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def __init__(self):
        msg = "optional dependency not found"
        raise ImportError(msg)

    def register(self, registry):
        ...


class FailingRegisterPlugin:
    """Plugin whose register() raises."""

    name = "fail-reg"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register(self, registry):
        msg = "registration failure"
        raise RuntimeError(msg)


# ------------------------------------------------------------------
# Lifecycle states
# ------------------------------------------------------------------


def test_plugin_goes_active_after_register():
    pm = PluginManager()
    plugin = DummyPlugin()
    pm.register(plugin)
    assert pm.get_plugin_state("dummy").value == "active"


def test_incompatible_plugin_goes_failed():
    class BadAPI:
        name = "bad"
        version = "1.0"
        api_version = "99.0"

        def register(self, registry):
            ...

    pm = PluginManager()
    pm.register(BadAPI())
    assert pm.get_plugin_state("bad").value == "failed"


def test_get_active_plugins():
    pm = PluginManager()
    pm.register(DummyPlugin())
    active = pm.get_active_plugins()
    assert "dummy" in active


def test_list_contributions():
    registry = PluginRegistry()
    pm = PluginManager(registry=registry)
    pm.register(DeclarativeReaderPlugin())
    pm.register(DeclarativeWriterPlugin())
    pm.register(DeclarativeProcessorPlugin())
    pm.register(DeclarativeVisualizerPlugin())
    pm.register(DeclarativeAccessorPlugin())

    assert "myformat" in pm.list_readers()
    assert "myformat" in pm.list_writers()
    assert "smooth" in pm.list_processors()
    assert "myplot" in pm.list_visualizers()
    assert "plugin_mean" in pm.list_accessors()


def test_get_active_plugins_excludes_failed():
    class BadAPI:
        name = "bad"
        version = "1.0"
        api_version = "99.0"

        def register(self, registry):
            ...

    pm = PluginManager()
    pm.register(DummyPlugin())
    pm.register(BadAPI())
    active = pm.get_active_plugins()
    assert "dummy" in active
    assert "bad" not in active


def test_get_failed_plugins():
    class BadAPI:
        name = "bad"
        version = "1.0"
        api_version = "99.0"

        def register(self, registry):
            ...

    pm = PluginManager()
    pm.register(BadAPI())
    failed = pm.get_failed_plugins()
    assert "bad" in failed


# ------------------------------------------------------------------
# PluginDescriptor
# ------------------------------------------------------------------


def test_get_plugin_descriptor():
    pm = PluginManager()
    plugin = DummyPlugin()
    pm.register(plugin)

    desc = pm.get_plugin_descriptor("dummy")
    assert desc is not None
    assert desc.name == "dummy"
    assert desc.version == "0.1.0"
    assert desc.state.value == "active"
    assert desc.error == ""


def test_get_plugin_descriptor_nonexistent():
    pm = PluginManager()
    assert pm.get_plugin_descriptor("nope") is None


# ------------------------------------------------------------------
# Activation / deactivation
# ------------------------------------------------------------------


def test_deactivate_plugin():
    pm = PluginManager()
    pm.register(DummyPlugin())
    assert pm.deactivate_plugin("dummy") is True
    assert pm.get_plugin_state("dummy").value == "disabled"


def test_activate_plugin():
    pm = PluginManager()
    pm.register(DummyPlugin())
    pm.deactivate_plugin("dummy")
    assert pm.activate_plugin("dummy") is True
    assert pm.get_plugin_state("dummy").value == "active"


def test_deactivate_nonexistent_plugin():
    pm = PluginManager()
    assert pm.deactivate_plugin("nope") is False


def test_activate_nonexistent_plugin():
    pm = PluginManager()
    assert pm.activate_plugin("nope") is False


def test_disabled_plugin_skipped_on_register():
    pm = PluginManager()
    pm.register(DummyPlugin())
    pm.deactivate_plugin("dummy")

    # Try registering the same plugin again
    pm.register(DummyPlugin())
    # Should still be disabled
    assert pm.get_plugin_state("dummy").value == "disabled"


# ------------------------------------------------------------------
# Error isolation
# ------------------------------------------------------------------


def test_register_with_constructor_error():
    """A plugin whose constructor raises is handled without crashing."""
    pm = PluginManager()

    def _register():
        try:
            plugin = FailingConstructorPlugin()
            pm.register(plugin)
        except ImportError:
            pm._plugin_states["fail-ctor"] = "failed"  # type: ignore[assignment]

    _register()
    # The manager should still be usable after a constructor failure
    healthy = DummyPlugin()
    pm.register(healthy)
    assert pm.get_plugin_state("dummy").value == "active"


def test_register_with_register_error():
    """A plugin whose register() raises is marked FAILED."""
    pm = PluginManager()
    plugin = FailingRegisterPlugin()
    pm.register(plugin)
    assert pm.get_plugin_state("fail-reg").value == "failed"


def test_failing_plugin_does_not_affect_others():
    """Other plugins remain ACTIVE after a sibling fails."""
    pm = PluginManager()
    pm.register(DummyPlugin())
    pm.register(FailingRegisterPlugin())
    assert pm.get_plugin_state("dummy").value == "active"
    assert pm.get_plugin_state("fail-reg").value == "failed"


# ------------------------------------------------------------------
# PluginState enum and PluginDescriptor dataclass
# ------------------------------------------------------------------


def test_plugin_state_values():
    assert PluginState.DISCOVERED.value == "discovered"
    assert PluginState.LOADED.value == "loaded"
    assert PluginState.ACTIVE.value == "active"
    assert PluginState.FAILED.value == "failed"
    assert PluginState.DISABLED.value == "disabled"


def test_plugin_descriptor_defaults():
    from spectrochempy.plugins.lifecycle import PluginDescriptor

    d = PluginDescriptor(name="test")
    assert d.name == "test"
    assert d.version == ""
    assert d.state == PluginState.DISCOVERED
    assert d.error is None
    assert d.entry_point is None
