# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Integration tests for the plugin system (lazy loading, discovery, user-facing errors)."""

from __future__ import annotations

import importlib.metadata as im
import logging
from typing import Any

import pytest

import spectrochempy
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.plugins.manager import ENTRY_POINT_GROUP
from spectrochempy.plugins.manager import PluginManager
from spectrochempy.plugins.namespace import PluginNamespace
from spectrochempy.plugins.namespace import has_namespace
from spectrochempy.plugins.registry import PluginRegistry

# ------------------------------------------------------------------
# Fake plugins for integration testing
# ------------------------------------------------------------------


class FakeReaderPlugin(SpectroChemPyPlugin):
    name = "fake_reader"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register_readers(self) -> list[dict]:
        def read_fake(path: str) -> str:
            return f"fake data from {path}"

        return [
            {
                "name": "fake",
                "func": read_fake,
                "description": "Read fake format",
                "extensions": [".fake"],
            }
        ]


class FakeNamespacePlugin(SpectroChemPyPlugin):
    name = "fakenamespace"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register_readers(self) -> list[dict]:
        def read_fake_ns(path: str) -> str:
            return f"fake ns data from {path}"

        return [
            {
                "name": "fakename",
                "func": read_fake_ns,
                "description": "Read fake ns format",
                "namespace": "fakenamespace",
                "extensions": [".fns"],
            }
        ]


class FakeAccessorPlugin(SpectroChemPyPlugin):
    name = "fakeaccessor"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register_accessors(self) -> list[dict]:
        def my_accessor(dataset: Any, factor: float = 2.0) -> str:
            return f"accessor applied with factor {factor}"

        return [
            {
                "namespace": "fakeaccessor",
                "name": "transform",
                "func": my_accessor,
                "description": "Fake accessor transform",
            }
        ]


class InvalidPlugin:
    """Plugin without name/version/api_version — malformed."""

    def register(self, registry: Any) -> None:
        registry.register_reader("bad", lambda x: x)


# ------------------------------------------------------------------
# A. Import core
# ------------------------------------------------------------------


class TestCoreImport:
    def test_import_spectrochempy(self):
        """Import spectrochempy works without plugin dependencies."""

        assert spectrochempy.__version__ is not None

    def test_access_nddataset(self):
        """scp.NDDataset is accessible."""

        assert spectrochempy.NDDataset is not None

    def test_access_read(self):
        """scp.read is accessible (core IO function)."""

        assert callable(spectrochempy.read)

    def test_access_read_omnic(self):
        """scp.read_omnic is accessible (built-in reader)."""

        assert callable(spectrochempy.read_omnic)


# ------------------------------------------------------------------
# B. Discover idempotent
# ------------------------------------------------------------------


class TestDiscoverIdempotent:
    def test_discover_idempotent(self):
        """Two discover() calls do not duplicate readers."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeReaderPlugin()
        pm.register(plugin)

        pm.discover()
        before = len(registry.available_readers)
        pm.discover()
        assert len(registry.available_readers) == before

    def test_discover_does_not_duplicate_plugins(self):
        """Calling discover() twice does not register plugins twice."""
        pm = PluginManager()
        pm.register(FakeReaderPlugin())
        active_before = pm.get_active_plugins()
        pm.discover()
        active_after = pm.get_active_plugins()
        assert active_before == active_after

    def test_discover_not_discovered_initial_state(self):
        """Fresh PluginManager starts as not_discovered."""
        pm = PluginManager()
        assert pm._discovery_state == "not_discovered"


# ------------------------------------------------------------------
# C. Plugin top-level function via __getattr__
# ------------------------------------------------------------------


class TestPluginTopLevelFunction:
    def test_reader_exposed_at_top_level(self, monkeypatch):
        """A registered reader is accessible via scp.read_fake."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeReaderPlugin()
        pm.register(plugin)

        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)

        read_fake = spectrochempy.read_fake
        assert callable(read_fake)
        assert read_fake("/some/path") == "fake data from /some/path"

    def test_reader_callable(self):
        """A registered reader function works when called."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeReaderPlugin()
        pm.register(plugin)

        reader = registry.get_reader("fake")
        assert reader is not None
        assert reader["func"]("/test") == "fake data from /test"


# ------------------------------------------------------------------
# D. Plugin namespace
# ------------------------------------------------------------------


class TestPluginNamespace:
    def test_namespace_accessible_with_prefix(self, monkeypatch):
        """Plugin namespace 'fakenamespace' is accessible via scp.fakenamespace."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeNamespacePlugin()
        pm.register(plugin)

        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)

        assert has_namespace(registry, "fakenamespace")
        ns = spectrochempy.fakenamespace
        assert isinstance(ns, PluginNamespace)

    def test_namespace_reader_accessible(self, monkeypatch):
        """Namespace reader 'scp.fakenamespace.read_fakename' works."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeNamespacePlugin()
        pm.register(plugin)

        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)

        ns = spectrochempy.fakenamespace
        result = ns.read_fakename("/test")
        assert result == "fake ns data from /test"

    def test_namespace_error_clear(self, monkeypatch):
        """Accessing a missing attribute on a namespace gives a clear error."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeNamespacePlugin()
        pm.register(plugin)

        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)

        ns = spectrochempy.fakenamespace
        with pytest.raises(AttributeError, match="plugin namespace 'fakenamespace'"):
            _ = ns.missing_attribute


# ------------------------------------------------------------------
# E. Missing plugin / stub
# ------------------------------------------------------------------


class TestMissingPlugin:
    def test_missing_reader_clear_error(self, monkeypatch):
        """Accessing scp.read_topspin without NMR plugin gives a clear error."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)

        with pytest.raises(AttributeError, match="requires the optional plugin"):
            _ = spectrochempy.read_topspin

    def test_missing_namespace_clear_error(self, monkeypatch):
        """Accessing scp.nmr without NMR plugin gives a clear error."""

        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        monkeypatch.setattr(spectrochempy, "plugin_manager", pm)
        monkeypatch.setattr(spectrochempy, "registry", registry)

        with pytest.raises(AttributeError, match="requires the optional plugin"):
            _ = spectrochempy.nmr

    def test_unknown_attribute_standard_error(self):
        """Accessing a truly unknown attribute gives a standard error."""

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = spectrochempy.nonexistent_attribute_xyz123


# ------------------------------------------------------------------
# F. Invalid plugin
# ------------------------------------------------------------------


class FailingRegisterPlugin:
    """Plugin whose imperative register() raises."""

    name = "fail-reg"
    version = "0.1.0"
    PLUGIN_API_VERSION = "1.0"

    def register(self, registry):
        msg = "registration failure"
        raise RuntimeError(msg)


class TestInvalidPlugin:
    def test_invalid_plugin_does_not_crash(self):
        """A malformed plugin does not crash the manager."""
        pm = PluginManager()
        plugin = InvalidPlugin()
        pm.register(plugin)

    def test_invalid_plugin_fails_gracefully(self):
        """A malformed plugin is tracked as FAILED."""
        pm = PluginManager()
        plugin = InvalidPlugin()
        pm.register(plugin)
        state = pm.get_plugin_state("invalidplugin")
        assert state is not None
        assert state.value == "failed"

    def test_valid_plugin_works_after_invalid(self):
        """A valid plugin can still register after an invalid one."""
        pm = PluginManager()
        pm.register(InvalidPlugin())
        pm.register(FakeReaderPlugin())
        assert pm.get_plugin_state("fake_reader").value == "active"

    def test_discover_does_not_crash_with_bad_entry_points(self, monkeypatch):
        """An entry point that loads a broken class does not crash discover()."""

        class BrokenEntryPoint:
            name = "broken_ep"
            value = "broken_module:BrokenPlugin"

            @staticmethod
            def load():
                msg = "broken module"
                raise ImportError(msg)

        original = im.entry_points

        def mock_entry_points(group=None):
            if group == ENTRY_POINT_GROUP:
                return [BrokenEntryPoint()]
            return original(group=group)

        monkeypatch.setattr(im, "entry_points", mock_entry_points)
        pm = PluginManager()
        pm.discover()  # must not raise

    def test_failed_plugin_not_available_via_get_plugin(self):
        """A FAILED plugin must not appear in get_plugin()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert pm.get_plugin("fail-reg") is None

    def test_failed_plugin_not_available_via_has_plugin(self):
        """A FAILED plugin must not appear in has_plugin()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert pm.has_plugin("fail-reg") is False

    def test_failed_plugin_not_in_list_plugins(self):
        """A FAILED plugin must not appear in list_plugins()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert "fail-reg" not in [
            p.name for p in pm.list_plugins() if hasattr(p, "name")
        ]

    def test_failed_plugin_not_active(self):
        """A FAILED plugin must not appear in get_active_plugins()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        assert "fail-reg" not in pm.get_active_plugins()

    def test_failed_plugin_reported_in_get_failed(self):
        """A FAILED plugin is still tracked via get_failed_plugins()."""
        pm = PluginManager()
        pm.register(FailingRegisterPlugin())
        failed = pm.get_failed_plugins()
        assert "fail-reg" in failed

    def test_load_plugin_returns_none_for_failed(self):
        """load_plugin returns None when the plugin fails to register."""
        pm = PluginManager()

        class AlwaysFails:
            name = "always-fails"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register(self, registry):
                msg = "always fails"
                raise RuntimeError(msg)

        result = pm.load_plugin("always-fails")
        assert result is None


# ------------------------------------------------------------------
# G. Dataset accessors
# ------------------------------------------------------------------


class TestDatasetAccessors:
    def test_accessor_not_duplicated_on_repeated_register(self):
        """Calling register() twice with the same plugin does not add duplicate accessors."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeAccessorPlugin()
        pm.register(plugin)
        assert len(registry.available_accessors) == 1
        pm.register(plugin)
        assert len(registry.available_accessors) == 1

    def test_accessor_does_not_become_reader(self):
        """An accessor is not accidentally exposed as a reader."""
        registry = PluginRegistry()
        pm = PluginManager(registry=registry)
        plugin = FakeAccessorPlugin()
        pm.register(plugin)
        assert registry.get_reader("transform") is None
        assert registry.get_accessor("fakeaccessor.transform") is not None


# ------------------------------------------------------------------
# H. Discovery state machine
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# I. __dir__ without side effects
# ------------------------------------------------------------------


class TestDirNoSideEffect:
    def test_dir_does_not_trigger_discovery(self, monkeypatch):
        """dir(scp) must not trigger plugin_manager.discover()."""

        discover_called = False
        original_discover = spectrochempy.plugin_manager.discover

        def tracking_discover():
            nonlocal discover_called
            discover_called = True
            original_discover()

        monkeypatch.setattr(spectrochempy.plugin_manager, "discover", tracking_discover)
        dir(spectrochempy)
        assert not discover_called, "dir(scp) triggered plugin_manager.discover()"


# ------------------------------------------------------------------
# J. Warnings for invalid contributions
# ------------------------------------------------------------------


class TestInvalidContributionWarnings:
    def test_warning_on_non_list_return(self, caplog):
        """register_readers returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadReturnPlugin:
            name = "bad-return"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return "not a list"

        pm = PluginManager()
        pm.register(BadReturnPlugin())
        assert any(
            "bad-return" in msg and "register_readers" in msg for msg in caplog.messages
        )

    def test_warning_on_non_dict_item(self, caplog):
        """register_readers returning a non-dict item logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadItemPlugin:
            name = "bad-item"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return ["not a dict"]

        pm = PluginManager()
        pm.register(BadItemPlugin())
        assert any("bad-item" in msg and "is str" in msg for msg in caplog.messages)

    def test_warning_on_missing_keys(self, caplog):
        """register_readers with missing 'name'/'func' keys logs a warning."""

        caplog.set_level(logging.WARNING)

        class MissingKeysPlugin:
            name = "missing-keys"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_readers(self):
                return [{"description": "no name or func"}]

        pm = PluginManager()
        pm.register(MissingKeysPlugin())
        assert any(
            "missing-keys" in msg and "missing required keys" in msg
            for msg in caplog.messages
        )

    def test_warning_on_writers_non_list(self, caplog):
        """register_writers returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadWritersPlugin:
            name = "bad-writers"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_writers(self):
                return 42

        pm = PluginManager()
        pm.register(BadWritersPlugin())
        assert any(
            "bad-writers" in msg and "register_writers" in msg
            for msg in caplog.messages
        )

    def test_warning_on_analyses_non_list(self, caplog):
        """register_analyses returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadAnalysesPlugin:
            name = "bad-analyses"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_analyses(self):
                return None

        pm = PluginManager()
        pm.register(BadAnalysesPlugin())
        assert any(
            "bad-analyses" in msg and "register_analyses" in msg
            for msg in caplog.messages
        )

    def test_warning_on_simulations_non_list(self, caplog):
        """register_simulations returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadSimPlugin:
            name = "bad-sim"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_simulations(self):
                return "invalid"

        pm = PluginManager()
        pm.register(BadSimPlugin())
        assert any(
            "bad-sim" in msg and "register_simulations" in msg
            for msg in caplog.messages
        )

    def test_warning_on_accessors_non_list(self, caplog):
        """register_accessors returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadAccessorPlugin:
            name = "bad-accessor"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_accessors(self):
                return [42]

        pm = PluginManager()
        pm.register(BadAccessorPlugin())
        assert any(
            "bad-accessor" in msg and "register_accessors" in msg
            for msg in caplog.messages
        )

    def test_warning_on_processors_non_list(self, caplog):
        """register_processors returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadProcPlugin:
            name = "bad-proc"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_processors(self):
                return (1, 2)

        pm = PluginManager()
        pm.register(BadProcPlugin())
        assert any(
            "bad-proc" in msg and "register_processors" in msg
            for msg in caplog.messages
        )

    def test_warning_on_visualizers_non_list(self, caplog):
        """register_visualizers returning a non-list logs a warning."""

        caplog.set_level(logging.WARNING)

        class BadVizPlugin:
            name = "bad-viz"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register_visualizers(self):
                return [{"name": "foo"}]  # missing func

        pm = PluginManager()
        pm.register(BadVizPlugin())
        assert any(
            "bad-viz" in msg and "missing required keys" in msg
            for msg in caplog.messages
        )


class TestDiscoveryStateMachine:
    def test_discovery_state_transition(self):
        """Discovery goes NOT_DISCOVERED -> DISCOVERING -> DISCOVERED."""
        pm = PluginManager()
        assert pm._discovery_state == "not_discovered"
        pm.discover()
        assert pm._discovery_state == "discovered"

    def test_reentrant_discovery_safe(self):
        """Calling discover() during discover() is safe (no infinite loop)."""
        pm = PluginManager()

        class ReentrantPlugin:
            name = "reentrant"
            version = "0.1.0"
            PLUGIN_API_VERSION = "1.0"

            def register(self, registry):
                pm.discover()

        pm.register(ReentrantPlugin())
        assert pm._discovery_state == "discovered"

    def test_discovery_skipped_when_already_discovering(self):
        """Calling discover() while already discovering returns immediately."""
        pm = PluginManager()
        pm._discovering = True
        pm.discover()
        assert pm._discovery_state == "not_discovered"

    def test_registration_does_not_trigger_discovery(self):
        """Registering a plugin directly does not trigger entry point discovery."""

        original = im.entry_points
        call_count = 0

        def counting_mock(group=None):
            nonlocal call_count
            call_count += 1
            return original(group=group) if group != ENTRY_POINT_GROUP else []

        pm = PluginManager()
        pm.register(FakeReaderPlugin())
        assert call_count == 0
