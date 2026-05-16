# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for plugin compatibility validation (spectrochempy.api.plugins)."""

from __future__ import annotations

from typing import Any

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.api.plugins.validation import _get_plugin_metadata
from spectrochempy.api.plugins.validation import _is_compatible_api_version
from spectrochempy.api.plugins.validation import _satisfies_min_version
from spectrochempy.api.plugins.validation import check_plugin_requires
from spectrochempy.api.plugins.validation import validate_plugin_compatibility

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _ValidPlugin(SpectroChemPyPlugin):
    name = "valid"
    version = "1.0.0"
    description = "A valid test plugin"


class _IncompatibleAPIPlugin(SpectroChemPyPlugin):
    name = "bad-api"
    version = "1.0.0"
    PLUGIN_API_VERSION = "2.0"


class _TooNewPlugin(SpectroChemPyPlugin):
    name = "toonew"
    version = "1.0.0"
    spectrochempy_min_version = "999.0.0"


class _MissingNamePlugin:
    version = "1.0.0"
    api_version = CORE_PLUGIN_API_VERSION

    def register(self, registry: Any) -> None:
        pass


class _PluginWithInfo(SpectroChemPyPlugin):
    name = "with-info"
    version = "2.0.0"
    description = "Has custom plugin_info"

    def plugin_info(self) -> dict[str, Any]:
        return {
            "name": "custom-name",
            "version": "3.0.0",
            "plugin_api_version": CORE_PLUGIN_API_VERSION,
            "spectrochempy_min_version": "0.0.0",
            "description": "Custom info",
            "capabilities": ["reader", "writer"],
        }


class _FallbackPlugin:
    name = "fallback"
    version = "0.5.0"
    api_version = CORE_PLUGIN_API_VERSION

    def register(self, registry: Any) -> None:
        pass


# ------------------------------------------------------------------
# Metadata extraction
# ------------------------------------------------------------------


class TestGetPluginMetadata:
    def test_from_plugin_info_method(self):
        plugin = _PluginWithInfo()
        meta = _get_plugin_metadata(plugin)
        assert meta["name"] == "custom-name"
        assert meta["version"] == "3.0.0"
        assert "capabilities" in meta

    def test_fallback_to_attributes(self):
        plugin = _FallbackPlugin()
        meta = _get_plugin_metadata(plugin)
        assert meta["name"] == "fallback"
        assert meta["version"] == "0.5.0"
        assert meta["plugin_api_version"] == CORE_PLUGIN_API_VERSION


# ------------------------------------------------------------------
# Version comparison
# ------------------------------------------------------------------


class TestIsCompatibleAPIVersion:
    def test_same_major(self):
        assert _is_compatible_api_version("1.0", "1.0") is True
        assert _is_compatible_api_version("1.2", "1.5") is True
        assert _is_compatible_api_version("1.0", "1.0") is True

    def test_different_major(self):
        assert _is_compatible_api_version("2.0", "1.0") is False
        assert _is_compatible_api_version("1.0", "2.0") is False
        assert _is_compatible_api_version("0.9", "1.0") is False

    def test_malformed(self):
        assert _is_compatible_api_version("", "1.0") is False
        assert _is_compatible_api_version("abc", "1.0") is False


class TestSatisfiesMinVersion:
    def test_exact_match(self):
        assert _satisfies_min_version("1.0.0", "1.0.0") is True

    def test_newer(self):
        assert _satisfies_min_version("2.0.0", "1.0.0") is True
        assert _satisfies_min_version("1.1.0", "1.0.0") is True

    def test_older(self):
        assert _satisfies_min_version("0.9.0", "1.0.0") is False

    def test_different_lengths(self):
        assert _satisfies_min_version("1.0", "1.0.0") is True
        assert _satisfies_min_version("1.0.0", "1.0") is True
        assert _satisfies_min_version("0.9", "1.0.0") is False

    def test_malformed(self):
        assert _satisfies_min_version("abc", "1.0.0") is False


# ------------------------------------------------------------------
# Compatibility validation
# ------------------------------------------------------------------


class TestValidatePluginCompatibility:
    def test_valid_plugin(self):
        plugin = _ValidPlugin()
        ok, errors = validate_plugin_compatibility(plugin)
        assert ok is True
        assert errors == []

    def test_incompatible_api_version(self):
        plugin = _IncompatibleAPIPlugin()
        ok, errors = validate_plugin_compatibility(plugin)
        assert ok is False
        assert any("API version" in e for e in errors)

    def test_too_new_for_spectrochempy(self):
        plugin = _TooNewPlugin()
        ok, errors = validate_plugin_compatibility(plugin)
        assert ok is False
        assert any(">= 999.0.0" in e for e in errors)
        assert "skipped" in errors[0].lower()

    def test_missing_required_metadata(self):
        plugin = _MissingNamePlugin()
        ok, errors = validate_plugin_compatibility(plugin)
        assert ok is False
        assert any("name" in e for e in errors)

    def test_uses_plugin_info_method(self):
        plugin = _PluginWithInfo()
        ok, errors = validate_plugin_compatibility(plugin)
        assert ok is True
        assert errors == []

    def test_fallback_attributes(self):
        plugin = _FallbackPlugin()
        ok, errors = validate_plugin_compatibility(plugin)
        assert ok is True
        assert errors == []

    def test_valid_registers_through_manager(self):
        from spectrochempy.plugins.manager import PluginManager

        pm = PluginManager()
        plugin = _ValidPlugin()
        pm.register(plugin)
        assert pm.has_plugin("valid")

    def test_incompatible_skipped_by_manager(self):
        from spectrochempy.plugins.manager import PluginManager

        pm = PluginManager()
        plugin = _IncompatibleAPIPlugin()
        pm.register(plugin)
        assert not pm.has_plugin("bad-api")


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# check_plugin_requires
# ------------------------------------------------------------------


class TestCheckPluginRequires:
    def test_no_requires(self):
        """Plugin without requires passes."""
        plugin = _ValidPlugin()
        issues = check_plugin_requires(plugin)
        assert issues == []

    def test_available_dependency(self):
        """Plugin with installed dependency passes."""

        class _PluginWithNumpy:
            name = "has-numpy"
            requires = ["numpy"]

        issues = check_plugin_requires(_PluginWithNumpy())
        assert issues == []

    def test_missing_dependency(self):
        """Plugin with missing dependency reports it."""

        class _PluginWithMissing:
            name = "missing-dep"
            requires = ["nonexistent_package_xyz"]

        issues = check_plugin_requires(_PluginWithMissing())
        assert any("nonexistent_package_xyz" in i for i in issues)

    def test_malformed_required_ignored(self):
        """Malformed dep string does not crash."""

        class _PluginWithBad:
            name = "bad-dep"
            requires = ["numpy>=1.0.0"]

        issues = check_plugin_requires(_PluginWithBad())
        assert issues == []


# ------------------------------------------------------------------
# PluginCapability — new values
# ------------------------------------------------------------------


class TestPluginCapabilityNewValues:
    def test_analysis_exists(self):
        from spectrochempy.api.plugins import PluginCapability

        assert hasattr(PluginCapability, "ANALYSIS")
        assert PluginCapability.ANALYSIS.value == "analysis"

    def test_simulation_exists(self):
        from spectrochempy.api.plugins import PluginCapability

        assert hasattr(PluginCapability, "SIMULATION")
        assert PluginCapability.SIMULATION.value == "simulation"

    def test_accessor_exists(self):
        from spectrochempy.api.plugins import PluginCapability

        assert hasattr(PluginCapability, "ACCESSOR")
        assert PluginCapability.ACCESSOR.value == "accessor"


# Public API import paths
# ------------------------------------------------------------------


class TestPublicAPIImports:
    def test_import_from_api(self):
        from spectrochempy.api import CORE_PLUGIN_API_VERSION as V1
        from spectrochempy.api import SpectroChemPyPlugin as Cls1
        from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION as V2
        from spectrochempy.api.plugins import SpectroChemPyPlugin as Cls2

        assert V1 == CORE_PLUGIN_API_VERSION
        assert V2 == CORE_PLUGIN_API_VERSION
        assert Cls1 is Cls2

    def test_import_hook_markers(self):
        from spectrochempy.api.plugins import hookimpl
        from spectrochempy.api.plugins import hookspec

        assert hookimpl is not None
        assert hookspec is not None

    def test_plugin_has_default_metadata(self):
        plugin = _ValidPlugin()
        info = plugin.plugin_info()
        assert info["name"] == "valid"
        assert info["version"] == "1.0.0"
        assert info["plugin_api_version"] == CORE_PLUGIN_API_VERSION
        assert info["spectrochempy_min_version"] == "0.0.0"


# ------------------------------------------------------------------
# PluginCapability
# ------------------------------------------------------------------


class TestPluginCapability:
    def test_import_from_api_plugins(self):
        from spectrochempy.api.plugins import PluginCapability

        assert PluginCapability.READER.value == "reader"
        assert PluginCapability.WRITER.value == "writer"
        assert PluginCapability.PROCESSOR.value == "processor"
        assert PluginCapability.VISUALIZER.value == "visualizer"

    def test_import_from_api(self):
        from spectrochempy.api import PluginCapability

        assert PluginCapability.READER is not None

    def test_values_are_strings(self):
        from spectrochempy.api.plugins import PluginCapability

        for cap in PluginCapability:
            assert isinstance(cap.value, str)

    def test_capabilities_unique(self):
        from spectrochempy.api.plugins import PluginCapability

        values = [c.value for c in PluginCapability]
        assert len(values) == len(set(values))
