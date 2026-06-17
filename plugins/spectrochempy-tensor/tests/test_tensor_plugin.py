# ruff: noqa: S101, PLC0415 - assert/local imports allowed in plugin tests

"""Tests for spectrochempy-tensor plugin registration and lifecycle."""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

import pytest
from spectrochempy_tensor import TensorPlugin

import spectrochempy as scp
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness


def _require_tensorly() -> None:
    pytest.importorskip("tensorly")


def _package_version() -> str:
    try:
        return version("spectrochempy-tensor")
    except PackageNotFoundError:
        return TensorPlugin.version


def test_plugin_metadata():
    """Plugin has required metadata fields."""
    plugin = TensorPlugin()
    assert plugin.name == "tensor"
    assert plugin.version == _package_version()
    assert plugin.description
    assert PluginCapability.ANALYSIS in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes compatibility checks."""
    _require_tensorly()
    plugin = TensorPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"


def test_registration():
    """Plugin registers CP via declarative analysis hook."""
    _require_tensorly()
    harness = PluginTestHarness()
    harness.register(TensorPlugin())

    analyses = harness.registry.extensions.list_category("analysis")
    assert "CP" in analyses
    assert analyses["CP"]["metadata"]["plugin"] == "tensor"
    assert analyses["CP"]["metadata"]["namespace"] == "tensor"


def test_lifecycle_state():
    """Plugin transitions to ACTIVE after registration."""
    _require_tensorly()
    harness = PluginTestHarness()
    harness.register(TensorPlugin())
    assert harness.get_plugin_state("tensor") == PluginState.ACTIVE


def test_package_namespace_exposes_cp(monkeypatch):
    """scp.tensor.CP is exposed by the registered plugin without deprecation."""
    _require_tensorly()
    import sys

    harness = PluginTestHarness()
    harness.register(TensorPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "tensor", raising=False)
    monkeypatch.delitem(sys.modules, "spectrochempy.tensor", raising=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        cp_class = scp.tensor.CP

    assert cp_class.__name__ == "CP"
    assert [w for w in captured if issubclass(w.category, DeprecationWarning)] == []


def test_root_alias_warns_once(monkeypatch):
    """scp.CP works as a deprecated compatibility alias."""
    _require_tensorly()
    harness = PluginTestHarness()
    harness.register(TensorPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "CP", raising=False)
    scp._EMITTED_PLUGIN_ROOT_WARNINGS.discard("CP")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        val1 = scp.CP
        val2 = scp.CP

    assert val1 is val2
    assert len(captured) == 1
    assert captured[0].category is DeprecationWarning
    message = str(captured[0].message)
    assert "scp.CP is deprecated since SpectroChemPy 0.9.0" in message
    assert "will be removed in 0.11.0" in message
    assert "scp.tensor.CP" in message
