# ruff: noqa: S101, PLC0415  # assert/local imports allowed in plugin tests

"""Tests for spectrochempy-iris plugin — extended IRIS analysis."""

from __future__ import annotations

import numpy as np
import pytest
from spectrochempy_iris import IrisPlugin
from spectrochempy_iris import batch_iris_analysis
from spectrochempy_iris import compare_kernel_models
from spectrochempy_iris import iris_analysis_report
from spectrochempy_iris import plot_distribution_grid
from spectrochempy_iris import plot_kernel_comparison

from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness

# ------------------------------------------------------------------
# Test data helpers
# ------------------------------------------------------------------


def _make_test_dataset(n_spectra: int = 19, n_channels: int = 50) -> object:
    """Create a synthetic dataset mimicking IRIS test data."""
    import spectrochempy as scp

    rng = np.random.default_rng(42)

    # Synthetic decaying exponential spectra with pressure-like variation
    p_values = np.logspace(-3, 0, n_spectra)
    wavenumbers = np.linspace(2100, 2000, n_channels)

    # Generate data with Langmuir-like behaviour
    p_mesh = p_values[:, np.newaxis]
    w_mesh = wavenumbers[np.newaxis, :]
    data = np.exp(-w_mesh / 50) * (p_mesh / (1 + p_mesh)) * 100
    data += rng.normal(0, 0.5, data.shape)

    ds = scp.NDDataset(data)
    ds.set_coordset(
        y=scp.Coord(p_values, title="Pressure"),
        x=scp.Coord(wavenumbers, title="Wavenumber"),
    )
    return ds


# ------------------------------------------------------------------
# Plugin registration tests
# ------------------------------------------------------------------


def test_plugin_metadata():
    """Plugin declares IRIS-related capabilities."""
    plugin = IrisPlugin()
    assert plugin.name == "iris"
    assert plugin.version == "0.1.0"
    assert plugin.description
    assert PluginCapability.ANALYSIS in plugin.capabilities
    assert PluginCapability.VISUALIZER in plugin.capabilities


def test_plugin_compatibility():
    """Plugin passes full compatibility check."""
    plugin = IrisPlugin()
    issues = check_plugin_compatibility(plugin)
    assert not issues, f"Compatibility issues: {issues}"


def test_registration():
    """Plugin registers analyses and visualizers via the plugin system."""
    harness = PluginTestHarness()
    harness.register(IrisPlugin())

    analyses = harness.registry.extensions.list_category("analysis")
    assert "batch_iris" in analyses
    assert "compare_kernels" in analyses
    assert "iris_report" in analyses

    viz = harness.get_visualizer("plot_iris_comparison")
    assert viz is not None
    viz = harness.get_visualizer("plot_iris_distribution_grid")
    assert viz is not None


def test_lifecycle_state():
    """Plugin is ACTIVE after registration."""
    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    assert harness.get_plugin_state("iris") == PluginState.ACTIVE


def test_capability_query():
    """Plugin contributions are discoverable via capability query."""
    harness = PluginTestHarness()
    harness.register(IrisPlugin())

    results = harness.registry.get_by_capability(PluginCapability.ANALYSIS)
    names = [r["name"] for r in results]
    assert "batch_iris" in names
    assert "compare_kernels" in names
    assert "iris_report" in names


def test_package_namespace_uses_isolated_plugin_manager(monkeypatch):
    """scp.iris exposes package-level IRIS APIs from the registered plugin."""
    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "iris", raising=False)

    assert scp.iris.batch_iris is batch_iris_analysis
    assert scp.iris.compare_kernels is compare_kernel_models
    assert scp.iris.iris_report is iris_analysis_report


def test_iris_namespace_does_not_shadow_load_iris(monkeypatch):
    """The IRIS plugin namespace does not collide with the core load_iris API."""
    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "iris", raising=False)

    assert callable(scp.load_iris)
    assert scp.iris.batch_iris is batch_iris_analysis
    assert not hasattr(scp.iris, "load_iris")


def test_iris_namespace_exposes_lazy_module_classes(monkeypatch):
    """scp.iris delegates unknown attributes to the plugin module lazily."""
    import sys

    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "iris", raising=False)
    monkeypatch.delitem(sys.modules, "spectrochempy_iris._core", raising=False)

    namespace = scp.iris
    assert "spectrochempy_iris._core" not in sys.modules
    assert "IRIS" in dir(namespace)
    assert "IrisKernel" in dir(namespace)
    assert "spectrochempy_iris._core" not in sys.modules

    iris_class = namespace.IRIS
    assert iris_class.__name__ == "IRIS"
    assert "spectrochempy_iris._core" in sys.modules

    kernel_class = namespace.IrisKernel
    assert kernel_class.__name__ == "IrisKernel"


def test_iris_classes_are_not_exposed_at_scp_root(monkeypatch):
    """Plugin module attributes stay under scp.iris, not at the scp root."""
    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "IRIS", raising=False)

    with pytest.raises(AttributeError):
        _ = scp.IRIS


# ------------------------------------------------------------------
# IRIS analysis function tests (use synthetic data)
# ------------------------------------------------------------------


def test_iris_analysis_report():
    """iris_analysis_report produces expected summary."""
    from spectrochempy_iris import IRIS
    from spectrochempy_iris import IrisKernel

    ds = _make_test_dataset()
    kernel = IrisKernel(ds, "langmuir", q=[-8, 1, 10])
    iris = IRIS()
    iris.fit(ds, kernel)

    report = iris_analysis_report(iris)
    assert "kernel_type" in report
    assert "n_lambdas" in report
    assert "lambda_values" in report
    assert "RSS_range" in report
    assert "SM_range" in report
    assert "q_range" in report
    assert "n_channels" in report
    assert report["n_lambdas"] >= 1
    assert report["n_channels"] == ds.data.shape[-1]


def test_batch_iris_analysis():
    """batch_iris_analysis runs IRIS on multiple datasets."""
    ds1 = _make_test_dataset(n_spectra=10, n_channels=20)
    ds2 = _make_test_dataset(n_spectra=10, n_channels=20)

    results = batch_iris_analysis(
        {"low_pressure": ds1, "high_pressure": ds2},
        kernel_type="langmuir",
        q=[-6, 1, 8],
    )

    assert len(results) == 2
    assert results[0]["label"] == "low_pressure"
    assert results[1]["label"] == "high_pressure"
    for r in results:
        assert "iris" in r
        assert "f" in r
        assert "RSS" in r
        assert "SM" in r


def test_batch_iris_list_input():
    """batch_iris_analysis accepts a list of datasets."""
    ds1 = _make_test_dataset(n_spectra=8, n_channels=15)
    ds2 = _make_test_dataset(n_spectra=8, n_channels=15)

    results = batch_iris_analysis([ds1, ds2], kernel_type="ca", q=[-6, 1, 6])
    assert len(results) == 2


def test_compare_kernel_models():
    """compare_kernel_models fits multiple kernels and returns results."""
    ds = _make_test_dataset(n_spectra=10, n_channels=20)

    results = compare_kernel_models(
        ds,
        kernels=["langmuir", "ca"],
        q=[-6, 1, 6],
    )

    assert len(results) == 2
    assert results[0]["kernel"] == "langmuir"
    assert results[1]["kernel"] == "ca"
    for r in results:
        assert "iris" in r
        assert r["iris"]._fitted


def test_compare_kernel_defaults():
    """compare_kernel_models uses default kernels when none specified."""
    ds = _make_test_dataset(n_spectra=8, n_channels=15)

    results = compare_kernel_models(ds, q=[-6, 1, 6])
    assert len(results) >= 2


def test_extra_kernels():
    """Plugin-provided custom kernels (freundlich, temkin) are callable."""
    from spectrochempy_iris import IRIS
    from spectrochempy_iris import IrisKernel
    from spectrochempy_iris import _kernel_freundlich
    from spectrochempy_iris import _kernel_temkin

    ds = _make_test_dataset(n_spectra=8, n_channels=15)

    for fn in (_kernel_freundlich, _kernel_temkin):
        kernel = IrisKernel(ds, fn, q=[-6, 1, 6])
        iris = IRIS()
        # Should not raise
        iris.fit(ds, kernel)
        assert iris._fitted


# ------------------------------------------------------------------
# Dataset accessor tests
# ------------------------------------------------------------------


def test_ndd_kernel_accessor():
    """NDDataset.iris_kernel_matrix returns an IrisKernel."""

    ds = _make_test_dataset()

    # Direct function call (the accessor pattern)
    result = IrisPlugin().register_accessors()[0]["func"](
        ds, kernel_type="langmuir", q=[-6, 1, 6]
    )
    assert result is not None
    # It should be an IrisKernel or similar
    assert hasattr(result, "kernel")


def test_namespaced_dataset_accessor(monkeypatch):
    """NDDataset.iris.kernel_matrix returns an IrisKernel."""
    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)

    ds = _make_test_dataset()
    result = ds.iris.kernel_matrix(kernel_type="langmuir", q=[-6, 1, 6])
    assert result is not None
    assert hasattr(result, "kernel")
    assert result._X.shape == ds.shape
    np.testing.assert_allclose(result._X.data, ds.data)

    legacy = ds.iris_kernel_matrix(kernel_type="langmuir", q=[-6, 1, 6])
    assert legacy is not None
    assert hasattr(legacy, "kernel")
    assert legacy._X.shape == ds.shape
    np.testing.assert_allclose(legacy._X.data, ds.data)


# ------------------------------------------------------------------
# Visualisation smoke tests
# ------------------------------------------------------------------


def test_plot_kernel_comparison():
    """plot_kernel_comparison creates a figure without error."""
    ds = _make_test_dataset(n_spectra=8, n_channels=15)

    results = compare_kernel_models(ds, kernels=["langmuir", "ca"], q=[-6, 1, 6])
    fig, axes = plot_kernel_comparison(results)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_distribution_grid():
    """plot_distribution_grid creates a figure without error."""
    ds1 = _make_test_dataset(n_spectra=8, n_channels=15)
    ds2 = _make_test_dataset(n_spectra=8, n_channels=15)

    results = batch_iris_analysis(
        {"A": ds1, "B": ds2},
        kernel_type="langmuir",
        q=[-6, 1, 6],
    )
    fig, axes = plot_distribution_grid(results)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
