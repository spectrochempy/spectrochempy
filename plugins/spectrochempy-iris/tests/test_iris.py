# ruff: noqa: S101, PLC0415  # assert/local imports allowed in plugin tests

"""Tests for spectrochempy-iris plugin — extended IRIS analysis."""

from __future__ import annotations

import warnings
from importlib.metadata import version

import numpy as np
import pytest
from spectrochempy_iris import IrisPlugin
from spectrochempy_iris import batch_iris_analysis
from spectrochempy_iris import compare_kernel_models
from spectrochempy_iris import iris_analysis_report
from spectrochempy_iris import plot_distribution_grid
from spectrochempy_iris import plot_kernel_comparison

from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import PluginState
from spectrochempy.api.plugins import check_plugin_compatibility
from spectrochempy.testing.plugins import PluginTestHarness
from spectrochempy.utils.exceptions import NotFittedError

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


@pytest.fixture()
def fitted_iris():
    """Return a fitted IRIS estimator using a small synthetic dataset."""
    from spectrochempy_iris import IRIS
    from spectrochempy_iris import IrisKernel

    ds = _make_test_dataset(n_spectra=8, n_channels=15)
    kernel = IrisKernel(ds, "langmuir", q=[-6, 1, 6])
    iris = IRIS()
    iris.fit(ds, kernel)
    return iris


# ------------------------------------------------------------------
# Plugin registration tests
# ------------------------------------------------------------------


def test_plugin_metadata():
    """Plugin declares IRIS-related capabilities."""
    plugin = IrisPlugin()
    assert plugin.name == "iris"
    assert plugin.version == version("spectrochempy-iris")
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
    import sys

    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "iris", raising=False)
    monkeypatch.delitem(sys.modules, "spectrochempy.iris", raising=False)

    assert scp.iris.batch_iris is batch_iris_analysis
    assert scp.iris.compare_kernels is compare_kernel_models
    assert scp.iris.iris_report is iris_analysis_report


def test_iris_namespace_does_not_shadow_load_iris(monkeypatch):
    """The IRIS plugin namespace does not collide with the core load_iris API."""
    import sys

    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "iris", raising=False)
    monkeypatch.delitem(sys.modules, "spectrochempy.iris", raising=False)

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
    monkeypatch.delitem(sys.modules, "spectrochempy.iris", raising=False)
    monkeypatch.delitem(sys.modules, "spectrochempy_iris._core", raising=False)

    namespace = scp.iris
    assert "spectrochempy_iris._core" not in sys.modules
    assert "IRIS" in dir(namespace)
    assert "IrisKernel" in dir(namespace)
    assert "spectrochempy_iris._core" not in sys.modules

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        iris_class = namespace.IRIS
    assert iris_class.__name__ == "IRIS"
    assert captured == []
    assert "spectrochempy_iris._core" in sys.modules

    kernel_class = namespace.IrisKernel
    assert kernel_class.__name__ == "IrisKernel"


def test_from_spectrochempy_import_iris_supports_lazy_classes(monkeypatch):
    """Importing iris from spectrochempy returns the same lazy namespace API."""
    import sys

    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "iris", raising=False)
    monkeypatch.delitem(sys.modules, "spectrochempy.iris", raising=False)

    from spectrochempy import iris

    assert iris.IRIS.__name__ == "IRIS"
    assert iris.IrisKernel.__name__ == "IrisKernel"


# ------------------------------------------------------------------
# Root-level compatibility alias tests
# ------------------------------------------------------------------


_IRIS_ROOT_ALIASES = [
    ("IRIS", "spectrochempy_iris._core"),
    ("IrisKernel", "spectrochempy_iris._core"),
    ("batch_iris", None),
    ("compare_kernels", None),
    ("iris_report", None),
]


def test_iris_namespaced_api_no_warning(monkeypatch):
    """scp.iris.* public objects are accessible without DeprecationWarning."""
    import sys

    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, "iris", raising=False)
    monkeypatch.delitem(sys.modules, "spectrochempy.iris", raising=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        assert scp.iris.IRIS.__name__ == "IRIS"
        assert scp.iris.IrisKernel.__name__ == "IrisKernel"
        assert scp.iris.batch_iris is batch_iris_analysis
        assert scp.iris.compare_kernels is compare_kernel_models
        assert scp.iris.iris_report is iris_analysis_report

    deprecation_warnings = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert (
        deprecation_warnings == []
    ), f"Expected no DeprecationWarning from namespaced API, got: {deprecation_warnings}"


@pytest.mark.parametrize("alias,heavy_module", _IRIS_ROOT_ALIASES)
def test_iris_root_alias_warns_once(monkeypatch, alias, heavy_module):
    """scp.<alias> works as a compatibility alias and emits DeprecationWarning once."""
    import sys

    import spectrochempy as scp

    harness = PluginTestHarness()
    harness.register(IrisPlugin())
    monkeypatch.setattr(scp, "plugin_manager", harness.manager)
    monkeypatch.setattr(scp, "registry", harness.registry)
    monkeypatch.delitem(scp.__dict__, alias, raising=False)
    if heavy_module:
        monkeypatch.delitem(sys.modules, heavy_module, raising=False)
    scp._EMITTED_PLUGIN_ROOT_WARNINGS.discard(alias)

    if heavy_module:
        assert heavy_module not in sys.modules

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        val1 = getattr(scp, alias)
        val2 = getattr(scp, alias)

    assert val1 is val2
    assert len(captured) == 1
    assert captured[0].category is DeprecationWarning
    assert f"scp.{alias} is deprecated since SpectroChemPy 0.9.0" in str(
        captured[0].message
    )
    assert "will be removed in 0.11.0" in str(captured[0].message)
    assert f"scp.iris.{alias}" in str(captured[0].message)

    if heavy_module:
        assert heavy_module in sys.modules


def test_iris_analysis_report():
    """iris_analysis_report produces expected summary."""
    from spectrochempy_iris import IRIS  # noqa: PLC0415
    from spectrochempy_iris import IrisKernel  # noqa: PLC0415

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


class TestIRISResult:
    """IRIS follows the core runtime Result contract."""

    def test_result_is_analysis_result(self, fitted_iris):
        assert isinstance(fitted_iris.result, AnalysisResult)
        assert fitted_iris.result.estimator == "IRIS"

    def test_outputs_match_direct_accessors(self, fitted_iris):
        result = fitted_iris.result
        np.testing.assert_allclose(result.f.data, fitted_iris.f.data)
        assert result.K is fitted_iris.K

    def test_diagnostics_match_direct_accessors(self, fitted_iris):
        result = fitted_iris.result
        assert result.RSS is fitted_iris.RSS
        assert result.SM is fitted_iris.SM
        assert result.lambdas is fitted_iris.lambdas

    def test_attribute_access_and_discovery(self, fitted_iris):
        result = fitted_iris.result
        assert set(result.outputs) == {"f", "K"}
        assert set(result.diagnostics) == {"RSS", "SM", "lambdas"}
        assert result.f is result.outputs["f"]
        assert result.RSS is result.diagnostics["RSS"]
        assert {"f", "K", "RSS", "SM", "lambdas"} <= set(dir(result))

    def test_parameters_describe_solver_and_regularization(self, fitted_iris):
        parameters = fitted_iris.result.parameters
        assert parameters == {
            "qpsolver": "osqp",
            "reg_par": None,
            "warm_start": False,
            "regularization": False,
            "search_reg": False,
        }

    def test_result_raises_before_fit(self):
        from spectrochempy_iris import IRIS

        with pytest.raises(NotFittedError):
            _ = IRIS().result

    def test_existing_direct_accessors_remain_available(self, fitted_iris):
        assert fitted_iris.f.shape == (1, 6, 15)
        assert fitted_iris.K.shape == (8, 6)
        assert fitted_iris.q.size == 6
        assert fitted_iris.RSS.shape == (1,)
        assert fitted_iris.SM.shape == (1,)


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
        assert "result" in r
        assert "f" in r
        assert "RSS" in r
        assert "SM" in r
        assert r["f"] is r["result"].f
        assert r["RSS"] is r["result"].RSS
        assert r["SM"] is r["result"].SM


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
        assert "result" in r
        assert isinstance(r["result"], AnalysisResult)
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
