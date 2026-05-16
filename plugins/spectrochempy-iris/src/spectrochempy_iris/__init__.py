# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""
spectrochempy-iris — IRIS analysis for SpectroChemPy.

Provides Integral Inversion for Spectroscopic data (IRIS) analysis,
including the :class:`IrisKernel` and :class:`IRIS` classes,
plus additional kernels, batch analysis, model comparison, and
extended visualisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin

from ._core import IRIS  # noqa: F401 — re-exported for user convenience
from ._core import IrisKernel  # noqa: F401 — re-exported for user convenience

if TYPE_CHECKING:
    from spectrochempy import NDDataset


class IrisPlugin(SpectroChemPyPlugin):
    """Extended IRIS analysis plugin for SpectroChemPy.

    Adds custom kernels, batch processing, model comparison, and
    extended visualisation on top of the core IRIS class.
    """

    name = "iris"
    version = "0.1.0"
    description = "Extended IRIS analysis: custom kernels, batch analysis, model comparison, enhanced plots"
    spectrochempy_min_version = "0.8.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.ANALYSIS, PluginCapability.VISUALIZER, PluginCapability.ACCESSOR]

    def register_analyses(self) -> list[dict]:
        """Declare analysis workflows extending core IRIS."""
        return [
            {
                "name": "batch_iris",
                "func": batch_iris_analysis,
                "description": "Run IRIS analysis on multiple datasets or conditions",
            },
            {
                "name": "compare_kernels",
                "func": compare_kernel_models,
                "description": "Compare multiple IRIS kernel models on the same data",
            },
            {
                "name": "iris_report",
                "func": iris_analysis_report,
                "description": "Generate a summary report of an IRIS analysis",
            },
        ]

    def register_visualizers(self) -> list[dict]:
        """Declare extended IRIS visualisation functions."""
        return [
            {
                "name": "plot_iris_comparison",
                "func": plot_kernel_comparison,
                "description": "Side-by-side comparison of multiple kernel fits",
            },
            {
                "name": "plot_iris_distribution_grid",
                "func": plot_distribution_grid,
                "description": "Grid of distribution functions across lambdas and conditions",
            },
        ]

    def register_accessors(self) -> list[dict]:
        """Declare dataset accessor methods for IRIS."""
        return [
            {
                "name": "iris_kernel_matrix",
                "func": _ndd_build_kernel,
                "description": "Build an IrisKernel from the dataset",
            },
        ]


# ------------------------------------------------------------------
# Dataset accessor implementation
# ------------------------------------------------------------------


def _ndd_build_kernel(
    self: NDDataset,
    kernel_type: str = "langmuir",
    q: list | None = None,
) -> object:
    """Build an IrisKernel from this dataset.

    Parameters
    ----------
    self : NDDataset
        The dataset (used as ``X`` in the kernel).
    kernel_type : str
        Predefined kernel name (``"langmuir"``, ``"ca"``, etc.) or a custom
        kernel name provided by this plugin.
    q : list or None
        Internal variable range.  If three elements, interpreted as
        ``[start, stop, num]`` for ``np.logspace``.

    Returns
    -------
    IrisKernel
    """
    from ._core import IrisKernel

    extra_kernels = {
        "freundlich": _kernel_freundlich,
        "temkin": _kernel_temkin,
        "linear": _kernel_linear,
    }

    selected_kernel = extra_kernels.get(kernel_type, kernel_type)
    return IrisKernel(self, selected_kernel, q=q)


# ------------------------------------------------------------------
# Additional kernel functions (deferred imports)
# ------------------------------------------------------------------


def _kernel_freundlich(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Freundlich isotherm kernel: K = p^b (with b = exp(q))."""
    import numpy as np

    b = np.exp(q)
    kernel_data = p[:, np.newaxis] ** b[np.newaxis, :]
    return kernel_data


def _kernel_temkin(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Temkin isotherm kernel: K = -exp(q) * ln(p)."""
    import numpy as np

    a = -np.exp(q)
    kernel_data = a[np.newaxis, :] * np.log(p[:, np.newaxis])
    kernel_data = np.maximum(kernel_data, 0.0)
    return kernel_data


def _kernel_linear(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Linear interaction kernel: K = p * q."""
    import numpy as np

    kernel_data = p[:, np.newaxis] * q[np.newaxis, :]
    return kernel_data


# ------------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------------


def batch_iris_analysis(
    datasets: list[NDDataset] | dict[str, NDDataset],
    kernel_type: str = "langmuir",
    q: list | None = None,
    reg_par: list | None = None,
) -> list[dict]:
    """Run IRIS analysis on multiple datasets or conditions.

    Parameters
    ----------
    datasets : list[NDDataset] or dict[str, NDDataset]
        Datasets to analyse.  If a dict, keys are used as condition labels.
    kernel_type : str
        Kernel type (``"langmuir"``, ``"ca"``, ``"freundlich"``,
        ``"temkin"``, etc.).
    q : list or None
        Internal variable ``[start, stop, num]`` for ``np.logspace``.
    reg_par : list or None
        Regularisation parameters.

    Returns
    -------
    list[dict] with keys ``label``, ``iris``, ``f``, ``RSS``, ``SM``.
    """
    from ._core import IRIS
    from ._core import IrisKernel

    extra_kernels = {
        "freundlich": _kernel_freundlich,
        "temkin": _kernel_temkin,
        "linear": _kernel_linear,
    }

    if isinstance(datasets, dict):
        items = list(datasets.items())
    else:
        items = [(f"dataset_{i}", ds) for i, ds in enumerate(datasets)]

    results = []
    for label, ds in items:
        kernel_fn = extra_kernels.get(kernel_type, kernel_type)
        kernel = IrisKernel(ds, kernel_fn, q=q)
        iris = IRIS(reg_par=reg_par)
        iris.fit(ds, kernel)

        results.append(
            {
                "label": label,
                "iris": iris,
                "f": iris.f,
                "RSS": iris.RSS,
                "SM": iris.SM,
            }
        )

    return results


def compare_kernel_models(
    dataset: NDDataset,
    kernels: list[str] | None = None,
    q: list | None = None,
    reg_par: list | None = None,
) -> list[dict]:
    """Fit multiple kernel models to the same dataset and compare results.

    Parameters
    ----------
    dataset : NDDataset
        Input spectroscopic data.
    kernels : list[str] or None
        Kernel names to compare.  Defaults to ``["langmuir", "ca",
        "freundlich"]``.
    q : list or None
        Internal variable ``[start, stop, num]``.
    reg_par : list or None
        Regularisation parameters (same for all models).

    Returns
    -------
    list[dict] with keys ``kernel``, ``iris``, ``RSS``, ``SM``,
    ``n_components``.
    """
    from ._core import IRIS
    from ._core import IrisKernel

    extra_kernels = {
        "freundlich": _kernel_freundlich,
        "temkin": _kernel_temkin,
        "linear": _kernel_linear,
    }

    builtin_kernels = ["langmuir", "ca", "diffusion",
                       "reactant-first-order", "product-first-order",
                       "stejskal-tanner"]

    if kernels is None:
        kernels = ["langmuir", "ca", "freundlich"]

    results = []
    for name in kernels:
        kernel_fn = extra_kernels.get(name, name)
        kernel = IrisKernel(dataset, kernel_fn, q=q)
        iris = IRIS(reg_par=reg_par)
        iris.fit(dataset, kernel)

        results.append(
            {
                "kernel": name,
                "iris": iris,
                "RSS": iris.RSS,
                "SM": iris.SM,
                "n_components": iris.f.shape[0],
            }
        )

    return results


def iris_analysis_report(iris_object: object) -> dict:
    """Generate a summary report of a fitted IRIS analysis.

    Parameters
    ----------
    iris_object : IRIS
        A fitted IRIS instance.

    Returns
    -------
    dict with keys ``kernel_type``, ``n_lambdas``, ``lambda_values``,
    ``RSS_range``, ``SM_range``, ``q_range``, ``n_channels``.
    """
    import numpy as np

    return {
        "kernel_type": str(type(iris_object.K).__name__),
        "n_lambdas": len(iris_object.lambdas),
        "lambda_values": iris_object.lambdas.data,
        "RSS_range": (float(np.min(iris_object.RSS)), float(np.max(iris_object.RSS))),
        "SM_range": (float(np.min(iris_object.SM)), float(np.max(iris_object.SM))),
        "q_range": (float(iris_object.q.data[0]), float(iris_object.q.data[-1])),
        "n_channels": iris_object.f.shape[-1],
    }


# ------------------------------------------------------------------
# Visualisation functions (deferred matplotlib imports)
# ------------------------------------------------------------------


def plot_kernel_comparison(
    comparison_results: list[dict],
    **kwargs,
) -> tuple:
    """Side-by-side comparison of distribution functions from multiple kernels.

    Parameters
    ----------
    comparison_results : list[dict]
        Output from ``compare_kernel_models()``.

    Returns
    -------
    (fig, axes) tuple.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(comparison_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for i, result in enumerate(comparison_results):
        f = result["iris"].f
        if f is not None and f.data.ndim >= 2:
            # Plot the first lambda's distribution
            dist = f[0].squeeze().data
            if dist.ndim == 1:
                axes[i].plot(f[0].y.data, dist)
            elif dist.ndim >= 2:
                axes[i].plot(f[0].y.data, dist[:, 0])
            axes[i].set_title(f"Kernel: {result['kernel']}")
            axes[i].set_xlabel("q")
            axes[i].set_ylabel("f(q)")
            rss = float(np.mean(result["RSS"]))
            axes[i].text(0.05, 0.95, f"RSS={rss:.2e}", transform=axes[i].transAxes,
                         va="top", fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_distribution_grid(
    batch_results: list[dict],
    **kwargs,
) -> tuple:
    """Grid of distribution functions across lambdas and conditions.

    Parameters
    ----------
    batch_results : list[dict]
        Output from ``batch_iris_analysis()``.

    Returns
    -------
    (fig, axes) tuple.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_conditions = len(batch_results)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(8, 3 * n_conditions),
                             squeeze=False)

    for i, result in enumerate(batch_results):
        ax = axes[i, 0]
        f = result["iris"].f
        lambdas = result["iris"].lambdas.data
        for j in range(min(len(lambdas), 5)):
            dist = f[j].squeeze().data
            if dist.ndim >= 1:
                ax.plot(result["iris"].q.data, dist[:, 0] if dist.ndim > 1 else dist,
                        label=rf"$\lambda$={lambdas[j]:.2e}")
        ax.set_title(f"Condition: {result['label']}")
        ax.set_xlabel("q")
        ax.set_ylabel("f(q)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig, axes


# Export accessor names so NDDataset can discover them
__dataset_methods__ = ["iris_kernel_matrix"]
