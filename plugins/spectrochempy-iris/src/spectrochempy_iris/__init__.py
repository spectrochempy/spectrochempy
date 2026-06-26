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

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin

# ------------------------------------------------------------------
# Lazy module-level access for public API (IRIS, IrisKernel, plotting)
# ------------------------------------------------------------------


def __getattr__(name: str):
    if name in ("IRIS", "IrisKernel"):
        from ._core import IRIS as _IRIS  # noqa: PLC0415
        from ._core import IrisKernel as _IrisKernel  # noqa: PLC0415

        return _IRIS if name == "IRIS" else _IrisKernel

    # Functions defined in _plotting.py
    if name in {"plot_iris_distribution", "plot_iris_lcurve", "plot_iris_merit"}:
        from ._plotting import plot_iris_distribution as _f1  # noqa: PLC0415
        from ._plotting import plot_iris_lcurve as _f2  # noqa: PLC0415
        from ._plotting import plot_iris_merit as _f3  # noqa: PLC0415

        _mapping = {
            "plot_iris_distribution": _f1,
            "plot_iris_lcurve": _f2,
            "plot_iris_merit": _f3,
        }
        return _mapping[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return [
        "IRIS",
        "IrisKernel",
        "IrisPlugin",
        "batch_iris_analysis",
        "compare_kernel_models",
        "iris_analysis_report",
        "plot_distribution_grid",
        "plot_iris_distribution",
        "plot_iris_lcurve",
        "plot_iris_merit",
        "plot_kernel_comparison",
    ]


if TYPE_CHECKING:
    from spectrochempy import NDDataset


# ------------------------------------------------------------------
# Plugin class
# ------------------------------------------------------------------


class IrisPlugin(SpectroChemPyPlugin):
    """
    Extended IRIS analysis plugin for SpectroChemPy.

    Adds custom kernels, batch processing, model comparison, and
    extended visualisation on top of the core IRIS class.
    """

    name = "iris"
    version = "0.1.5"
    description = "Extended IRIS analysis: custom kernels, batch analysis, model comparison, enhanced plots"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [
        PluginCapability.ANALYSIS,
        PluginCapability.VISUALIZER,
        PluginCapability.ACCESSOR,
    ]
    root_exports = {
        "IRIS": {
            "target": "IRIS",
            "deprecated": True,
            "replacement": "scp.iris.IRIS",
        },
        "IrisKernel": {
            "target": "IrisKernel",
            "deprecated": True,
            "replacement": "scp.iris.IrisKernel",
        },
        "batch_iris": {
            "target": "batch_iris",
            "deprecated": True,
            "replacement": "scp.iris.batch_iris",
        },
        "compare_kernels": {
            "target": "compare_kernels",
            "deprecated": True,
            "replacement": "scp.iris.compare_kernels",
        },
        "iris_report": {
            "target": "iris_report",
            "deprecated": True,
            "replacement": "scp.iris.iris_report",
        },
    }

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
                "namespace": "iris",
                "name": "kernel_matrix",
                "legacy_names": ["iris_kernel_matrix"],
                "func": _ndd_build_kernel,
                "description": "Build an IrisKernel from the dataset",
            },
        ]


# ------------------------------------------------------------------
# Helper / kernel functions (deferred imports internally)
# ------------------------------------------------------------------


def _kernel_freundlich(p, q):
    """Freundlich isotherm kernel: K = p^b (with b = exp(q))."""
    import numpy as np  # noqa: PLC0415

    b = np.exp(q)
    return p[:, np.newaxis] ** b[np.newaxis, :]


def _kernel_temkin(p, q):
    """Temkin isotherm kernel: K = -exp(q) * ln(p)."""
    import numpy as np  # noqa: PLC0415

    a = -np.exp(q)
    kernel_data = a[np.newaxis, :] * np.log(p[:, np.newaxis])
    return np.maximum(kernel_data, 0.0)


def _kernel_linear(p, q):
    """Linear interaction kernel: K = p * q."""
    import numpy as np  # noqa: PLC0415

    return p[:, np.newaxis] * q[np.newaxis, :]


def _ndd_build_kernel(
    self: NDDataset,
    kernel_type: str = "langmuir",
    q: list | None = None,
) -> object:
    from ._core import IrisKernel as _IrisKernel  # noqa: PLC0415

    extra_kernels = {
        "freundlich": _kernel_freundlich,
        "temkin": _kernel_temkin,
        "linear": _kernel_linear,
    }
    selected_kernel = extra_kernels.get(kernel_type, kernel_type)
    return _IrisKernel(self, selected_kernel, q=q)


def batch_iris_analysis(
    datasets: list[NDDataset] | dict[str, NDDataset],
    kernel_type: str = "langmuir",
    q: list | None = None,
    reg_par: list | None = None,
) -> list[dict]:
    """
    Run IRIS analysis on multiple datasets or conditions.

    Parameters
    ----------
    datasets : list[NDDataset] or dict[str, NDDataset]
        Datasets to analyse. If a dict, keys are used as condition labels.
    kernel_type : str, optional
        Kernel type name or callable. Default is ``"langmuir"``.
    q : list, optional
        Internal variable parameter vector or ``[start, stop, num]``.
    reg_par : list, optional
        Regularisation parameters (``[min, max]`` or ``[start, stop, num]``).

    Returns
    -------
    list[dict]
        Each dict contains ``label``, ``iris``, ``result``, ``f``, ``RSS``,
        ``SM``.
    """
    from ._core import IRIS as _IRIS  # noqa: PLC0415
    from ._core import IrisKernel as _IrisKernel  # noqa: PLC0415

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
        kernel = _IrisKernel(ds, kernel_fn, q=q)
        iris = _IRIS(reg_par=reg_par)
        iris.fit(ds, kernel)
        result = iris.result

        results.append(
            {
                "label": label,
                "iris": iris,
                "result": result,
                "f": result.f,
                "RSS": result.RSS,
                "SM": result.SM,
            }
        )

    return results


def compare_kernel_models(
    dataset: NDDataset,
    kernels: list[str] | None = None,
    q: list | None = None,
    reg_par: list | None = None,
) -> list[dict]:
    """
    Compare multiple IRIS kernel models on the same dataset.

    Parameters
    ----------
    dataset : NDDataset
        Dataset to fit.
    kernels : list[str], optional
        Kernel names to compare. Defaults to ``["langmuir", "ca", "freundlich"]``.
    q : list, optional
        Internal variable parameter vector or ``[start, stop, num]``.
    reg_par : list, optional
        Regularisation parameters (``[min, max]`` or ``[start, stop, num]``).

    Returns
    -------
    list[dict]
        Each dict contains ``kernel``, ``iris``, ``result``, ``RSS``, ``SM``,
        ``n_components``.
    """
    from ._core import IRIS as _IRIS  # noqa: PLC0415
    from ._core import IrisKernel as _IrisKernel  # noqa: PLC0415

    extra_kernels = {
        "freundlich": _kernel_freundlich,
        "temkin": _kernel_temkin,
        "linear": _kernel_linear,
    }

    if kernels is None:
        kernels = ["langmuir", "ca", "freundlich"]

    results = []
    for name in kernels:
        kernel_fn = extra_kernels.get(name, name)
        kernel = _IrisKernel(dataset, kernel_fn, q=q)
        iris = _IRIS(reg_par=reg_par)
        iris.fit(dataset, kernel)
        result = iris.result

        results.append(
            {
                "kernel": name,
                "iris": iris,
                "result": result,
                "RSS": result.RSS,
                "SM": result.SM,
                "n_components": result.f.shape[0],
            }
        )

    return results


def iris_analysis_report(iris_object: object) -> dict:
    """
    Generate a summary report of an IRIS analysis.

    Parameters
    ----------
    iris_object : IRIS
        A fitted IRIS object.

    Returns
    -------
    dict
        Report containing ``kernel_type``, ``n_lambdas``, ``lambda_values``,
        ``RSS_range``, ``SM_range``, ``q_range``, ``n_channels``.
    """
    import numpy as np  # noqa: PLC0415

    return {
        "kernel_type": str(type(iris_object.K).__name__),
        "n_lambdas": len(iris_object.lambdas),
        "lambda_values": iris_object.lambdas.data,
        "RSS_range": (float(np.min(iris_object.RSS)), float(np.max(iris_object.RSS))),
        "SM_range": (float(np.min(iris_object.SM)), float(np.max(iris_object.SM))),
        "q_range": (float(iris_object.q.data[0]), float(iris_object.q.data[-1])),
        "n_channels": iris_object.f.shape[-1],
    }


def plot_kernel_comparison(
    comparison_results: list[dict],
    **kwargs,
) -> tuple:
    """
    Plot a side-by-side comparison of multiple kernel fits.

    Parameters
    ----------
    comparison_results : list[dict]
        Output from :func:`compare_kernel_models`.
    **kwargs
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray of Axes)
        The figure and axes array.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    n = len(comparison_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for i, result in enumerate(comparison_results):
        f = result["iris"].f
        if f is not None and f.data.ndim >= 2:
            dist = f[0].squeeze().data
            if dist.ndim == 1:
                axes[i].plot(f[0].y.data, dist)
            elif dist.ndim >= 2:
                axes[i].plot(f[0].y.data, dist[:, 0])
            axes[i].set_title(f"Kernel: {result['kernel']}")
            axes[i].set_xlabel("q")
            axes[i].set_ylabel("f(q)")
            rss = float(np.mean(result["RSS"]))
            axes[i].text(
                0.05,
                0.95,
                f"RSS={rss:.2e}",
                transform=axes[i].transAxes,
                va="top",
                fontsize=8,
            )

    fig.tight_layout()
    return fig, axes


def plot_distribution_grid(
    batch_results: list[dict],
    **kwargs,
) -> tuple:
    """
    Plot a grid of distribution functions across conditions and lambdas.

    Parameters
    ----------
    batch_results : list[dict]
        Output from :func:`batch_iris_analysis`.
    **kwargs
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, numpy.ndarray of Axes)
        The figure and axes array.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n_conditions = len(batch_results)
    fig, axes = plt.subplots(
        n_conditions, 1, figsize=(8, 3 * n_conditions), squeeze=False
    )

    for i, result in enumerate(batch_results):
        ax = axes[i, 0]
        f = result["iris"].f
        lambdas = result["iris"].lambdas.data
        for j in range(min(len(lambdas), 5)):
            dist = f[j].squeeze().data
            if dist.ndim >= 1:
                ax.plot(
                    result["iris"].q.data,
                    dist[:, 0] if dist.ndim > 1 else dist,
                    label=rf"$\lambda$={lambdas[j]:.2e}",
                )
        ax.set_title(f"Condition: {result['label']}")
        ax.set_xlabel("q")
        ax.set_ylabel("f(q)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig, axes
