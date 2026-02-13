# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Plot dispatcher for SpectroChemPy.

This module routes plotting calls to the appropriate backend based on
the specified backend name. Currently supports matplotlib.
"""

from typing import Any, Optional

_BACKEND_REGISTRY = {}


def _register_backend(name: str, module_path: str) -> None:
    """Register a plotting backend."""
    _BACKEND_REGISTRY[name] = module_path


def get_available_backends() -> list[str]:
    """Return list of available plotting backends."""
    return list(_BACKEND_REGISTRY.keys())


def plot_dataset(
    dataset: Any,
    method: Optional[str] = None,
    backend: str = "matplotlib",
    **kwargs: Any,
) -> Any:
    """
    Route plotting to the appropriate backend.

    Parameters
    ----------
    dataset : NDDataset
        The dataset to plot.
    method : str, optional
        Plotting method (e.g., "pen", "stack", "surface").
        If None, method is chosen based on data dimensionality.
    backend : str, optional
        Backend to use ("matplotlib"). Default is "matplotlib".
    **kwargs
        Additional arguments passed to the plotting function.

    Returns
    -------
    Any
        The matplotlib axes or appropriate return value from the backend.
    """
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. Available backends: {get_available_backends()}"
        )

    module_path = _BACKEND_REGISTRY[backend]

    # Lazy import of the backend module
    backend_module = __import__(module_path, fromlist=["plot_dataset_impl"])
    return backend_module.plot_dataset_impl(dataset, method, **kwargs)


_register_backend("matplotlib", "spectrochempy.plot.backends.matplotlib_backend")
