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

from typing import Any

_BACKEND_REGISTRY = {}


def _register_backend(name: str, module_path: str) -> None:
    """Register a plotting backend."""
    _BACKEND_REGISTRY[name] = module_path


def get_available_backends() -> list[str]:
    """Return list of available plotting backends."""
    return list(_BACKEND_REGISTRY.keys())


def plot_dataset(
    dataset: Any,
    method: str | None = None,
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

    See Also
    --------
    plot_compare : Compare two datasets with optional residual.
    plot_multiple : Overlay several 1D datasets on a single axes.
    multiplot : Arrange datasets on a grid of axes.

    Notes
    -----
    For ``method="image"``, ``"contour"``, ``"contourf"`` and ``"map"``, the
    contour levels start at ``start * maximum``, where ``maximum`` is the data
    maximum and ``start`` defaults to ``preferences.contour_start`` (0.05). For
    non-negative data, values below ``start * maximum`` are not drawn;
    ``start=0`` draws them. For data with negative values, the levels go
    from ``-maximum`` to ``-start * maximum`` and from ``start * maximum`` to
    ``maximum``: contour and map lines skip values whose magnitude is below
    ``start * maximum``, image and contourf fill this band with a single
    color, and values below ``-maximum`` are not drawn. ``start=0`` currently
    raises an error for such data (duplicated zero level). This assumes a
    positive maximum: for data without positive values, nothing is drawn, or
    an error is raised if the maximum is 0. Pass explicit ``levels`` to
    control the drawn values.

    Axes whose coordinate is in ppm, or in cm⁻¹ with a title that does not
    contain "raman", are displayed in decreasing order (see `Coord.reversed`).
    This applies to the x axis, and to the y axis of image, contour, contourf,
    map, surface and waterfall plots; for stack and lines plots, the y axis
    shows the intensity. Pass ``x_reverse=False`` or ``y_reverse=False`` to
    keep an axis increasing.
    """
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. Available backends: {get_available_backends()}"
        )

    module_path = _BACKEND_REGISTRY[backend]

    # Lazy import of the backend module
    import importlib

    backend_module = importlib.import_module(module_path)
    return backend_module.plot_dataset_impl(dataset, method, **kwargs)


_register_backend("matplotlib", "spectrochempy.plotting.backends.matplotlib_backend")

plot = plot_dataset
