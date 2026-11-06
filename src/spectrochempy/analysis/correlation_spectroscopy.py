# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Two-dimensional correlation spectroscopy."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset

__all__ = ["Correlation2DResult", "correlation_spectroscopy"]


@dataclass(frozen=True)
class Correlation2DResult:
    """
    Maps returned by :func:`correlation_spectroscopy`.

    Attributes
    ----------
    synchronous : NDDataset
        Synchronous correlation map.
    asynchronous : NDDataset
        Asynchronous Hilbert–Noda correlation map.
    perturbation_dim : str
        Dimension used as the ordered perturbation sequence.
    """

    synchronous: NDDataset
    asynchronous: NDDataset
    perturbation_dim: str


def correlation_spectroscopy(
    dataset: NDDataset,
    perturbation_dim: str = "y",
) -> Correlation2DResult:
    r"""
    Calculate synchronous and asynchronous 2D correlation spectra.

    Spectra are mean-centered along the perturbation dimension. The synchronous
    map is calculated as ``A.T @ A / (n - 1)``; the asynchronous map uses the
    Hilbert–Noda matrix ``N[i, j] = 1 / (pi * (j - i))`` for ``i != j`` and
    ``A.T @ N @ A / (n - 1)``. The asynchronous sign follows the input order of
    the perturbation sequence.

    The perturbation coordinate must be real numeric, monotonic, and classified
    as linear by its SpectroChemPy ``Coord``. This classification follows the
    coordinate's ``sigdigits`` and ``linearize_below`` settings. A default
    ``Coord`` uses ``linearize_below=0.1`` and may normalize a near-linear
    source axis before this function sees it. Configure ``linearize_below``
    when constructing the coordinate if source deviations must remain visible
    to this validation.

    Parameters
    ----------
    dataset : NDDataset
        Two-dimensional spectral data. One dimension represents the ordered
        perturbation sequence; the other represents the spectral coordinate.
    perturbation_dim : str, default: "y"
        Dimension containing the perturbation sequence.

    Returns
    -------
    Correlation2DResult
        Result with ``synchronous`` and ``asynchronous`` NDDataset maps. Both
        maps use the input spectral coordinate on their ``y`` and ``x`` axes,
        and their units are the square of the input signal units.

    Raises
    ------
    TypeError
        If *dataset* is not an NDDataset or its values are not real numeric data.
    ValueError
        If the input is not two-dimensional, has fewer than two perturbation
        points, contains masked or non-finite values, or has a complex,
        non-monotonic, or non-uniform perturbation coordinate.

    Notes
    -----
    The Hilbert–Noda transform assumes equally spaced perturbation points. The
    function follows the ``Coord.linear`` classification rather than applying
    a second numeric tolerance to coordinate differences.

    References
    ----------
    I. Noda, "Generalized Two-Dimensional Correlation Method Applicable to
    Infrared, Raman, and Other Types of Spectroscopy", Applied Spectroscopy
    47, 1329–1336 (1993), https://doi.org/10.1366/0003702934067694.
    """
    if not isinstance(dataset, NDDataset):
        raise TypeError("dataset must be an NDDataset")
    if dataset.ndim != 2:
        raise ValueError("dataset must be two-dimensional")

    if not isinstance(perturbation_dim, str):
        raise TypeError("perturbation_dim must be a dimension name")

    dimensions = list(dataset.dims)
    if perturbation_dim not in dimensions:
        raise ValueError(
            f"perturbation_dim must be one of {tuple(dimensions)!r}, "
            f"got {perturbation_dim!r}"
        )

    perturbation_axis = dimensions.index(perturbation_dim)
    spectral_dim = dimensions[1 - perturbation_axis]
    perturbation_coord = dataset.coord(perturbation_dim)
    if not isinstance(perturbation_coord, Coord):
        raise ValueError("perturbation dimension must have a single Coord")
    if np.iscomplexobj(perturbation_coord.data):
        raise ValueError("perturbation coordinate must be real numeric data")
    try:
        perturbation_values = np.asarray(perturbation_coord.data, dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("perturbation coordinate must be numeric") from exc

    if perturbation_values.ndim != 1 or not np.isfinite(perturbation_values).all():
        raise ValueError(
            "perturbation coordinate must be a finite one-dimensional axis"
        )
    if perturbation_values.size < 2:
        raise ValueError("at least two perturbation points are required")

    spacing = np.diff(perturbation_values)
    if not (np.all(spacing > 0.0) or np.all(spacing < 0.0)):
        raise ValueError("perturbation coordinate must be strictly monotonic")
    if spacing.size > 1 and not perturbation_coord.linear:
        raise ValueError(
            "perturbation coordinate must be uniformly spaced; "
            "interpolate the dataset before calculating 2D correlation spectra"
        )

    if np.any(dataset.mask):
        raise ValueError("dataset must not contain masked values")
    if np.iscomplexobj(dataset.data):
        raise TypeError("dataset values must be real numeric data")
    try:
        values = np.asarray(dataset.data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("dataset values must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("dataset must contain only finite values")

    dynamic_spectra = np.moveaxis(values, perturbation_axis, 0)
    dynamic_spectra = dynamic_spectra - dynamic_spectra.mean(axis=0, keepdims=True)
    n_perturbations = dynamic_spectra.shape[0]

    synchronous = dynamic_spectra.T @ dynamic_spectra / (n_perturbations - 1)

    indices = np.arange(n_perturbations)
    index_difference = indices[np.newaxis, :] - indices[:, np.newaxis]
    np.fill_diagonal(index_difference, 1)
    hilbert_noda = 1.0 / (np.pi * index_difference)
    np.fill_diagonal(hilbert_noda, 0.0)
    asynchronous = (dynamic_spectra.T @ hilbert_noda @ dynamic_spectra) / (
        n_perturbations - 1
    )

    spectral_coord = dataset.coord(spectral_dim)

    def _spectral_coordset():
        if spectral_coord is None:
            return None
        return [deepcopy(spectral_coord), deepcopy(spectral_coord)]

    output_units = None if dataset.units is None else dataset.units**2

    synchronous_dataset = NDDataset(
        synchronous,
        coordset=_spectral_coordset(),
        title="Synchronous 2D-COS",
        units=output_units,
    )
    asynchronous_dataset = NDDataset(
        asynchronous,
        coordset=_spectral_coordset(),
        title="Asynchronous 2D-COS",
        units=output_units,
    )

    return Correlation2DResult(
        synchronous=synchronous_dataset,
        asynchronous=asynchronous_dataset,
        perturbation_dim=perturbation_dim,
    )
