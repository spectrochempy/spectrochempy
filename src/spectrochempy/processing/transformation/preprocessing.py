# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Standard chemometric preprocessing operations.

These functions implement common preprocessing steps used in chemometrics
and spectroscopic data analysis.  They operate along a chosen dimension and
respect SpectroChemPy conventions for masks, units, coordinates, metadata,
history, and inplace behaviour.

Implemented operations
----------------------
- ``normalize`` : scale spectra to unit maximum, area, vector norm, or [0, 1]
- ``center``    : subtract the mean
- ``autoscale`` : subtract the mean and divide by the standard deviation
- ``snv``       : Standard Normal Variate (autoscale per observation)
- ``msc``       : Multiplicative Scatter Correction

"""

__all__ = [
    "normalize",
    "center",
    "autoscale",
    "snv",
    "msc",
]

__dataset_methods__ = __all__

import numpy as np

from spectrochempy.utils.exceptions import SpectroChemPyError


def normalize(dataset, method="max", dim="x", inplace=False):
    r"""
    Normalize data along a dimension.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    method : `str`, optional, default:'max'
        Normalization method:

        * ``'max'``     — divide by the maximum absolute value.
        * ``'sum'``     — divide by the sum of absolute values.
        * ``'vector'``  — divide by the Euclidean (L2) norm.
        * ``'minmax'``  — scale linearly to the range ``[0, 1]``.

    dim : `str` or `int`, optional, default:'x'
        Dimension along which the normalization is computed.
    inplace : `bool`, optional, default:`False`
        If `True`, the normalization is performed in place.

    Returns
    -------
    `NDDataset`
        The normalized dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4.spg")
    >>> nd = dataset.normalize(method="max", dim="x")

    """
    new = dataset.copy() if not inplace else dataset
    axis, dim_name = new.get_axis(dim)

    data = new.masked_data

    if method == "max":
        norm = np.ma.max(np.ma.abs(data), axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        new._data = data / norm
        new.history = f"normalize (max) applied on dimension {dim_name}"

    elif method == "sum":
        norm = np.ma.sum(np.ma.abs(data), axis=axis, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        new._data = data / norm
        new.history = f"normalize (sum) applied on dimension {dim_name}"

    elif method == "vector":
        norm = np.sqrt(np.ma.sum(data**2, axis=axis, keepdims=True))
        norm = np.where(norm == 0, 1, norm)
        new._data = data / norm
        new.history = f"normalize (vector) applied on dimension {dim_name}"

    elif method == "minmax":
        dmin = np.ma.min(data, axis=axis, keepdims=True)
        dmax = np.ma.max(data, axis=axis, keepdims=True)
        rng = dmax - dmin
        rng = np.where(rng == 0, 1, rng)
        new._data = (data - dmin) / rng
        new.history = f"normalize (minmax) applied on dimension {dim_name}"

    else:
        raise SpectroChemPyError(
            f"Unknown normalization method '{method}'. "
            f"Choose from 'max', 'sum', 'vector', 'minmax'."
        )

    return new


def center(dataset, dim="y", inplace=False):
    r"""
    Subtract the mean along a dimension (mean-centering).

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the mean is computed and subtracted.
    inplace : `bool`, optional, default:`False`
        If `True`, centering is performed in place.

    Returns
    -------
    `NDDataset`
        The centered dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4.spg")
    >>> nd = dataset.center(dim="x")

    """
    new = dataset.copy() if not inplace else dataset
    axis, dim_name = new.get_axis(dim)

    mean = np.ma.mean(new.masked_data, axis=axis, keepdims=True)
    new._data = new.masked_data - mean
    new.history = f"center applied on dimension {dim_name}"
    return new


def autoscale(dataset, dim="y", inplace=False):
    r"""
    Mean-center and scale to unit variance along a dimension.

    This is the classic *autoscaling* (or *z-score* / *standard-score*)
    operation used before PCA, PLS, and other multivariate analyses.

    .. math::

       x_{ij}^\prime = \frac{x_{ij} - \bar{x}_j}{s_j}

    where :math:`\bar{x}_j` and :math:`s_j` are the mean and standard
    deviation along the chosen dimension.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the mean and standard deviation are computed.
    inplace : `bool`, optional, default:`False`
        If `True`, autoscaling is performed in place.

    Returns
    -------
    `NDDataset`
        The autoscaled dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4.spg")
    >>> nd = dataset.autoscale(dim="x")

    """
    new = dataset.copy() if not inplace else dataset
    axis, dim_name = new.get_axis(dim)

    data = new.masked_data
    mean = np.ma.mean(data, axis=axis, keepdims=True)
    std = np.ma.std(data, axis=axis, keepdims=True)

    # Avoid division by zero: where std == 0 the centred data is already 0
    std_safe = np.where(std == 0, 1, std)
    new._data = (data - mean) / std_safe
    new.history = f"autoscale applied on dimension {dim_name}"
    return new


def snv(dataset, inplace=False):
    r"""
    Apply Standard Normal Variate (SNV) correction.

    SNV is equivalent to autoscaling each observation (spectrum) individually
    along its spectral axis (``dim='x'``), so that every spectrum has zero
    mean and unit variance.

    .. math::

       x_i^\prime = \frac{x_i - \bar{x}_i}{s_i}

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    inplace : `bool`, optional, default:`False`
        If `True`, SNV is performed in place.

    Returns
    -------
    `NDDataset`
        The SNV-corrected dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4.spg")
    >>> nd = dataset.snv()

    See Also
    --------
    autoscale : General mean-center and unit-variance scaling.

    """
    # SNV is conventionally applied per spectrum (dim='x')
    new = autoscale(dataset, dim="x", inplace=inplace)
    new.history = "snv applied"
    return new


def msc(dataset, reference=None, dim="y", inplace=False):
    r"""
    Multiplicative Scatter Correction (MSC).

    MSC corrects for multiplicative and additive effects caused by
    light-scattering or path-length variations.  Each observation is
    linearly regressed against a reference spectrum and corrected as:

    .. math::

       x_i^\prime = \frac{x_i - a_i}{b_i}

    where :math:`a_i` and :math:`b_i` are the intercept and slope of
    the least-squares fit of observation :math:`i` to the reference.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data (2-D: observations × features).
    reference : `NDDataset` or array-like, optional
        1-D reference spectrum.  If `None`, the mean spectrum is used.
    dim : `str` or `int`, optional, default:'y'
        Dimension that identifies individual observations (spectra).
    inplace : `bool`, optional, default:`False`
        If `True`, MSC is performed in place.

    Returns
    -------
    `NDDataset`
        The MSC-corrected dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4.spg")
    >>> nd = dataset.msc()

    See Also
    --------
    snv : Standard Normal Variate correction.
    autoscale : General mean-center and unit-variance scaling.

    """
    new = dataset.copy() if not inplace else dataset
    axis, dim_name = new.get_axis(dim)

    data = new.masked_data

    if data.ndim != 2:
        raise SpectroChemPyError(
            "msc currently supports only 2-D datasets (observations × features)."
        )

    spectral_axis = 1 if axis == 0 else 0

    # Reference spectrum
    if reference is None:
        ref = np.ma.mean(data, axis=axis)
        ref = np.asarray(ref)  # ensure plain ndarray
    else:
        if hasattr(reference, "masked_data"):
            ref = reference.masked_data
        else:
            ref = np.ma.masked_invalid(np.asarray(reference))
        if ref.ndim != 1:
            raise SpectroChemPyError("msc reference must be a 1-D spectrum.")
        if ref.size != data.shape[spectral_axis]:
            raise SpectroChemPyError(
                f"msc reference size ({ref.size}) does not match "
                f"dataset spectral size ({data.shape[spectral_axis]})."
            )

    # Expand reference for broadcasting: singleton on the observation axis,
    # full size on the spectral axis.
    ref_shape = [1, 1]
    ref_shape[spectral_axis] = -1
    ref_b = ref.reshape(ref_shape)

    # Closed-form least-squares:  x = a + b * ref
    #   b = (n*sum(x*ref) - sum(x)*sum(ref)) / (n*sum(ref^2) - sum(ref)^2)
    #   a = (sum(x) - b*sum(ref)) / n
    n = ref.size
    sum_ref = np.ma.sum(ref)
    sum_ref2 = np.ma.sum(ref**2)
    den = n * sum_ref2 - sum_ref**2

    if den == 0:
        raise SpectroChemPyError(
            "msc denominator is zero; reference spectrum is constant."
        )

    # Per-observation sums over the spectral axis
    sum_x = np.ma.sum(data, axis=spectral_axis, keepdims=True)
    sum_xref = np.ma.sum(data * ref_b, axis=spectral_axis, keepdims=True)

    b = (n * sum_xref - sum_ref * sum_x) / den
    a = (sum_x - b * sum_ref) / n

    # Avoid division by zero on slopes
    b_safe = np.where(b == 0, 1, b)
    new._data = (data - a) / b_safe
    new.history = f"msc applied on dimension {dim_name}"
    return new
