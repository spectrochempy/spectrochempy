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
    "pareto_scale",
    "range_scale",
    "robust_scale",
    "log_transform",
]

__dataset_methods__ = __all__

from spectrochempy.processing.transformation.preprocessing_transformers import (
    AutoscaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    CenterTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    LogTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    MSCTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    NormalizeTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    ParetoScaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    RangeScaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    RobustScaleTransformer,
)
from spectrochempy.processing.transformation.preprocessing_transformers import (
    SNVTransformer,
)


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
    result = NormalizeTransformer(method=method, dim=dim).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result


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
    result = CenterTransformer(dim=dim).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result


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
    result = AutoscaleTransformer(dim=dim).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result


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
    result = SNVTransformer().fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = "snv applied"
        return dataset
    result.history = "snv applied"
    return result


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
    result = MSCTransformer(reference=reference, dim=dim).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result


def pareto_scale(dataset, dim="y", inplace=False):
    r"""
    Apply Pareto scaling along a dimension.

    Pareto scaling is a compromise between mean-centering and autoscaling:
    the data are centered and divided by the square-root of the standard
    deviation.

    .. math::

       x_{ij}^\prime = \frac{x_{ij} - \bar{x}_j}{\sqrt{s_j}}

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the statistics are computed.
    inplace : `bool`, optional, default:`False`
        If `True`, scaling is performed in place.

    Returns
    -------
    `NDDataset`
        The Pareto-scaled dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4y-activation.spg")
    >>> nd = dataset.pareto_scale(dim="y")

    """
    result = ParetoScaleTransformer(dim=dim).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result


def range_scale(dataset, dim="y", inplace=False):
    r"""
    Scale data by the range along a dimension.

    Each variable (or observation) is divided by its range
    (``max - min``).  This is sometimes called *min-max scaling* or
    *range normalisation* in the chemometric literature.

    .. math::

       x_{ij}^\prime = \frac{x_{ij}}{\max(x_j) - \min(x_j)}

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the range is computed.
    inplace : `bool`, optional, default:`False`
        If `True`, scaling is performed in place.

    Returns
    -------
    `NDDataset`
        The range-scaled dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4y-activation.spg")
    >>> nd = dataset.range_scale(dim="y")

    See Also
    --------
    normalize : General normalisation (includes min-max to [0, 1]).

    """
    result = RangeScaleTransformer(dim=dim).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result


def robust_scale(dataset, dim="y", inplace=False):
    r"""
    Apply robust scaling along a dimension.

    The data are centered on the median and scaled by the
    median absolute deviation (MAD).  This makes the scaling resistant
    to outliers.

    .. math::

       x_{ij}^\prime = \frac{x_{ij} - \mathrm{median}(x_j)}{\mathrm{MAD}(x_j)}

    where :math:`\mathrm{MAD} = \mathrm{median}(|x - \mathrm{median}|)`
    and the result is multiplied by 1.4826 so that the MAD estimates
    the standard deviation of a normal distribution.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    dim : `str` or `int`, optional, default:'y'
        Dimension along which the median and MAD are computed.
    inplace : `bool`, optional, default:`False`
        If `True`, scaling is performed in place.

    Returns
    -------
    `NDDataset`
        The robustly-scaled dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4y-activation.spg")
    >>> nd = dataset.robust_scale(dim="y")

    """
    result = RobustScaleTransformer(dim=dim).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result


def log_transform(dataset, method="log1p", eps=1e-10, inplace=False):
    r"""
    Apply a logarithmic transform.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    method : `str`, optional, default:'log1p'
        Transform to apply:

        * ``'log1p'`` — compute ``log(1 + x)`` (stable for small or zero values).
        * ``'log'``   — compute ``log(x)``.  If the data contain values
          :math:`\le 0`, a small offset ``eps`` is added automatically.

    eps : `float`, optional, default:1e-10
        Offset added when ``method='log'`` and non-positive values are present.
    inplace : `bool`, optional, default:`False`
        If `True`, the transform is performed in place.

    Returns
    -------
    `NDDataset`
        The log-transformed dataset.

    Examples
    --------
    >>> dataset = scp.read("irdata/nh4y-activation.spg")
    >>> nd = dataset.log_transform(method="log1p")

    """
    result = LogTransformer(method=method, eps=eps).fit_transform(dataset)
    if inplace:
        dataset._data = result._data
        dataset.history = result.history
        return dataset
    return result
