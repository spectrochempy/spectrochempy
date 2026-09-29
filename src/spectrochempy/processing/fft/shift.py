# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Shift and frequency-shift functions operating on the selected dataset dimension.

Adapted from NMRGLUE proc_base (New BSD License)

Architecture note
-----------------
The implementation is generic array/signal processing: point shifts, circular
shifts and Fourier-domain shifts do not depend on NMR metadata or hypercomplex
storage.  The NMR origin is historical and remains only in some terminology and
docstrings.  If the NMR plugin later needs Bruker/nmrglue-compatible aliases or
domain-specific defaults, those aliases should be added in the plugin while
these generic kernels stay in core.
"""

__all__ = ["rs", "ls", "roll", "cs", "fsh", "fsh2", "dc"]
__dataset_methods__ = __all__

import numpy as np

from spectrochempy.utils.decorators import _units_agnostic_method

pi = np.pi


def _right_shift(array, pts=0.0, **kwargs):
    points = int(pts)
    shifted = np.roll(array, points, axis=-1)
    shifted[..., :points] = 0
    return shifted


def _left_shift(array, pts=0.0, **kwargs):
    points = int(pts)
    shifted = np.roll(array, -points, axis=-1)
    if points:
        shifted[..., -points:] = 0
    return shifted


def _roll(array, pts=0.0, neg=False, **kwargs):
    points = int(pts)
    shifted = np.roll(array, points, axis=-1)
    if neg and points:
        if points > 0:
            shifted[..., :points] = -shifted[..., :points]
        else:
            shifted[..., points:] = -shifted[..., points:]
    return shifted


def _roll_mask(mask, pts=0.0, **kwargs):
    return np.roll(mask, int(pts), axis=-1)


def _dc_tail_selection(length, size):
    """Return the executed tail slice and its normalized number of points."""
    selection = slice(-int(length * size), None)
    start, stop, step = selection.indices(size)
    return selection, len(range(start, stop, step))


def _dc_history_parameters(parameters, *, axis_size):
    """Describe the tail that the DC kernel actually averages."""
    _, tail_points = _dc_tail_selection(parameters["len"], axis_size)
    return {"tail_points": tail_points}


# ======================================================================================
# Public methods
# ======================================================================================
@_units_agnostic_method(mask_transform=_right_shift, structured_history=True)
def rs(dataset, pts=0.0, **kwargs):
    """
    Right shift and zero fill.

    For multidimensional NDDataset,
    the shift is by default performed on the last dimension.

    Parameters
    ----------
    dataset : nddataset
        NDDataset to be right-shifted.
    pts : int
        Number of points to right shift.

    Returns
    -------
    dataset
        Dataset right shifted and zero filled.
        Masked source points move with their values; introduced zeros are
        valid, unmasked points.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the shift method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    roll : shift without zero filling.

    """
    return _right_shift(dataset, pts=pts)


@_units_agnostic_method(mask_transform=_left_shift, structured_history=True)
def ls(dataset, pts=0.0, **kwargs):
    """
    Left shift and zero fill.

    For multidimensional NDDataset,
    the shift is by default performed on the last dimension.

    Parameters
    ----------
    dataset : nddataset
        NDDataset to be left-shifted.
    pts : int
        Number of points to right shift.

    Returns
    -------
    `NDDataset`
        Modified dataset.
        Masked source points move with their values; introduced zeros are
        valid, unmasked points.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the shift method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    roll : shift without zero filling.

    """
    return _left_shift(dataset, pts=pts)


# no decorator as it delegate to roll
def cs(dataset, pts=0.0, neg=False, **kwargs):
    """
    Circular shift.

    For multidimensional NDDataset,
    the shift is by default performed on the last dimension.

    Parameters
    ----------
    dataset : nddataset
        NDDataset to be shifted.
    pts : int
        Number of points toshift.
    neg : bool
        True to negate the shifted points.

    Returns
    -------
    dataset
        Dataset shifted.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the shift method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    roll : shift without zero filling.

    """
    return roll(dataset, pts=pts, neg=neg, **kwargs)


@_units_agnostic_method(mask_transform=_roll_mask, structured_history=True)
def roll(dataset, pts=0.0, neg=False, **kwargs):
    """
    Roll dimensions.

    For multidimensional NDDataset,
    the shift is by default performed on the last dimension.

    Parameters
    ----------
    dataset : nddataset
        Dataset to be shifted.
    pts : int
        Number of points toshift.
    neg : bool
        True to negate the shifted points.

    Returns
    -------
    dataset
        Dataset shifted.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the shift method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    ls, rs, cs, fsh, fsh2

    """
    return _roll(dataset, pts=pts, neg=neg)


@_units_agnostic_method(structured_history=True)
def fsh(dataset, pts, **kwargs):
    """
    Frequency shift by Fourier transform. Negative signed phase correction.

    For multidimensional NDDataset,
    the shift is by default performed on the last dimension.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    pts : float
        Number of points to frequency shift the data.  Positive value will
        shift the spectrum to the right, negative values to the left.

    Returns
    -------
    dataset
        dataset shifted.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the shift method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    ls, rs, cs, roll, fsh2

    """
    from spectrochempy.processing.fft.fft import _fft
    from spectrochempy.processing.fft.fft import _ifft

    s = float(dataset.shape[-1])

    data = _ifft(dataset)
    data = np.exp(-2.0j * pi * pts * np.arange(s) / s) * data
    return _fft(data)


@_units_agnostic_method(structured_history=True)
def fsh2(dataset, pts, **kwargs):
    """
    Frequency Shift by Fourier transform. Positive signed phase correction.

    For multidimensional NDDataset,
    the shift is by default performed on the last dimension.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    pts : float
        Number of points to frequency shift the data.  Positive value will
        shift the spectrum to the right, negative values to the left.

    Returns
    -------
    dataset
        dataset shifted.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'
        Specify on which dimension to apply the shift method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, keyword parameter, optional, default=False
        True if we make the transform inplace.  If False, the function return a new dataset.

    See Also
    --------
    ls, rs, cs, roll, fsh2

    """
    s = float(dataset.shape[-1])

    data = np.fft.fft(np.fft.ifftshift(dataset, -1)) * dataset.shape[-1]
    data = np.exp(2.0j * pi * pts * np.arange(s) / s) * data
    return np.fft.fftshift(np.fft.ifft(data).astype(data.dtype)) * data.shape[-1]


@_units_agnostic_method(
    structured_history=True,
    history_description="DC baseline correction performed",
    history_parameter_transform=_dc_history_parameters,
)
def dc(dataset, len=0.25, **kwargs):
    """
    Time domain baseline correction.

    Parameters
    ----------
    dataset : nddataset
        The time domain dataset to be corrected.
    len : float, optional
        Fraction of the final part of the selected dimension to average.

    Returns
    -------
    dc
        DC corrected array.

    """
    selection, _ = _dc_tail_selection(len, dataset.shape[-1])
    dc = np.mean(np.atleast_2d(dataset)[..., selection])
    dataset -= dc

    return dataset
