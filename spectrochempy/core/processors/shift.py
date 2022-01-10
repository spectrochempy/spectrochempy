# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

"""
A collection of NMR spectral processing functions which operate on the last
dimension (1) of 2D arrays.

Adapted from NMRGLUE proc_base (New BSD License)
"""

__all__ = ["rs", "ls", "roll", "cs", "fsh", "fsh2"]
__dataset_methods__ = __all__

import numpy as np

from spectrochempy.core.processors.utils import _units_agnostic_method

pi = np.pi


# ======================================================================================================================
# Public methods
# ======================================================================================================================


@_units_agnostic_method
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
    data = np.roll(dataset, int(pts))
    data[..., : int(pts)] = 0
    return data


@_units_agnostic_method
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
    dataset
        dataset left shifted and zero filled.

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
    data = np.roll(dataset, -int(pts))
    data[..., -int(pts) :] = 0
    return data


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

    return roll(dataset, pts, neg, **kwargs)


@_units_agnostic_method
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
    data = np.roll(dataset, int(pts))
    if neg:
        if pts > 0:
            data[..., : int(pts)] = -data[..., : int(pts)]
        else:
            data[..., int(pts) :] = -data[..., int(pts) :]
    return data


@_units_agnostic_method
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
        dataset shifted

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
    from spectrochempy.core.processors.fft import _fft, _ifft

    s = float(dataset.shape[-1])

    data = _ifft(dataset)
    data = np.exp(-2.0j * pi * pts * np.arange(s) / s) * data
    data = _fft(data)

    return data


@_units_agnostic_method
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
        dataset shifted

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

    from spectrochempy.core.processors.fft import _fft_positive, _ifft_positive

    s = float(dataset.shape[-1])

    data = _ifft_positive(dataset)
    data = np.exp(2.0j * pi * pts * np.arange(s) / s) * data
    data = _fft_positive(data)

    return data
