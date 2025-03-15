# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np


# ======================================================================================
# Public methods
# ======================================================================================
def get_n_decimals(n, sigdigits=3):
    """
    Return the number of significant decimals to use when rounding a float.

    Parameters
    ----------
    n : float
    sigdigits : int, optional, default: 3
        Number of significant digits.

    Returns
    -------
    int
        number of significant decimals to use when rounding float.
    """
    try:
        n_decimals = sigdigits - int(np.floor(np.log10(abs(n)))) - 1
    except OverflowError:
        n_decimals = 2
    return n_decimals


def spacings(arr, sd=4):
    """
    Return the spacing in the one-dimensional input array.

    Return a scalar for the spacing in the one-dimensional input array
    (if it is uniformly spaced, else return an array of the different spacings.

    Parameters
    ----------
    arr : 1D np.array
    sd : int, optional, default: 4
        Number of significant digits.

    Returns
    -------
    float or array
        Spacing or list of spacing in the given array.
    """
    if len(arr) == 0:
        raise ValueError("Input array cannot be empty.")
    if len(arr) == 1:
        return 0

    spacings = np.diff(arr)
    # we need to take into account only the significant digits
    nd = get_n_decimals(spacings.max(), sd)
    spacings = list(set(np.around(spacings, nd)))

    if len(spacings) == 1:
        # uniform spacing
        return spacings[0]
    return spacings


def gt_eps(arr):
    """
    Check that an array has at least some values greater than epsilon.

    Parameters
    ----------
    arr : array to check

    Returns
    -------
    bool : results of checking
        True means that at least some values are greater than epsilon.
    """
    from spectrochempy.utils.constants import EPSILON

    return np.any(arr > EPSILON)


def largest_power_of_2(value):
    """
    Find the nearest power of two equal to or larger than a value.

    Parameters
    ----------
    value : int
        Value to find the nearest power of two equal to or larger than.

    Returns
    -------
    pw : int
        Power of 2.
    """
    return int(pow(2, np.ceil(np.log(value) / np.log(2))))
