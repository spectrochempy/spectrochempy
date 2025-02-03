#!/usr/bin/env python3
"""
WHITTAKER-EILERS SMOOTHER in Python 3 using numpy and scipy.

This implementation is based on the work by Eilers [1].

References
----------
.. [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636

Notes
-----
Coded by M. H. V. Werts (CNRS, France)
Tested on Anaconda 64-bit (Python 3.6.4, numpy 1.14.0, scipy 1.0.0)

Warm thanks go to Simon Bordeyne who pioneered a first (non-sparse) version
of the smoother in Python.

This code was downloaded from:
https://github.com/mhvwerts/whittaker-eilers-smoother/blob/master/whittaker_smooth.py
See Licence in the LICENCES folder of the repository
(LICENCES/WHITTAKER_SMOOTH_LICENCE.rst)
and has been slightly modified.

"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu


def speyediff(N, d, format="csc"):
    """
    Construct a d-th order sparse difference matrix.

    Parameters
    ----------
    N : int
        Size of the initial N x N identity matrix.
    d : int
        Order of the difference matrix.
    format : str, optional
        Sparse matrix format, by default "csc".

    Returns
    -------
    scipy.sparse.csc_matrix
        A (N-d) x N sparse difference matrix.

    """
    # assert not (d < 0), "d must be non negative"  (this will be checked elsewhere)
    shape = (N - d, N)
    diagonals = np.zeros(2 * d + 1)
    diagonals[d] = 1.0
    for _i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d + 1)
    return sparse.diags(diagonals, offsets, shape, format=format)


def whittaker_smooth(y, lmbd, d=2):
    """
    Whittaker smoothing algorithm implementation.

    The larger 'lmbd', the smoother the data. For smoothing of a complete
    data series, sampled at equal intervals. This implementation uses sparse
    matrices enabling high-speed processing of large input vectors.

    Parameters
    ----------
    y : array-like
        Vector containing raw data.
    lmbd : float
        Parameter for the smoothing algorithm (roughness penalty).
    d : int, optional
        Order of the smoothing, by default 2.

    Returns
    -------
    array-like
        Vector of the smoothed data.

    References
    ----------
    .. [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636

    """
    m = len(y)
    E = sparse.eye(m, format="csc")
    D = speyediff(m, d, format="csc")
    coefmat = E + lmbd * D.conj().T.dot(D)
    return splu(coefmat).solve(y)
