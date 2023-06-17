#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHITTAKER-EILERS SMOOTHER in Python 3 using numpy and scipy

based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother",
        Anal. Chem. 2003, (75), 3631-3636
coded by M. H. V. Werts (CNRS, France)
tested on Anaconda 64-bit (Python 3.6.4, numpy 1.14.0, scipy 1.0.0)

Warm thanks go to Simon Bordeyne who pioneered a first (non-sparse) version
of the smoother in Python.

# This code was downloaded from
# https://github.com/mhvwerts/whittaker-eilers-smoother/blob/master/whittaker_smooth.py
# See Licence in the LICENCES folder of the repository
# (LICENCES/WHITTAKER_SMOOTH_LICENCE.rst)
# and has been slightly modified.

"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu


def speyediff(N, d, format="csc"):
    """
    (utility function)
    Construct a d-th order sparse difference matrix based on
    an initial N x N identity matrix

    Final matrix (N-d) x N
    """

    # assert not (d < 0), "d must be non negative"  (this will be checked elsewhere)
    shape = (N - d, N)
    diagonals = np.zeros(2 * d + 1)
    diagonals[d] = 1.0
    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d + 1)
    spmat = sparse.diags(diagonals, offsets, shape, format=format)
    return spmat


def whittaker_smooth(y, lmbd, d=2):
    """
    Implementation of the Whittaker smoothing algorithm,
    based on the work by Eilers [1].

    [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636

    The larger 'lmbd', the smoother the data.
    For smoothing of a complete data series, sampled at equal intervals

    This implementation uses sparse matrices enabling high-speed processing
    of large input vectors

    ---------

    Arguments :

    y       : vector containing raw data
    lmbd    : parameter for the smoothing algorithm (roughness penalty)
    d       : order of the smoothing

    ---------

    Returns :

    z       : vector of the smoothed data.
    """

    m = len(y)
    E = sparse.eye(m, format="csc")
    D = speyediff(m, d, format="csc")
    coefmat = E + lmbd * D.conj().T.dot(D)
    z = splu(coefmat).solve(y)
    return z
