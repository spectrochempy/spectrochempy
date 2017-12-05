# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================



# TODO: create tests

__all__ = ['Lstsq']
_methods = __all__[:]

import numpy as np

from ..dataset.api import NDDataset


def Lstsq(A, B, rcond=-1):
    """
    Extension of numpy.linag.lstsq to hddatasets
    Return the least-squares solution to the linear equation
    A X = B where A and B are datasets of appropriate dimension.

    Parameters
    -----------
    A :

    B :

    rcond :


    Returns
    -------
    X: dataset of Least-squares solution. When B is two-dimensional,
    the solutions are in the K columns of X.

    res: Sums of residuals; squared Euclidean 2-norm for each column in b - a*x
        If the rank of a is < N or M <= N, this is an empty array. 
        If b is 1-dimensional, this is a (1,) shape array. 
        Otherwise the shape is (K,).

    rank : int,   Rank of matrix a.

    s : (min(M, N),) ndarray, Singular values of a

    """

    X, res, rank, s = np.linalg.lstsq(A.data, B.data, rcond)

    X = NDDataset(X)
    X.name = A.name + ' \ ' + B.name
    X.axes[0] = A.axes[1]
    X.axes[1] = B.axes[1]
    X.history = 'computed by spectrochempy.lstsq \n'
    return X, res, rank, s
