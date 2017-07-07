# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

#TODO: create tests

__all__ = ['lstsq']

import numpy as np

from ..core import NDDataset

def lstsq(A, B, rcond = -1):
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
    return (X, res, rank, s)

