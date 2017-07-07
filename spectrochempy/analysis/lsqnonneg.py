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

#todo: create tests

__all__ = ['lsqnonneg']

from ..core import NDDataset

import numpy as np

def lsqnonneg(C, d, x0=None, tol=None, itmax_factor=3):

    '''Linear least squares with nonnegativity constraints
    (x, resnorm, residual) = lsqnonneg(C,d)
    returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real

    # python implementation of NNLS algorithm
    # References: Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, 
    #             Prentice-Hall, Chapter 23, p. 161, 1974.
    # Contributed by Klaus Schuch (schuch@igi.tugraz.at)
    # based on MATLAB's lsqnonneg function
    #
    # AT: if C and d are datasets, x will be a dataset with relevant dims
    #     else return np array

    '''

    returnDataset = False

    if isinstance(C, NDDataset) and isinstance(d, NDDataset):

        returnDataset = True

        X = NDDataset(np.zeros((C.shape[1], d.shape[1])))
        X.axes[0] = C.axes[1].copy()
        X.axes[1] = d.axes[1].copy()
    
        C = C.data
        d = d.data
        if isinstance(x0, NDDataset):
            x0 = x0.data
            
    eps = 2.22e-16    # from matlab

    def norm1(x):
        return abs(x).sum().max()

    def msize(x, dim):
        s = x.shape
        if dim >= len(s):
            return 1
        else:
            return s[dim]
        
    if tol is None:
        tol = 10*eps*norm1(C)*(max(C.shape)+1)

    C = np.asarray(C)
    (m,n) = C.shape
    P = np.zeros(n)
    Z = np.arange(1, n+1)

    if x0 is None:
        x=P
    else:
        if any(x0 < 0):
            x=P
        else:
            x=x0

    ZZ=Z
    
    resid = d - np.dot(C, x)

    w = np.dot(C.T, resid)
    outeriter=0
    it=0
    itmax=itmax_factor*n
    exitflag=1

    # outer loop to put variables into set to hold positive coefficients
    while np.any(Z) and np.any(w[ZZ-1] > tol):

        outeriter += 1

        t = w[ZZ-1].argmax()
        t = ZZ[t]

        P[t-1]=t
        Z[t-1]=0

        PP = np.where(P != 0)[0]+1
        ZZ = np.where(Z != 0)[0]+1

        CP = np.zeros(C.shape)
        CP[:, PP-1] = C[:, PP-1]
        CP[:, ZZ-1] = np.zeros((m, msize(ZZ, 1)))

        z=np.dot(np.linalg.pinv(CP), d)
        z[ZZ-1] = np.zeros((msize(ZZ,1), msize(ZZ,0)))
        
        # inner loop to remove elements from the positve set which no longer belong

        while np.any(z[PP-1] <= tol):

            it += 1

            if it > itmax:
                max_error = z[PP-1].max()
                raise Exception('Exiting: Iteration count (=%d) exceeded\n Try raising the tolerance tol. (max_error=%d)' % (it, max_error))

            QQ = np.where((z <= tol) & (P != 0))[0]
            alpha = min(x[QQ]/(x[QQ] - z[QQ]))
            x = x + alpha*(z-x)

            ij = np.where((abs(x) < tol) & (P != 0))[0]+1
            Z[ij-1] = ij
            P[ij-1] = np.zeros(max(ij.shape))
            PP = np.where(P != 0)[0]+1
            ZZ = np.where(Z != 0)[0]+1

            CP[:, PP-1] = C[:, PP-1]
            CP[:, ZZ-1] = np.zeros((m, msize(ZZ, 1)))

            z=np.dot(np.linalg.pinv(CP), d)
            z[ZZ-1] = np.zeros((msize(ZZ,1), msize(ZZ,0)))
        x = z
        resid = d - np.dot(C, x)
        w = np.dot(C.T, resid)

        if returnDataset:
            X.data = x
            x = X.copy()
            
    return (x, sum(resid * resid), resid)