# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implement the NNMF (Non Negative Matrix Factorization) class.
"""

__all__ = ["NNMF"]

__dataset_methods__ = []

import numpy as np
from numpy.linalg import norm
from time import time
from sys import stdout

from traitlets import HasTraits, Float, Int, Instance

from spectrochempy.core import info_, debug_
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import get_loglevel, set_loglevel, INFO


class NNMF(HasTraits):
    """
    Performs a Non Negative Matrix Factorization of a |NDDataset|.

    Parameters
    ----------
    dataset: |NDDataset|
        The data to be analysed.
    Ci: |NDDataset|
        Initial concentration profile.
    Sti: |NDDataset|
        Initial Spectral profile.
    **kwargs
        Optional keyword parameters. See Other Parameters below.

    Other Parameters
    ----------------
    tol: float, optional
        Tolerance for a relative stopping condition.
    maxtime: float, optional
        Time limit.
    maxiter: float
        Limit number of iterations.
    verbose:
        Print calculation details

    Returns
    -------
    C, St: |NDDataset|
        Optimized concentration and spectral profile.

    Notes
    -----
    Algorithm based on

    C.-J. Lin. Projected gradient methods for non-negative matrix factorization.
    Neural Computation, 19(2007), 2756-2779.

    If you find this tool useful, please cite the above work.
    Author : Chih-Jen Lin, National Taiwan University
    Copyright (c) 2005-2008 Chih-Jen Lin
    See LICENCES in the root directory.

    The current code is based on the Python translation by Anthony Di Franco:
    https://www.csie.ntu.edu.tw/~cjlin/nmf/others/nmf.py
    """

    tol = Float()
    maxtime = Float()
    maxit = Int()
    C = Instance(NDDataset)
    St = Instance(NDDataset)

    def __init__(self, dataset, Ci, Sti, **kwargs):
        super().__init__()

        tol = kwargs.get("tol", 0.1)
        maxtime = kwargs.get("maxtime", 60)
        maxit = kwargs.get("maxiter", 100)

        if kwargs.get("verbose", False) and get_loglevel() > INFO:
            set_loglevel(INFO)

        self.C = Ci.copy()
        self.St = Sti.copy()

        self.C.data, self.St.data = self.nmf(
            dataset.data, self.C.data, self.St.data, tol, maxtime, maxit
        )

        self.C.name = "Optimmized concentration profile"
        self.C.history = "nnmf optimiaation"
        self.St.name = "Optimized spectral profile"
        self.St.history = "nnmf optimiaation"

    def nmf(self, V, Winit, Hinit, tol, maxtime, maxiter):
        """
        NMF by alternative non-negative least squares using projected gradients.

        Parameters
        ==========
        V: |ndarray|
            numpy array to be analysed
        Winit,Hinit: |ndarray|
            Initial solutions for concentration and spectral profile..
        tol: float
            Tolerance for a relative stopping condition.
        maxtime: float
            Limit of time.
        maxiter: int
            Limit number for iterations.

        Returns
        =======
        W,H: |ndarray|
            Output solution.
        """

        W = Winit
        H = Hinit

        initt = time()

        gradW = np.dot(W, np.dot(H, H.T)) - np.dot(V, H.T)
        gradH = np.dot(np.dot(W.T, W), H) - np.dot(W.T, V)
        initgrad = norm(np.r_[gradW, gradH.T])
        info_(f"Init gradient norm {initgrad:.3f}")
        tolW = max(0.001, tol) * initgrad
        tolH = tolW

        for myiter in range(1, maxiter):
            # stopping condition
            projnorm = norm(
                np.r_[
                    gradW[np.logical_or(gradW < 0, W > 0)],
                    gradH[np.logical_or(gradH < 0, H > 0)],
                ]
            )

            if projnorm < tol * initgrad or time() - initt > maxtime:
                break

            (W, gradW, iterW) = self.nlssubprob(V.T, H.T, W.T, tolW, 1000)
            W = W.T
            gradW = gradW.T

            if iterW == 1:
                tolW = 0.1 * tolW

            (H, gradH, iterH) = self.nlssubprob(V, W, H, tolH, 1000)

            if iterH == 1:
                tolH = 0.1 * tolH

            if myiter % 10 == 0:
                stdout.write(".")

        info_(f"\nIter = {myiter} Final proj-grad norm {projnorm:.3f}")
        return W, H

    @staticmethod
    def nlssubprob(V, W, Hinit, tol, maxiter):
        """
        Parameters
        ----------
        V, W
            Constant matrices.
        Hinit
            initial solution.
        tol: stopping tolerance.
        maxiter: limit of iterations.

        Returns
        -------
        H, grad
            Output solution and gradient.

        """

        H = Hinit
        WtV = np.dot(W.T, V)
        WtW = np.dot(W.T, W)

        alpha = 1
        beta = 0.1

        for n_iter in range(1, maxiter + 1):
            grad = np.dot(WtW, H) - WtV
            if norm(grad * np.logical_or(grad < 0, H > 0)) < tol:
                break

        Hp = H

        # search step size
        for inner_iter in range(20):
            # gradient step
            Hn = H - alpha * grad
            # gradient step
            Hn *= Hn > 0
            d = Hn - H
            gradd = np.dot(grad.ravel(), d.ravel())
            dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
            if inner_iter == 0:
                decr_alpha = not suff_decr
                Hp = H
            if decr_alpha:
                if suff_decr:
                    H = Hn
                    break
                alpha = alpha * beta
            else:
                if not suff_decr or (Hp == Hn).all():
                    H = Hp
                    break
                alpha = alpha / beta
                Hp = Hn

        if n_iter == maxiter:
            debug_("Max iter in nlssubprob")

        return H, grad, n_iter


# ============================================================================
if __name__ == "__main__":
    pass
