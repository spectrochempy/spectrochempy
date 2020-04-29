# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module implement the NNMF (Non Negative Matrix Factorization) class

"""
# TODO: create  tests

__all__ = ['NNMF']

__dataset_methods__ = []

import numpy as np
from numpy.linalg import norm
from time import time
from sys import stdout

from traitlets import HasTraits


class NNMF(HasTraits):
    """
    Performs a Non Negative Matrix Factorization of a |NDDataset|.

    Algorithm based on :
    C.-J. Lin. Projected gradient methods for non-negative matrix factorization. Neural Computation, 19(2007), 2756-2779.
    If you find this tool useful, please cite the above work.
    Author : Chih-Jen Lin, National Taiwan University
    Copyright (c) 2005-2008 Chih-Jen Lin
    All rights reserved.
    Python/numpy translation : Anthony Di Franco ;

    """

    def __init__(self, X, Ci, Sti, **kwargs):
        """
        self.C, self.St = nnmf(X, Ci, Sti,**kwargs)
        C,St : output solution
        Ci,Sti : initial solution
        kwargs['tol'] : tolerance for a relative stopping condition
        kwargs['maxtime'], kwargs['maxit'] : limit of time and iterations
        """
        tol = kwargs.get('tol', 0.1)

        maxtime = kwargs.get('maxtime', 60)

        maxit = kwargs.get('maxit', 100)

        self.C = Ci.copy()
        self.C.name = 'Conc profile optimized by nnmf'
        self.C.history = ''

        self.St = Sti.copy()
        self.St.name = 'Spectral profile optimized by nnmf'
        self.St.history = ''

        self.C.data, self.St.data = self.nmf(X.data, Ci.data, Sti.data, tol,
                                             maxtime,
                                             maxit)

    @staticmethod
    def nmf(V, Winit, Hinit, tol, timelimit, maxiter):
        """
        (W,H) = nmf(V,Winit,Hinit,tol,timelimit,maxiter)
        W,H : output solution
        Winit,Hinit : initial solution
        tol : tolerance for a relative stopping condition
        timelimit, maxiter : limit of time and iterations
        """

        def nlssubprob(V, W, Hinit, tol, maxiter):
            """
            H, grad : output solution and gradient
            iter : #iterations used
            V, W : constant matrices
            Hinit : initial solution
            tol : stopping tolerance
            maxiter : limit of iterations
            """

            H = Hinit
            WtV = np.dot(W.T, V)
            WtW = np.dot(W.T, W)

            alpha = 1
            beta = 0.1

            for n_iter in range(1, maxiter + 1):
                grad = np.dot(WtW, H) - WtV
                if norm(grad * np.logical_or(grad < 0, H > 0)) < tol: break

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
                    else:
                        alpha = alpha * beta
                else:
                    if not suff_decr or (Hp == Hn).all():
                        H = Hp
                        break
                    else:
                        alpha = alpha / beta
                        Hp = Hn

            if n_iter == maxiter:
                info_('Max iter in nlssubprob')

            return H, grad, n_iter

        W = Winit

        H = Hinit

        initt = time()

        gradW = np.dot(W, np.dot(H, H.T)) - np.dot(V, H.T)
        gradH = np.dot(np.dot(W.T, W), H) - np.dot(W.T, V)
        initgrad = norm(np.r_[gradW, gradH.T])
        info_('Init gradient norm {:.3f}'.format(initgrad))
        tolW = max(0.001, tol) * initgrad
        tolH = tolW

        for myiter in range(1, maxiter):
            # stopping condition
            projnorm = norm(np.r_[gradW[np.logical_or(gradW < 0, W > 0)],
                                  gradH[np.logical_or(gradH < 0, H > 0)]])

            if projnorm < tol * initgrad or time() - initt > timelimit:
                break

            (W, gradW, iterW) = nlssubprob(V.T, H.T, W.T, tolW, 10000)
            W = W.T
            gradW = gradW.T

            if iterW == 1:
                tolW = 0.1 * tolW

            (H, gradH, iterH) = nlssubprob(V, W, H, tolH, 10000)

            if iterH == 1:
                tolH = 0.1 * tolH

            if myiter % 10 == 0: stdout.write('.')

        info_(
            '\nIter = {} Final proj-grad norm {:.3f}'.format(myiter,
                                                             projnorm))
        return W, H
