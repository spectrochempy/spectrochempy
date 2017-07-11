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


"""
based on:
NMF by alternative non-negative least squares using projected gradients

Author: Chih-Jen Lin, National Taiwan University
Python/numpy translation: Anthony Di Franco

Licence ???
"""
#TODO: create  tests

__all__ = ['nnmf']

import numpy as np
from numpy.linalg import norm
from time import time
from sys import stdout

import logging
logger = logging.getLogger()


class nnmf:

    def __init__(self, X, Ci, Sti, **options):
        """
        self.C, self.St = nnmf(X, Ci, Sti,**options)
        C,St: output solution
        Ci,Sti: initial solution
        options['tol']: tolerance for a relative stopping condition
        options['maxtime'], options['maxit']: limit of time and iterations
        """
        tol = options.get('tol', 0.1)

        maxtime = options.get('maxtime', 60)

        maxit = options.get('maxit', 100)

        self.C = Ci.copy()
        self.C.name = 'Conc profile optimized by nnmf'
        self.C.history = ''

        self.St = Sti.copy()
        self.St.name = 'Spectral profile optimized by nnmf'
        self.St.history = ''

        self.C.data, self.St.data = self.nmf(X.data, Ci.data, Sti.data, tol, maxtime,
                                        maxit)


    @staticmethod
    def nmf(V, Winit, Hinit, tol, timelimit, maxiter):
        """
        (W,H) = nmf(V,Winit,Hinit,tol,timelimit,maxiter)
        W,H: output solution
        Winit,Hinit: initial solution
        tol: tolerance for a relative stopping condition
        timelimit, maxiter: limit of time and iterations
        """

        def nlssubprob(V, W, Hinit, tol, maxiter):
            """
            H, grad: output solution and gradient
            iter: #iterations used
            V, W: constant matrices
            Hinit: initial solution
            tol: stopping tolerance
            maxiter: limit of iterations
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
                log.info('Max iter in nlssubprob')

            return (H, grad, n_iter)

        W = Winit

        H = Hinit

        initt = time()

        gradW = np.dot(W, np.dot(H, H.T)) - np.dot(V, H.T)
        gradH = np.dot(np.dot(W.T, W), H) - np.dot(W.T, V)
        initgrad = norm(np.r_[gradW, gradH.T])
        log.info('Init gradient norm {:.3f}'.format(initgrad))
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

        log.info(
            '\nIter = {} Final proj-grad norm {:.3f}'.format(myiter, projnorm))
        return (W, H)




