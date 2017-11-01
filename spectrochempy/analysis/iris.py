# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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


__all__ = ['Iris']
_classes = __all__[:]

import numpy as np

from scipy import optimize


class Iris:
    """Infrared inversion spectroscopy"""

    def __init__(self, X, invertParam):

        # check options
        if 'kernel' in invertParam:
            kernel = invertParam['kernel']
        else:
            kernel = 'langmuir'

        eps = np.linspace(invertParam['epsRange'][0],
                          invertParam['epsRange'][1],
                          invertParam['epsRange'][2])

        w = np.zeros((invertParam['epsRange'][2], 1))
        w[0] = 0.5 * (g[-1] - g[0]) / (invertParam['epsRange'][2] - 1)
        w[-1] = w[0]

        for j in range(1, invertParam['epsRange'][2] - 1):
            w[j] = 2 * w[0]

        K = np.zeros((X.shape[0], invertParam['epsRange'][2]))

        if kernel == 'langmuir':
            for i in range(X.shape[0]):
                for j in range(invertParam['epsRange'][2]):
                    K[i, j] = w[j] * X.dims[0].axes[1][i] * np.exp(-eps[j]) / (
                    1 + X.dims[0].axes[1][i] * np.exp(-eps[j]))
        if kernel == 'custom':
            for i in range(X.shape[0]):
                for j in range(invertParam['epsRange'][2]):
                    pass
                    # todo....
                    # K[i,j] = w[j] * fun(X.dims[0].axes[1][i], eps[j]) 

        W = np.eye(X.shape[0])

        m = invertParam['epsRange'][2]

        S = np.zeros((m, m))
        S[0, 0] = 6
        S[0, 1] = -4
        S[0, 2] = 1
        S[1, 0] = -4
        S[1, 1] = 6
        S[1, 2] = -4
        S[1, 3] = 1

        for i in range(2, m - 2):
            S[i, i - 2] = 1
            S[i, i - 1] = -4
            S[i, i] = 6
            S[i, i + 1] = -4
            S[i, i + 2] = 1

        S[m - 2, m - 4] = 1
        S[m - 2, m - 3] = -4
        S[m - 2, m - 2] = 6
        S[m - 2, m - 1] = -4
        S[m - 1, m - 3] = 1
        S[m - 1, m - 2] = -4
        S[m - 1, m - 1] = 6

        S = ((g[m - 1] - g[0]) / (m - 1)) ** (-3) * S

        lambdaReg = np.logspace(invertParam['lambdaRange'][0],
                                invertParam['lambdaRange'][1],
                                invertParam['lambdaRange'][2])
        n_lambda = lambdaReg.shape[0]
        f = np.zeros((m, X.shape[1], n_lambda))
        RSS = np.zeros((X.shape[1], n_lambda))
        SM = np.zeros((X.shape[1], n_lambda))
        RSST = np.zeros((n_lambda, 1))
        SMT = np.zeros((n_lambda, 1))

        print(('Solving for ' + str(
            n_lambda) + ' regularization parameters (lambda) \n'))

        for i, lambdaR in enumerate(lambdaReg):
            print(('... Solving for lambda = ' + str(lambdaR) + '...'))
            Q = 2 * (np.dot(K.T, np.dot(W, K)) + lambdaR * S)
            c = -2 * np.dot(X.data.T, np.dot(W, K))

            x_prec = np.random.randn(m, 1)
            for j, freq in enumerate(X.dims[1].axes[0]):
                x0 = x_prec

                # x0 = np.zeros((m,1))
                def objective(x, sign=1.):
                    return sign * (
                    0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(c[j, :], x))

                def jac(x, sign=1.):
                    return sign * (np.dot(x.T, Q) + c[j, :])

                cons = {'type': 'ineq',
                        'fun': lambda x: x,
                        'jac': lambda x: np.eye(m)}

                opt = {'disp': False}

                res_cons = optimize.minimize(objective, x0, jac=jac,
                                             constraints=cons,
                                             method='SLSQP', options=opt)

                f[:, j, i] = res_cons['x']
                RSS[j, i] = np.linalg.norm(X.data[:, j] - np.dot(K, f[:, j, i]))
                SM[j, i] = np.linalg.norm(
                    np.dot(np.dot(np.transpose(f[:, j, i]), S), f[:, j, i]))
                x_prec = res_cons['x']

        RSST[i] = sum(RSS[:, i])
        SMT[i] = sum(SM[:, i])

        print('\n Done')
        # TODO: check this. __init__ should not return a value
        #         return(f)
