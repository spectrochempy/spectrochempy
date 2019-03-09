# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implement the IRIS class.

"""
__all__ = ['IRIS']

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.processors.npy import dot

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


class IRIS:
    """
    2D Infrared inversion spectroscopy

    """

    def __init__(self, X, param, **kwargs):
        """
        Parameters
        -----------
        X : |NDDataset|
            The dataset on which to perform the 2D-IRIS analysis
        param : dict
            Dict of inversion parameters with the following keys:

            *   'kernel': the name of  the kernel used to make the inversion. The kernel K(p, eps) is a functional
                relationship holding between 'p', the experimental variable that was changed in the rows direction of X
                (e.g. temperature, pressure, time, ...) and the concentration of pure species characterized
                by the physico-chemical parameter 'eps' (e.g. adsorption/desorption energy, ...).
                Default: 'langmuir'. If another kernel name is given, the kernel function must be passed in param['ker']
            *   'ker': a two-variable lambda function ker(p, eps) where p and eps are the external experimental
                variable and  the internal physico-chemical parameter, respectively - TODO: implement other kernels: e.g. 'CA', 'TPD'
            *   'epsRange': array_like of three values [start, stop, num] defining the interval of eps values.
                start, stop: the starting and end values of eps, num: number of values.
            *   'lambdaRange': array_like of three values [start, stop, num] defining the interval of regularization
                parameter. Its values are speced evenly on a log scale with 10^start and 10^stop: a the starting and
                end values and num the number of values.
            *   'p': array or coordinate of the external variable. If none is given, p = X.y.values
            *   'guess': method to guess the initial distribution function for the current wavelength.
            *   'previous': takes the distribution at the previous wavelength, 'zero' takes a null distribution
                function, 'random'  takes a random distribution function.
        verbose : bool
            If set to True, prints informations during the 2D IRIS  analysis.
            In any case, the same information is returned in self._log

        """

        # check options
        # defined the kernel
        if 'kernel' in param:
            kername = param['kernel']
            if param['kernel'].lower() not in ['langmuir', 'ca']:
                try:
                    ker = param['ker']
                except KeyError:
                    print('A kernel function must be given')

            elif param['kernel'].lower() == 'langmuir':
                ker = lambda p_, eps_: np.exp(-eps_) * p_ / (1 + np.exp(-eps_) * p_)

            elif param['kernel'].lower() == 'ca':
                ker = lambda p_, eps_: 0 if p_ < np.exp(eps_) else 1


        else:
            # set the kernel to default ('langmuir')
            kername = 'langmuir'
            ker = lambda p_, eps_: np.exp(-eps_) * p_ / (1 + np.exp(-eps_) * p_)

        eps = np.linspace(param['epsRange'][0],
                          param['epsRange'][1],
                          param['epsRange'][2])

        lambdaReg = np.logspace(param['lambdaRange'][0],
                                param['lambdaRange'][1],
                                param['lambdaRange'][2])

        if 'p' in param:
            p = param['p']
            # check p
            if isinstance(p, Coord):
                if p.shape[1] != X.shape[0]:
                    raise ValueError('\'p\' should be consistent with the y coordinate of the dataset')
                pval = p.data  # values
                # (values contains unit! to use it we must either have eps with units or noramlise p
            else:
                if len(p) != X.shape[0]:
                    raise ValueError('\'p\' should be consistent with the y coordinate of the dataset')
                p = Coord(p, title='External variable')
                pval = p.data  # values
        else:
            p = X.y
            pval = X.y.data  # values

        if 'guess' in param:
            guess = param['guess']
        else:
            guess = 'previous'

        verbose = kwargs.get('verbose', False)

        w = np.zeros((len(eps), 1))
        w[0] = 0.5 * (eps[-1] - eps[0]) / (len(eps) - 1)  #
        w[-1] = w[0]

        for j in range(1, len(eps) - 1):
            w[j] = 2 * w[0]

        K = NDDataset(np.zeros((p.size, len(eps))))
        K.set_coords(y=p, x=Coord(eps, title='epsilon'))
        for i, p_i in enumerate(pval):
            for j, eps_j in enumerate(eps):
                K.data[i, j] = w[j] * ker(p_i, eps_j)

        W = NDDataset(np.eye(X.shape[0]))

        m = len(eps)

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

        S = ((eps[m - 1] - eps[0]) / (m - 1)) ** (-3) * S

        n_lambda = lambdaReg.shape[0]
        f = np.zeros((n_lambda, m, len(X.x.data)))

        RSS = np.zeros((n_lambda, 1))
        SM = np.zeros((n_lambda, 1))

        if verbose:
            print('Solving for {} wavenumbers and {} regularization parameters \n'.format(X.shape[1], n_lambda))

        for i, lambdaR in enumerate(lambdaReg):
            if verbose:
                print('... Solving for lambda = {} ...'.format(lambdaR))

            Q = 2 * (np.dot(K.data.T, np.dot(W.data, K.data)) + lambdaR * S)
            c = -2 * np.dot(X.data.T, np.dot(W.data, K.data))

            x_prec = np.random.randn(m, 1)

            for j, freq in enumerate(X.x.data):

                if guess == 'previous':
                    x0 = x_prec
                elif guess == 'zero':
                    x0 = np.zeros((m, 1))
                elif guess == 'random':
                    x0 = np.random.randn(m, 1)

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

                x_prec = f[i, :, j] = res_cons['x']

            res = X.data - np.dot(K.data, f[i].data)
            RSS[i] = np.sum(res ** 2)
            SM[i] = np.linalg.norm(
                np.dot(np.dot(np.transpose(f[i]), S), f[i]))

        if verbose:
            print('\n Done')

        f = NDDataset(f)
        f.name = '2D distribution functions'
        f.title = 'pseudo-concentration'
        f.history = '2D IRIS analysis of {} dataset with the {} kernel'.format(X.name, kername)
        xcoord = X.coords['x']
        ycoord = Coord(data=eps, title='epsilon')
        zcoord = Coord(data=lambdaReg, title='lambda')
        f.set_coords(z=zcoord, y=ycoord, x=xcoord)
        self._f = f
        self._K = K
        self._X = X
        self._RSS = RSS
        self._SM = SM

    def transform(self):
        """
        Apply the inversion of the X dataset (m x n) and returns
        the 2D distribution functions `f[i]` obtained for a given
        regularization parameter :math:`\lambda_i` using the following
        factorization: :math:`X = K.f[i]`.
        :math:`K` is a (m x q) matrix holding the values of the kernel
        function for the m values of the external variable (`p`) and the
        q values of the internal variable (`epsilon`).
        :math: `f[i]` is the (q x n) matrix holding the values of the
        2D-distribution function

        Return
        -------
        f : |NDDataset|
            object (l x m x n) containing the l 2D-distribution
            functions f[i] obtained for each value of the regularization
            parameter.

        """

        return self._f

    def inverse_transform(self):
        """
        Transform data back to the original space

        The following matrix operation is performed: :math:`\hat{X} = K.f[i]`
        for each value of the regularization parameter.

        Return
        ------
        X_hat : |NDDataset|
            The reconstructed dataset.

        """
        X_hat = NDDataset(np.zeros((self._f.z.size, self._X.y.size, self._X.x.size)),
                          title=self._X.title, units=self._X.units)

        X_hat.name = '2D-IRIS Reconstructed datasets'
        X_hat.set_coords(z=self._f.z, y=self._X.y, x=self._X.x)
        for i in range(X_hat.z.size):
            X_hat[i] = np.dot(self._K.data, self._f[i].data.squeeze())
        return X_hat

    def plotlcurve(self, **kwargs):
        """
        Plots the L Curve

        Parameters
        ----------
        scale : str, optional, default: 'll'
            2 letters among 'l' (log) or 'n' (non-log) indicating whether the y and x
            axes should be log scales.

        Returns
        -------
        ax : subplot axis

        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('L curve')
        scale = kwargs.get('scale', 'll').lower()
        ax.scatter(self._RSS, self._SM)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Smoothness')
        if scale[1] == 'l':
            ax.set_xscale('log')
        if scale[0] == 'l':
            ax.set_yscale('log')

        return ax
