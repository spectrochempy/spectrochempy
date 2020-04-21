
# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implements the IRIS class.

"""
__all__ = ['IRIS']
__dataset_methods__ = []

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.npy import dot

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import quadprog

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
            Dict of inversion parameters with the following keys :

            *   'custom_kernel': a two-variable lambda function ker(p, eps) where p and eps are the external experimental
                variable and  the internal physico-chemical parameter, respectively. If not given, one of the
                pre-defined kernel must be defined in param['kernel'], below.
            *   'kernel': the name of  the kernel used to make the inversion. The kernel K(p, eps) is a functional
                relationship holding between 'p', the experimental variable that was changed in the rows direction of X
                (e.g. temperature, pressure, time, ...) and the concentration of pure species characterized
                by the physico-chemical parameter 'eps' (e.g. adsorption/desorption energy, ...).
            *   'epsRange': array-like of three values [start, stop, num] defining the interval of eps values.
                start, stop: the starting and end values of eps, num: number of values.
            *   'lambdaRange': None or array_like of two values [min, max] or three values [start, stop, num] defining
                the interval of the regularization parameter. It set to None, a non regularized solution is computed.
                If two values are entered, the optimum regularization parameter is searched between 10^min and 10^max
                using the Cultrera_Callegaro algorithm (arXiv:1608.04571v2). If three valyes are given num values
                are spaced evenly on a log scale between 10^start and 10^stop.
            *   'p': array or coordinate of the external variable. If none is given, p = X.y.values

        verbose : bool
            If set to True, prints informations during the 2D IRIS  analysis.
            In a
            ny case, the same information is returned in self._log
        """
        # check options
        # defines the kernel

        if 'custom_kernel' in param:
            ker = param['custom_kernel']

        elif 'kernel' in param:
            if param['kernel'].lower() == 'langmuir':
                ker = lambda p_, eps_: np.exp(-eps_) * p_ / (1 + np.exp(-eps_) * p_)

            elif param['kernel'].lower() == 'ca':
                ker = lambda p_, eps_: 0 if p_ < np.exp(eps_) else 1

            elif param['kernel'].lower() == 'reactant-first-order':
                ker = lambda t, lnk: np.exp(-1 * np.exp(lnk) * t)

            elif param['kernel'].lower() == 'product-first-order':
                ker = lambda t, lnk: 1 - np.exp(-1 * np.exp(lnk) * t)
            else:
                raise NameError('This kernel: <{}> is not implemented'.format(ker))

        else:
            raise NameError('A kernel must be given !')

        # define eps values
        eps = np.linspace(param['epsRange'][0],
                          param['epsRange'][1],
                          param['epsRange'][2])

        # defines regularization parameter values
        if 'lambdaRange' not in param:
            lambdaRange = None
        else:
            lambdaRange=param['lambdaRange']
        if lambdaRange == None:
            regularization = False
            searchLambda = False
            lamb = [0]
        elif len(lambdaRange) == 2:
            regularization =True
            searchLambda = True
        elif len(param['lambdaRange']) == 3:
            regularization = True
            searchLambda = False
            lamb = np.logspace(param['lambdaRange'][0],
                               param['lambdaRange'][1],
                               param['lambdaRange'][2])
        else:
            raise ValueError('lambdaRange should either None or a set of 2 or 3 integers')

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

        # define containers for outputs
        if not regularization:
            f = np.zeros((1, len(eps), len(X.x.data)))
            RSS = np.zeros((1))
            SM = np.zeros((1))

        if regularization and not searchLambda:
            f = np.zeros((len(lamb), len(eps), len(X.x.data)))
            RSS = np.zeros((len(lamb)))
            SM = np.zeros((len(lamb)))

        if regularization and searchLambda:
            f = np.zeros((4, len(eps), len(X.x.data)))
            RSS = np.zeros((4))
            SM = np.zeros((4))

        # Define K matrix (kernel)
        # first some weighting coefficients for the numerical quadrature of the Fredholm integral
        w = np.zeros((len(eps), 1))
        w[0] = 0.5 * (eps[-1] - eps[0]) / (len(eps) - 1)  #
        for j in range(1, len(eps) - 1):
            w[j] = 2 * w[0]
        w[-1] = w[0]
        # then compute K  (TODO: allow using weighting matrix W)
        K = NDDataset(np.zeros((p.size, len(eps))))
        K.set_coords(y=p, x=Coord(eps, title='epsilon'))
        for i, p_i in enumerate(pval):
            for j, eps_j in enumerate(eps):
                K.data[i, j] = w[j] * ker(p_i, eps_j)

        # Define S matrix (sharpness), see function Smat() below
        S = Smat(eps)

        # solve untregularized problem
        if not regularization:
            if verbose:
                print('Solving for {} wavenumbers and {} spectra, no regularization\n'
                      .format(X.shape[1], X.shape[0]))
            # une scipy.nnls() to solve the linear problem: X = K f
            for j, freq in enumerate(X.x.data):
                f[0, :, j] = optimize.nnls(K.data, X[:,j].data.squeeze())[0]
            res = X.data - np.dot(K.data, f[0].data)
            RSS[0] = np.sum(res ** 2)
            SM[0] = np.linalg.norm(np.dot(np.dot(np.transpose(f[0]), S), f[0]))
            if verbose:
                print('-->  residuals = {:.2e}    curvature = {:.2e}'.format(RSS[0], SM[0]))

        if regularization:
            # some matrices used for QP optimization do not depend on lambdaR
            # and are computed here. The standard form used by quadprog() is
            # minimize (1/2) xT G x - aT x ; subject to: C.T x >= b

            # The first part of the G matrix is independent of lambda:  G = G0 + 2 * lambdaR S
            G0 = 2 * np.dot(K.data.T, K.data)
            a = 2 * np.dot(X.data.T, K.data)
            C = np.eye(len(eps))
            b = np.zeros(len(eps))

            def solve_lambda(X, K, G0, lamda, S, verbose):
                ''' QP optimization
                parameters:
                ----------
                X: NDDataset of experimental spectra
                K: NDDataset, kernel datase
                G0: the lamda independent part of G 
                lamda: regularization parameter
                S: penalty function (shaprness)
                verbose: print info
    
                returns: f, RSS and SM for a given regularization parameter
                ------- '''
                fi = np.zeros((len(eps), len(X.x.data)))
                if verbose :
                    print('... Solving for lambda = {} ...'.format(lamda))
                G = nearestPD(G0 + 2 * lamda * S)
                for j, freq in enumerate(X.x.data):
                    fi[:, j] = quadprog.solve_qp(G, a[j].squeeze(), C, b)[0]

                resi = X.data - np.dot(K.data, fi)
                RSSi = np.sum(resi ** 2)
                SMi = np.linalg.norm(np.dot(np.dot(np.transpose(fi), S), fi))
                if verbose:
                    print('-->  residuals = {:.2e}    curvature = {:.2e}'.format(RSSi, SMi))
                return fi, RSSi, SMi

            def menger(x, y):
                '''returns the Menger curvature of a triplet of
                points. x, y = sets of 3 cartesian coordinates '''

                numerator = 2 * (x[0]*y[1] + x[1]*y[2] + x[2]*y[0]
                                 - x[0]*y[2] - x[1]*y[0] - x[2]*y[1])
                # euclidian distances
                r01 = (x[1] - x[0])**2 + (y[1] - y[0])**2
                r12 = (x[2] - x[1])**2 + (y[2] - y[1])**2
                r02 = (x[2] - x[0])**2 + (y[2] - y[0])**2

                denominator = np.sqrt(r01 * r12 * r02)
                return numerator/denominator

            if not searchLambda :
                if verbose:
                    print('Solving for {} wavenumbers, {} spectra and {} regularization parameters \n'
                        .format(X.shape[1], X.shape[0], len(lamb)))

                    for i, lamda in enumerate(lamb):
                        f[i], RSS [i], SM[i] = solve_lambda(X, K, G0, lamda, S, verbose)

            else:
                if verbose:
                    print('Solving for {} wavenumbers and {} spectra, search regularization parameter '
                      'in [10**{}, 10**{}]\n'
                          .format(X.shape[1], X.shape[0], str(min(lambdaRange)), str(max(lambdaRange))))

                    x = lamda = np.ndarray((4))
                    epsilon = 0.1
                    phi = (1 + np.sqrt(5)) / 2

                    x[0] = min(lambdaRange)
                    x[3] = max(lambdaRange)
                    x[1] = (x[3] + phi * x[0])/ (1 + phi)
                    x[2] = x[0] + x[3] - x[1]
                    lamb = 10**x
                    if verbose:
                        print('Log lambda= ' + str(x))
                    for i, xi in enumerate(x):
                        f[i], RSS[i], SM[i] = solve_lambda(X, K, G0, 10**xi, S, verbose)

                    Rx = RSS
                    Sy = SM
                    while "convergence not reached":
                        C1 = menger(Rx[0:3], Sy[0:3])
                        C2 = menger(Rx[1:4], Sy[1:4])
                        if verbose:
                            print('Curvatures: C1 = {} ; C2 = {}'.format(C1, C2))
                        while C2 < 0:
                            x[3] = x[2]; Rx[3] = Rx[2]; Sy[3] = Sy[2]
                            x[2] = x[1]; Rx[2] = Rx[1]; Sy[2] = Sy[1]
                            x[1] = (x[3] + phi * x[0]) / (1 + phi)
                            if verbose:
                                print('Log lambda= ' + str(x))
                            f_, Rx[1], S[1] = solve_lambda(X, K, G0, 10**x[1], S, verbose)
                            lamb = np.append(lamb, np.array(10**x[1]))
                            f = np.concatenate((f, np.atleast_3d(f_.T).T))
                            RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                            SM = np.concatenate((SM, np.array(Sy[1:2])))
                            C2 = menger(Rx[1:4], Sy[1:4])
                            print('new curvature: C2 = {}'.format(C2))
                        if C1 > C2:
                            xx = x[1]
                            x[3] = x[2]; Rx[3] = Rx[2]; Sy[3] = Sy[2]
                            x[2] = x[1]; Rx[2] = Rx[1]; Sy[2] = Sy[1]
                            x[1] = (x[3] + phi * x[0]) / (1 + phi)
                            if verbose:
                                print('Log lambda= ' + str(x))
                            f_, Rx[1], S[1] = solve_lambda(X, K, G0, 10**x[1], S, verbose)
                            f = np.concatenate((f, np.atleast_3d(f_.T).T))
                            lamb = np.append(lamb, np.array(10**x[1]))
                            RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                            SM = np.concatenate((SM, np.array(Sy[1:2])))
                        else:
                            xx = x[2]

                            x[0] = x[1]; Rx[0] = Rx[1]; Sy[0] = Sy[1]
                            x[1] = x[2]; Rx[1] = Rx[2]; Sy[1] = Sy[2]
                            x[2] = x[0] - (x[1] - x[3])
                            if verbose:
                                print('Log lambda= ' + str(x))
                            f_, Rx[2], S[2] = solve_lambda(X, K, G0, 10**x[2], S, verbose)
                            f = np.concatenate((f, np.atleast_3d(f_.T).T))
                            lamb = np.append(lamb, np.array(10**x[2]))
                            RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                            SM = np.concatenate((SM, np.array(Sy[1:2])))
                        if (10**x[3] - 10**x[0])/10**x[3] < epsilon:
                            break
                    if verbose:
                        print('\n optimum found !')
        if verbose:
            print('\n Done.')
        f = NDDataset(f)
        f.name = '2D distribution functions'
        f.title = 'pseudo-concentration'
        f.history = '2D IRIS analysis of {} dataset'.format(X.name)
        xcoord = X.coords['x']
        ycoord = Coord(data=eps, title='epsilon')
        zcoord = Coord(data=lamb, title='lambda')
        f.set_coords(z=zcoord, y=ycoord, x=xcoord)
        self.f = f
        self.K = K
        self.X = X
        self.lamda = lamb
        self.RSS = RSS
        self.SM = SM


    def reconstruct(self):
        """
        Transform data back to the original space

        The following matrix operation is performed : :math:`\hat{X} = K.f[i]`
        for each value of the regularization parameter.

        Returns
        -------
        X_hat : |NDDataset|
            The reconstructed dataset.

        """
        X_hat = NDDataset(np.zeros((self.f.z.size, self.X.y.size, self.X.x.size)),
                          title=self.X.title, units=self.X.units)

        X_hat.name = '2D-IRIS Reconstructed datasets'
        X_hat.set_coords(z=self.f.z, y=self.X.y, x=self.X.x)
        for i in range(X_hat.z.size):
            X_hat[i] = np.dot(self.K.data, self.f[i].data.squeeze())
        return X_hat

    def plotlcurve(self, **kwargs):
        """
        Plots the L Curve

        Parameters
        ----------
        scale : str, optional, default='ll'
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
        plt.plot(self.RSS, self.SM, 'o')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Curvature')
        if scale[1] == 'l':
            ax.set_xscale('log')
        if scale[0] == 'l':
            ax.set_yscale('log')
        return ax

    def plotmerit(self, index=None, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        index : optional, int, list or tuple of int. Index(es) of the inversions (i.e. of the lambda values) to consider.
         If 'None': plots for all indices. default: None

        Returns
        -------
        list of axes
        """

        colX, colXhat, colRes = kwargs.get('colors', ['blue', 'green', 'red'])

        X_hats = self.reconstruct()
        axeslist = []
        if index is None:
            index = range(len(self.lamda))
        if type(index) is int:
            index = [index]
        for i in index:
            res = self.X - X_hats[i].squeeze()
            ax = self.X.plot()
            ax.plot(self.X.x.data, X_hats[i].squeeze().T.data, color=colXhat)
            ax.plot(self.X.x.data, res.T.data, color=colRes)
            ax.set_title('2D IRIS merit plot, $\lambda$ = ' + str(self.lamda[i]))
            axeslist.append(ax)
        return axeslist

    def plotdistribution(self, index=None, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        index : optional, int, list or tuple of int. Index(es) of the inversions (i.e. of the lambda values) to consider.
         If 'None': plots for all indices. default: None
        other optional arguments are passed in the plots
        Returns
        -------
        list of axes
        """

        axeslist = []
        if index is None:
            index = range(len(self.lamda))
        if type(index) is int:
            index = [index]
        for i in index:
            self.f[i].plot(method='map', **kwargs)
        return axeslist

# --------------------------------------------
# Utility functions

def Smat(eps):
    """ returns the matrix used to compute the norm of f second derivative  """
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
    return S


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    copyright: see https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
        print('makes PD matrix')
    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky

    copyright: see https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False



