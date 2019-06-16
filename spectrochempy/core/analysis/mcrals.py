# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implements the MCRALS class.

"""

__all__ = ['MCRALS']

__dataset_methods__ = ['MCRALS']

import numpy as np
from traitlets import HasTraits, Instance

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.analysis.pca import PCA
from spectrochempy.core.processors.npy import dot
from spectrochempy.core import info_, set_loglevel, INFO

class MCRALS(HasTraits):
    """
    Performs MCR-ALS of a dataset knowing the initial C or St matrix

    """

    _C = Instance(NDDataset)
    """|NDDataset| - Concentration profile of pure species"""
    _St = Instance(NDDataset)
    """|NDDataset| - Spectra profile of pure species"""
    _param = Instance(dict)
    """dict - Parameters of the MCS-ALS optimization"""
    _log = Instance(str)
    """str - Log of the MCS-ALS iterations"""

    def __init__(self, X, guess, **kwargs):
        """
        Parameters
        ----------
        X: |NDDataset|
            The dataset on which to perform the MCR-ALS analysis
        guess: |NDDataset|
            Initial concentration or spectra
        param: dict
            Dict of optimization parameters with the following keys:

            *   'tol': float, optional, convergence criterion on the change of resisuals.
                (percent change of standard deviation of residuals). default=0.1
            *   'maxit': maximum number of ALS minimizations. default = 50
            *   'maxdiv': maximum number of successive non-converging iterations. default=5
            *   'nonnegConc': array or tuple indicating species non-negative concentration
                profiles. For instance [1, 0, 1] indicates that species #0
                and #2 have non-negative conc profiles while species #1
                can have negative concentrations.
                Default [1, ..., 1]  (only non-negative concentrations)
            *   'unimodConc': array or tuple indicating species having unimodal concentrationsprofiles.
                Default [1, ..., 1]  (only unimodal concentration profiles)
            *   'nonnegSpec': array or tuple indicating species having non-negative spectra
                Default [1, ..., 1]  (only non-negative spectra)
            *   'unimodSpec': array or tuple indicating species having unimodal spectra
                Default [0, ..., 0]  (no unimodal cocentration profiles)
        verbose: bool
            If set to True, prints a summary of residuals and residuals change at each iteration. default = False.
            In anyu case, the same information is returned in self._log

        """
        # TODO: add example

        verbose = kwargs.get('verbose', False)
        if verbose:
            set_loglevel(INFO)

        # Check initial data
        # ------------------------------------------------------------------------

        initConc, initSpec = False, False

        if X.shape[0] == guess.shape[0]:
            initConc = True
            C = guess.copy()
            C.name = 'Pure conc. profile, mcs-als of ' + X.name
            nspecies = C.shape[1]

        elif X.shape[1] == guess.shape[1]:
            initSpec = True
            St = guess.copy()
            St.name = 'Pure spectra profile, mcs-als of ' + X.name
            nspecies = St.shape[0]

        else:
            raise ValueError('the dimensions of initial concentration '
                             'or spectra dataset do not match the data')

        ny, nx = X.shape

        # makes a PCA with same number of species
        Xpca = PCA(X).inverse_transform(n_pc=nspecies)

        # Get optional parameters in kwargs or set them to their default
        # ------------------------------------------------------------------------

        # TODO: make a preference  file to set this kwargs
        param = kwargs.get('param', dict())

        ### optimization

        tol = param.get('tol', 0.1)

        maxit = param.get('maxit', 50)

        maxdiv = param.get('maxdiv', 5)

        ### constraints on concentrations

        nonnegConc = param.get('nonnegConc', [1] * nspecies)

        unimodConc = param.get('unimodConc', [1] * nspecies)

        unimodTol = param.get('unimodTol', 1.1)

        unimodMod = param.get('unimodMod', 'strict')

        monoDecConc = param.get('monoDecConc', [0] * nspecies)

        monoDecTol = param.get('monoDecTol', 1.1)

        monoIncConc = param.get('monoIncConc', [0] * nspecies)

        monoIncTol = param.get('monoIncTol', 1.1)

        ### constraints on concentrations

        nonnegSpec = param.get('nonnegSpec', [1] * nspecies)

        normSpec = param.get('normSpec', 'no')

        #    if ('unimodSpec' in kwargs): unimodSpec = kwargs['unimodSpec']
        #    else: unimodSpec = np.zeros((1, nspecies))

        # Compute initial spectra or concentrations   (first iteration...)
        # ------------------------------------------------------------------------

        if initConc:
            if C.coords is None:
                C.set_coords(y=X.y, x=C.x)
            St = NDDataset(np.linalg.lstsq(C.data, X.data)[0])
            St.name = 'Pure spectra profile, mcs-als of ' + X.name
            St.title = X.title
            cy = C.x.copy() if C.x else None
            cx = X.x.copy() if X.x else None
            St.set_coords(y=cy, x=cx)

        if initSpec:
            if St.coords is None:
                St.set_coords(y=St.y, x=X.x)
            Ct = np.linalg.lstsq(St.data.T, X.data.T)[0]
            C = NDDataset(Ct.T)
            C.name = 'Pure conc. profile, mcs-als of ' + X.name
            C.title = 'Concentration'
            cx = St.y.copy() if St.y else None
            cy = X.y.copy() if X.y else None
            C.set_coords(y=cy, x=cx)

        change = tol + 1
        stdev = X.std()  # .data[0]
        niter = 0
        ndiv = 0

        logs = '*** ALS optimisation log***\n'
        logs += '#iter     Error/PCA        Error/Exp      %change\n'
        logs += '---------------------------------------------------'
        info_(logs)

        while change >= tol and niter < maxit and ndiv < maxdiv:

            C.data = np.linalg.lstsq(St.data.T, X.data.T)[0].T
            niter += 1

            # Force non-negative concentration
            # --------------------------------
            if np.nonzero(nonnegConc)[0].size != 0:
                for s in np.nditer(np.nonzero(nonnegConc)):
                    C.data[:, s] = C.data[:, s].clip(min=0)

            # Force unimodal concentration
            # ----------------------------
            if np.nonzero(unimodConc)[0].size != 0:
                for s in np.nditer(np.nonzero(unimodConc)):
                    maxid = np.argmax(C.data[:, s])
                    curmax = C.data[maxid, s]
                    curid = maxid

                    while curid > 0:
                        curid -= 1
                        if C.data[curid, s] > curmax * unimodTol:
                            if unimodMod == 'strict':
                                C.data[curid, s] = C.data[curid + 1, s]
                            if unimodMod == 'smooth':
                                C.data[curid, s] = (C.data[curid, s] + C.data[
                                    curid + 1, s]) / 2
                                C.data[curid + 1, s] = C.data[curid, s]
                                curid = curid + 2
                        curmax = C.data[curid, s]

                    curid = maxid
                    while curid < ny - 1:
                        curid += 1
                        if C.data[curid, s] > curmax * unimodTol:
                            if unimodMod == 'strict':
                                C.data[curid, s] = C.data[curid - 1, s]
                            if unimodMod == 'smooth':
                                C.data[curid, s] = (C.data[curid, s] + C.data[
                                    curid - 1, s]) / 2
                                C.data[curid - 1, s] = C.data[curid, s]
                                curid = curid - 2
                        curmax = C.data[curid, s]

            # Force monotonic increase
            # ------------------------
            if np.nonzero(monoIncConc)[0].size != 0:
                for s in np.nditer(np.nonzero(monoIncConc)):
                    for curid in np.arange(ny - 1):
                        if C.data[curid + 1, s] < C.data[curid, s] / monoIncTol:
                            C.data[curid + 1, s] = C.data[curid, s]

            # Force monotonic decrease
            # ------------------------
            if np.nonzero(monoDecConc)[0].size != 0:
                for s in np.nditer(np.nonzero(monoDecConc)):
                    for curid in np.arange(ny - 1):
                        if C.data[curid + 1, s] > C.data[curid, s] * monoDecTol:
                            C.data[curid + 1, s] = C.data[curid, s]

            St.data = np.linalg.lstsq(C.data, X.data)[0]

            # Force non-negative spectra
            # --------------------------
            if np.nonzero(nonnegSpec)[0].size != 0:
                for s in np.nditer(np.nonzero(nonnegSpec)):
                    St.data[s, :] = St.data[s, :].clip(min=0)

            # rescale spectra & concentrations
            if normSpec == 'max':
                alpha = np.max(St.data, axis=1).reshape(nspecies, 1)
                St.data = St.data / alpha
                C.data = C.data * alpha.T
            elif normSpec == 'euclid':
                alpha = np.linalg.norm(St.data, axis=1).reshape(nspecies, 1)
                St.data = St.data / alpha
                C.data = C.data * alpha.T

            # compute residuals
            # -----------------
            X_hat = dot(C, St)
            stdev2 = (X_hat - X).std()
            change = 100 * (stdev2 - stdev) / stdev

            stdev_PCA = (X_hat - Xpca).std()  # TODO: Check PCA: values are different from the Arnaud version ?

            logentry = '{:3d}      {:10f}      {:10f}      {:10f}'.format(niter, stdev_PCA, stdev2, change)
            logs += logentry + '\n'
            info_(logentry)
            stdev = stdev2

            if change > 0:
                ndiv += 1
            else:
                ndiv = 0
                change = -change

            if change < tol:
                logentry = 'converged !'
                logs += logentry + '\n'
                info_(logentry)

            if ndiv == maxdiv:
                logline = 'Optimization not improved since {} iterations... unconverged or \'tol\' set too small ?\n'.format(
                    maxdiv)
                logline += 'Stop ALS optimization'
                logs += logline + '\n'
                info_(logline)

            if niter == maxit:
                logline = 'Convergence criterion (\'tol\') not reached after {:d} iterations.'.format(maxit)
                logline += 'Stop ALS optimization'
                logs += logline + '\n'
                info_(logline)

        self._X = X
        self._param = param
        self._C = C
        self._St = St
        self._log = logs

    def transform(self):
        """
        Return the concentration and spectra matrix determined by MCRALS

        Returns
        -------
        C: |NDDataset|
            The concentration matrix
        St: |NDDataset|
            The spectra matrix

        """
        return self._C, self._St

    def inverse_transform(self):
        """
        Transform data back to the original space.

        The following matrice operation is performed: :math:`X'_{hat} = C'.S'^t`

        Returns
        -------
        X_hat: |NDDataset|
            The reconstructed dataset based on the MCS-ALS optimization.

        """

        # reconstruct from concentration and spectra profiles
        C = self._C
        St = self._St

        X_hat = dot(C, St)

        X_hat.history = 'Dataset reconstructed by MCS ALS optimization'
        X_hat.title = 'X_hat: ' + self._X.title
        return X_hat

    def plot(self, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals

        Returns
        -------
        ax: subplot

        """

        colX, colXhat, colRes = kwargs.get('colors', ['blue', 'green', 'red'])

        X_hat = self.inverse_transform()

        res = self._X - X_hat

        ax = self._X.plot(labbel='$X$')
        ax.plot(X_hat.data.T, color=colXhat, label='$\hat{X}')
        ax.plot(res.data.T, color=colRes, label='Residual')
        ax.set_title('MCR ALS plot: ' + self._X.name)

        return ax
