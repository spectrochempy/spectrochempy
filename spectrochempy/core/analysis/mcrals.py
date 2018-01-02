# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


__all__ = ['MCRALS']

__dataset_methods__ = ['MCRALS']

import numpy as np
from traitlets import HasTraits, Instance

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.application import log


class MCRALS(HasTraits):
    """Performs MCR-ALS of a dataset knowing the initial C or St matrix

    """

    X = Instance(NDDataset)
    guess = Instance(NDDataset)
    C = Instance(NDDataset)
    St = Instance(NDDataset)

    def __init__(self, X, guess, **kwargs):
        """
        Parameters
        -----------
        X : |NDDataset|
            The dataset on which to perfomr the MCR-ALS analysis
        guess : |NDDataset|
            Initial concentration or spectra

        Other Parameters
        ----------------
        **kwargs : optional parameters
            The possible parameters are described below
        tol : float, optional, default=1e-3
            convergence tolerance
        maxit : maximum number of ALS minimizations
        maxdiv : maximum number of non-converging iteratiobs
        nonnegConc : array or tuple indicating species non-negative concentration
                       profiles. For instance [1, 0, 1] indicates that species #0
                       and #2 have non-negative conc profiles while species #1
                       can have negative concentrations
                       Default [1, ..., 1]  (only non-negative cocentrations)
        unimodConc : array or tuple indicating species having unimodal concentrations
                       profiles.
                       Default [1, ..., 1]  (only unimodal cocentration profiles)
        nonnegSpec : array or tuple indicating species having non-negative spectra
                       Default [1, ..., 1]  (only non-negative spectra)
        unimodSpec : array or tuple indicating species having unimodal spectra
                       Default [0, ..., 0]  (no unimodal cocentration profiles)

        """
        # TODO: make a test file

        # Check initial data
        # ------------------------------------------------------------------------

        initConc, initSpec = False, False

        if X.shape[0] == guess.shape[0]:
            initConc = True
            C = guess.copy()
            nspecies = C.shape[1]

        elif X.guess[1] == guess.shape[1]:
            initSpec = True
            St = guess.copy()
            nspecies = St.shape[0]

        else:
            raise ValueError('the dimensions of initial concentration '
                             'or spectra dataset do not match the data')

        nspc, nwn = X.shape

        # Get optional parameters in kwargs or set them to their default
        # ------------------------------------------------------------------------

        # TODO: make a reference  file to set this kwargs
        tol = kwargs.get('tol', 0.001)

        maxit = kwargs.get('maxit', 50)

        maxdiv = kwargs.get('maxdiv', 5)

        nonnegConc = kwargs.get('nonnegConc', [1] * nspecies)

        unimodConc = kwargs.get('unimodConc', [1] * nspecies)

        unimodTol = kwargs.get('unimodTol', 1.1)

        unimodMod = kwargs.get('unimodMod', 'strict')

        monoDecConc = kwargs.get('monoDecConc', [0] * nspecies)

        monoDecTol = kwargs.get('monoDecTol', 1.1)

        monoIncConc = kwargs.get('monoIncConc', [0] * nspecies)

        monoIncTol = kwargs.get('monoIncTol', 1.1)

        nonnegSpec = kwargs.get('nonnegSpec', [1] * nspecies)

        #    if ('unimodSpec' in kwargs): unimodSpec = kwargs['unimodSpec']
        #    else: unimodSpec = np.zeros((1, nspecies))

        # Compute initial spectra or concentrations   (first iteration...)
        # ------------------------------------------------------------------------

        if initConc:
            C_i = C.data
            St_i = np.linalg.lstsq(C_i, X.data)[0]
            self.St = St = NDDataset(St_i)
            St.name = C.name + ' \ ' + X.name
            St.coordset = [C.y, X.x]

        if initSpec:
            St_i = St.data
            Ct_i = np.linalg.lstsq(St_i.T, X.data.T)[0]
            C_i = Ct_i.T
            self.C = C = NDDataset(C_i)
            C.name = X.name + ' / ' + St.name
            C.coordset = [St.y, X.x]

        delta = tol + 1
        niter = 0
        ndiv = 0
        res = np.infty

        while delta >= tol and niter < maxit and ndiv < maxdiv:

            Ct_i = np.linalg.lstsq(St_i.T, X.data.T)[0]
            C_i = Ct_i.T
            niter += 1

            # Force non-negative concentration
            # --------------------------------
            if np.nonzero(nonnegConc)[0].size != 0:
                for s in np.nditer(np.nonzero(nonnegConc)):
                    C_i[:, s] = C_i[:, s].clip(min=0)

            # Force unimodal concentration
            # ----------------------------
            if np.nonzero(unimodConc)[0].size != 0:
                for s in np.nditer(np.nonzero(unimodConc)):
                    maxid = np.argmax(C_i[:, s])
                    curmax = C_i[maxid, s]
                    curid = maxid

                    while curid > 0:
                        curid -= 1
                        if C_i[curid, s] > curmax * unimodTol:
                            if unimodMod == 'strict':
                                C_i[curid, s] = C_i[curid + 1, s]
                            if unimodMod == 'smooth':
                                C_i[curid, s] = (C_i[curid, s] + C_i[
                                    curid + 1, s]) / 2
                                C_i[curid + 1, s] = C_i[curid, s]
                                curid = curid + 2
                        curmax = C_i[curid, s]

                    curid = maxid
                    while curid < nspc - 1:
                        curid += 1
                        if C_i[curid, s] > curmax * unimodTol:
                            if unimodMod == 'strict':
                                C_i[curid, s] = C_i[curid - 1, s]
                            if unimodMod == 'smooth':
                                C_i[curid, s] = (C_i[curid, s] + C_i[
                                    curid - 1, s]) / 2
                                C_i[curid - 1, s] = C_i[curid, s]
                                curid = curid - 2
                        curmax = C_i[curid, s]

            # Force monotonic increase
            # ------------------------
            if np.nonzero(monoIncConc)[0].size != 0:
                for s in np.nditer(np.nonzero(monoIncConc)):
                    for curid in np.arange(nspc - 1):
                        if C_i[curid + 1, s] < C_i[curid, s] / monoIncTol:
                            C_i[curid + 1, s] = C_i[curid, s]

            # Force monotonic decrease
            # ------------------------
            if np.nonzero(monoDecConc)[0].size != 0:
                for s in np.nditer(np.nonzero(monoDecConc)):
                    for curid in np.arange(nspc - 1):
                        if C_i[curid + 1, s] > C_i[curid, s] * monoDecTol:
                            C_i[curid + 1, s] = C_i[curid, s]

            St_i = np.linalg.lstsq(C_i, X.data)[0]

            # Force non-negative spectra
            # --------------------------
            if np.nonzero(nonnegSpec)[0].size != 0:
                for s in np.nditer(np.nonzero(nonnegSpec)):
                    St_i[s, :] = St_i[s, :].clip(min=0)

            # compute residuals
            # -----------------
            res2 = np.linalg.norm(X.data - np.dot(C_i, St_i))
            delta = res2 - res
            res = res2
            log.info(niter, res2, delta)

            if delta > 0:
                ndiv += 1
            else:
                delta = -delta

        C.data = C_i
        St.data = St_i

    def transform(self):
        """
        Return the concentration and spectra matrix determined by MCRALS

        Returns
        -------
        C : |NDDataset|
            The concentration matrix
        St : |NDDataset|
            The spectra matrix

        """
        return self.C, self.St
