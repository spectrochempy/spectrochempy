# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implement the EFA (Evolving Factor Analysis) class.
"""
from datetime import datetime, timezone

import numpy as np
from traitlets import HasTraits, Instance, Float

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils import MASKED
from .svd import SVD

__all__ = ["EFA"]

__dataset_methods__ = []


# from spectrochempy.core.plotters.plot1d import plot_multiple
# from spectrochempy.utils import show


class EFA(HasTraits):
    """
    Evolving Factor Analysis.

    Performs an Evolving Factor Analysis (forward and reverse) of the input |NDDataset|.
    """

    _f_ev = Instance(NDDataset)
    _b_ev = Instance(NDDataset)
    _cutoff = Float(allow_none=True)

    def __init__(self, dataset):
        """
        Parameters
        ----------
        dataset : |NDDataset| object
            The input dataset has shape (M, N). M is the number of
            observations (for examples a series of IR spectra) while N
            is the number of features (for example the wavenumbers measured
            in each IR spectrum).

        """
        super().__init__()

        # check if we have the correct input
        # ----------------------------------

        X = dataset

        if isinstance(X, NDDataset):
            # As seen below, we cannot performs SVD on the masked array
            # so let's take the ndarray only
            self._X = X
            M, N = X.shape
        else:
            raise TypeError(
                f"An object of type NDDataset is expected as input, but an object of type"
                f" {type(X).__name__} has been provided"
            )

        # max number of components
        K = min(M, N)

        # in case some row are masked, we need to take this into account
        if X.is_masked:
            masked_rows = np.all(X.mask, axis=-1)
        else:
            masked_rows = np.array([False] * M)

        K = min(K, len(np.where(~masked_rows)[0]))

        # --------------------------------------------------------------------
        # forward analysis
        # --------------------------------------------------------------------

        f = NDDataset(
            np.zeros((M, K)),
            coordset=[X.y, Coord(range(K))],
            title="EigenValues",
            description="Forward EFA of " + X.name,
            history=str(datetime.now(timezone.utc)) + ": created by spectrochempy ",
        )

        # in case some row are masked, take this into account, by masking
        # the corresponding rows of f
        f[masked_rows] = MASKED

        # performs the analysis
        for i in range(M):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                fsvd = SVD(X[: i + 1], compute_uv=False)
                k = fsvd.s.size
                # print(i, k)
                f[i, :k] = fsvd.s.data ** 2
                f[i, k:] = MASKED
            else:
                f[i] = MASKED
            print(f"Evolving Factor Analysis: {int(i / (2 * M) * 100)}% \r", end="")
        # --------------------------------------------------------------------
        # backward analysis
        # --------------------------------------------------------------------

        b = NDDataset(
            np.zeros((M, K)),
            coordset=[X.y, Coord(range(K))],
            title="EigenValues",
            name="Backward EFA of " + X.name,
            history=str(datetime.now(timezone.utc)) + ": created by spectrochempy ",
        )

        b[masked_rows] = MASKED

        for i in range(M - 1, -1, -1):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                bsvd = SVD(X[i:M], compute_uv=False)
                k = bsvd.s.size
                b[i, :k] = bsvd.s.data ** 2
                b[i, k:] = MASKED
            else:
                b[i] = MASKED
            print(
                f"Evolving Factor Analysis: {int(100 - i / (2 * M) * 100)} % \r", end=""
            )

        self._f_ev = f
        self._b_ev = b

    @property
    def cutoff(self):
        """
        Cutoff value (float).
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, val):
        self._cutoff = val

    @property
    def f_ev(self):
        """
        Eigenvalues for the forward analysis (|NDDataset|).
        """
        f = self._f_ev
        if self._cutoff is not None:
            f.data = np.max((f.data, np.ones_like(f.data) * self._cutoff), axis=0)
        return f

    @property
    def b_ev(self):
        """
        Eigenvalues for the backward analysis (|NDDataset|).
        """
        b = self._b_ev
        if self.cutoff is not None:
            b.data = np.max((b.data, np.ones_like(b.data) * self.cutoff), axis=0)
        return b

    def get_conc(self, n_pc=None):
        """
        Computes abstract concentration profile (first in - first out).

        Parameters
        ----------
        n_pc : int, optional, default:3
            Number of pure species for which the concentration profile must be
            computed.

        Returns
        --------
        concentrations
            Concentration profile.
        """
        M, K = self.f_ev.shape
        if n_pc is None:
            n_pc = K
        n_pc = min(K, n_pc)

        f = self.f_ev
        b = self.b_ev

        xcoord = Coord(range(n_pc), title="PS#")
        c = NDDataset(
            np.zeros((M, n_pc)),
            coordset=CoordSet(y=self._X.y, x=xcoord),
            name=f"C_EFA[{self._X.name}]",
            title="relative concentration",
            description="Concentration profile from EFA",
            history=f"{datetime.now(timezone.utc)}: created by spectrochempy",
        )
        if self._X.is_masked:
            masked_rows = np.all(self._X.mask, axis=-1)
        else:
            masked_rows = np.array([False] * M)

        for i in range(M):
            if masked_rows[i]:
                c[i] = MASKED
                continue
            c[i] = np.min((f.data[i, :n_pc], b.data[i, :n_pc][::-1]), axis=0)
        return c


# ======================================================================================================================
if __name__ == "__main__":
    pass
