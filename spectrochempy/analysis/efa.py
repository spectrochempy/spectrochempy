# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement the EFA (Evolving Factor Analysis) class.
"""
import numpy as np
import traitlets as tr

from spectrochempy.analysis._base import (
    DecompositionAnalysis,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.core import info_
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring

__all__ = ["EFA"]
__configurables__ = ["EFA"]


@signature_has_configurable_traits
class EFA(DecompositionAnalysis):

    _docstring.delete_params("DecompositionAnalysis.see_also", "EFA")

    __doc__ = _docstring.dedent(
        r"""
    Evolving Factor Analysis (EFA).

    Evolving factor analysis (`EFA`\ ) is a method that allows model-free resolution of
    overlapping peaks into concentration profiles and normalized spectra of components.

    Originally developed for GC and GC-MS experiments (See *e.g.,*
    :cite:t:`maeder:1986` , :cite:t:`roach:1992`\ ), it is also suitable for
    analysis spectra such as those obtained by Operando FTIR for example.

    The model used in this class allow to perform a forward and reverse analysis of the
    input `NDDataset` .

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_EFA)s

    Examples
    --------
    >>> # Init the model
    >>> model = scp.EFA()
    >>> # Read an experimental 2D spectra (N x M )
    >>> X = scp.read("irdata/nh4y-activation.spg")
    >>> # Fit the model
    >>> _ = model.fit(X)
    >>> # Display components spectra (2 x M)
    >>> model.n_components = 2
    >>> _ = model.components.plot(title="Component spectra")
    >>> # Get the abstract concentration profile based on the FIFO EFA analysis
    >>> c = model.transform()
    >>> # Plot the transposed concentration matrix  (2 x N)
    >>> _ = c.T.plot(title="Concentration")
    >>> scp.show()
    """
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------
    cutoff = tr.Float(default_value=None, allow_none=True, help="Cut-off value.").tag(
        config=True
    )

    n_components = tr.Int(
        allow_none=True, default_value=None, help="Number of components to keep."
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):

        # Call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )
        if "used_components" in kwargs:
            deprecated("used_components", replace="n_components", removed="0.6.5")
            kwargs["n_components"] = kwargs.pop("used_components")

    def _fit(self, X, Y=None):
        # X has already been validated and eventually
        # preprocessed. X is now a nd-array with masked elements removed.
        # and this method should return _outfit
        # Y is not used but necessary to fit the superclass

        # max number of components
        M, N = X.shape
        K = min(M, N)

        # ------------------------------------------------------------------------------
        # forward analysis
        # ------------------------------------------------------------------------------
        f = np.zeros((M, K))
        for i in range(M):

            s = np.linalg.svd(X[: i + 1], compute_uv=False)
            k = s.size
            f[i, :k] = s**2
            info_(f"Evolving Factor Analysis: {int(i / (2 * M) * 100)}% \r")

        # ------------------------------------------------------------------------------
        # backward analysis
        # ------------------------------------------------------------------------------
        b = np.zeros((M, K))
        for i in range(M - 1, -1, -1):
            # if some rows are masked, we must skip them
            s = np.linalg.svd(X[i:M], compute_uv=False)
            k = s.size
            b[i, :k] = s**2
            info_(f"Evolving Factor Analysis: {int(100 - i / (2 * M) * 100)} % \r")

        # store the components number (real or desired)
        self._n_components = K

        # return results
        _outfit = f, b
        return _outfit

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _transform(self, X=None):
        # X is ignored for EFA
        # Return concentration profile
        return self._get_conc()

    def _get_conc(self):
        f, b = self._outfit
        M = f.shape[0]
        K = self._n_components
        if self.n_components is not None:
            K = min(K, self.n_components)
        c = np.zeros((M, K))
        for i in range(M):
            c[i] = np.min((f[i, :K], b[i, :K][::-1]), axis=0)
        return c

    def _get_components(self):
        # compute the components from the original dataset and the EFA concentrations
        St = np.dot(self._get_conc().T, self._X_preprocessed)
        return St

    # ----------------------------------------------------------------------------------
    # Public methods/properties
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def fit(self, X):
        """
        Fit the `EFA` model on a `X` dataset.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s

        Returns
        -------
        %(analysis_fit.returns)s

        See Also
        --------
        %(analysis_fit.see_also)s
        """
        return super().fit(X, Y=None)

    @_docstring.dedent
    def fit_transform(self, X, **kwargs):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        %(kwargs)s

        Returns
        -------
        %(analysis_transform.returns)s

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s
        """
        return super().fit_transform(X, **kwargs)

    def inverse_transform(self):
        """Not implemented."""

    def reconstruct(self):
        """Not implemented."""

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title="keep", typex="components")
    def f_ev(self):
        """
        Eigenvalues for the forward analysis ( `NDDataset` ).
        """
        f = self._outfit[0]
        if self.cutoff is not None:
            f = np.max((f, np.ones_like(f) * self.cutoff), axis=0)
        return f

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title="keep", typex="components")
    def b_ev(self):
        """
        Eigenvalues for the backward analysis ( `NDDataset` ).
        """
        b = self._outfit[1]
        if self.cutoff is not None:
            b = np.max((b, np.ones_like(b) * self.cutoff), axis=0)
        return b
