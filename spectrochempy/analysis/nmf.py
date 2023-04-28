# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of NMF model (using scikit-learn library)
"""
import logging

import traitlets as tr
from numpy.random import RandomState
from sklearn import decomposition

from spectrochempy.analysis._base import DecompositionAnalysis
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring

__all__ = ["NMF"]
__configurables__ = ["NMF"]


# ======================================================================================
# class NMF
# ======================================================================================
@signature_has_configurable_traits
class NMF(DecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "NMF")

    __doc__ = _docstring.dedent(
        """
    Non-Negative Matrix Factorization (NMF).

    Use `sklearn.decomposition.NMF`\ .

    Find two non-negative matrices, *i.e.,* matrices with all non-negative elements,
    (``W``\ , ``H``\ ) whose product approximates the non-negative matrix `X`.
    This factorization can be used for example for dimensionality reduction,
    source separation or topic extraction.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_NMF)s
    """
    )

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to NMF, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions
    _nmf = tr.Instance(
        decomposition.NMF,
        help="The instance of sklearn.decomposition.NMF used in this model",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------

    n_components = tr.Integer(
        default_value=None,
        allow_none=True,
        help="Number of components to use. If None is passed, all are used.",
    ).tag(config=True)

    init = tr.Enum(
        ["random", "nndsvd", "nndsvda", "nndsvdar", "custom"],
        default_value=None,
        allow_none=True,
        help=(
            "Method used to initialize the procedure.\n\n"
            "Valid options:\n\n"
            "* `None` : 'nndsvda' if n_components <= min(n_samples, n_features), "
            "otherwise random.\n"
            "* `random` : non-negative random matrices, scaled with:\n"
            "  sqrt(X.mean() / n_components)\n"
            "* `nndsvd` : Nonnegative Double Singular Value Decomposition (NNDSVD) "
            "initialization (better for sparseness)\n"
            "* `nndsvda` : NNDSVD with zeros filled with the average of X "
            "(better when sparsity is not desired)\n"
            "* `nndsvdar` NNDSVD with zeros filled with small random values "
            "(generally faster, less accurate alternative to NNDSVDa "
            "for when sparsity is not desired)\n"
            "* `custom` : use custom matrices W and H."
        ),
    ).tag(config=True)

    solver = tr.Enum(
        ["cd", "mu"],
        default_value="cd",
        help=(
            "Numerical solver to use:\n"
            "- 'cd' is a Coordinate Descent solver.\n"
            "- 'mu' is a Multiplicative Update solver."
        ),
    ).tag(config=True)

    beta_loss = tr.Union(
        (tr.Float(), tr.Enum(["frobenius", "kullback-leibler", "itakura-saito"])),
        default_value="frobenius",
        help=(
            "Beta divergence to be minimized, measuring the distance between X"
            "and the dot product WH. Note that values different from 'frobenius' "
            "(or 2) and 'kullback-leibler' (or 1) lead to significantly slower fits.\n"
            "Note that for beta_loss <= 0 (or 'itakura-saito'), the input matrix X "
            "cannot contain zeros. Used only in 'mu' solver."
        ),
    ).tag(config=True)

    tol = tr.Float(default_value=1e-4, help="Tolerance of the stopping condition.").tag(
        config=True
    )

    max_iter = tr.Integer(
        default_value=200, help="Maximum number of iterations before timing out."
    ).tag(config=True)

    random_state = tr.Union(
        (tr.Integer(), tr.Instance(RandomState)),
        allow_none=True,
        default_value=None,
        help=(
            "Used for initialisation (when `init` == 'nndsvdar' or 'random'), and "
            "in Coordinate Descent. Pass an int, for reproducible results across "
            "multiple function calls."
        ),
    ).tag(config=True)

    alpha_W = tr.Float(
        default_value=0.0,
        help="Constant that multiplies the regularization terms of `W` . Set it to zero"
        "(default) to have no regularization on `W` .",
    ).tag(config=True)

    alpha_H = tr.Union(
        (tr.Float(), tr.Enum(["same"])),
        default_value="same",
        help=(
            "Constant that multiplies the regularization terms of `H` . Set it to zero"
            'to have no regularization on `H` . If "same" (default), it takes the same'
            "value as `alpha_W` ."
        ),
    ).tag(config=True)

    l1_ratio = tr.Float(
        default_value=0.0,
        help=(
            "The regularization mixing parameter, with 0 <= l1_ratio <= 1.\n"
            "- For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius "
            "Norm).\n"
            "- For l1_ratio = 1 it is an elementwise L1 penalty.\n"
            "- For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2."
        ),
    ).tag(config=True)

    shuffle = tr.Bool(
        default_value=False,
        help="If true, randomize the order of coordinates in the CD solver.",
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
        if "used_components" in kwargs:
            deprecated("used_components", replace="n_components", removed="0.6.5")
            kwargs["n_components"] = kwargs.pop("used_components")

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

        # initialize sklearn NMF
        self._nmf = decomposition.NMF(
            n_components=self.n_components,
            init=self.init,
            beta_loss=self.beta_loss,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio,
            verbose=self.parent.log_level == logging.INFO,
        )

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):
        # this method is called by the abstract class fit.
        # Input X is a np.ndarray
        # Y is ignored in this model

        # call the sklearn _fit function on data
        # _outfit is a tuple handle the eventual output of _fit for further processing.

        # The _outfit members are np.ndarrays
        _outfit = self._nmf.fit(X)
        self._n_components = int(
            self._nmf.n_components
        )  # cast the returned int64 to int
        return _outfit

    def _transform(self, X):
        return self._nmf.transform(X)

    def _inverse_transform(self, X_transform):
        # we need to  set self._nmf.components_ to a compatible size but without
        # destroying the full matrix:
        store_components_ = self._nmf.components_
        self._nmf.components_ = self._nmf.components_[: X_transform.shape[1]]
        X = self._nmf.inverse_transform(X_transform)
        # restore
        self._nmf.components_ = store_components_
        return X

    def _get_components(self):
        self._components = self._nmf.components_
        return self._components

    _docstring.keep_params("analysis_fit.parameters", "X")

    @_docstring.dedent
    def fit(self, X):
        """
        Fit the NMF  model on X.

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
