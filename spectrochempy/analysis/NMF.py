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

from spectrochempy.analysis._analysisutils import NotFittedError
from spectrochempy.analysis.abstractanalysis import DecompositionAnalysisConfigurable

__all__ = ["NMF"]
__configurables__ = ["NMF"]


# ======================================================================================
# class FastICA
# ======================================================================================
class NMF(DecompositionAnalysisConfigurable):
    """Non-Negative Matrix Factorization (NMF)

    Use `~sklearn.NMF` .

    Find two non-negative matrices, i.e. matrices with all non-negative elements, (W, H)
    whose product approximates the non-negative matrix X. This factorization can be used
    for example for dimensionality reduction, source separation or topic extraction.

    Parameters
    ----------
    log_level : ["INFO", "DEBUG", "WARNING", "ERROR"], optional, default:"WARNING"
        The log level at startup
    config : Config object, optional
        By default the configuration is determined by the NMF.cfg.py
        file in the configuration directory. A traitlets.config.Config() object can
        eventually be used here.
    warm_start : bool, optional, default: false
        When fitting with NMF repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        it may be possible to reuse previous model learned from the previous parameter
        value, saving time.
        When warm_start is true, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to fit.
    **kwargs
        Optional configuration  parameters.

    See Also
    --------
    PCA : Perform Principal Components Analysis.
    EFA : Perform an Evolving Factor Analysis (forward and reverse).
    MCRALS : Perform MCR-ALS of a dataset knowing the initial C or St matrix.
    SVD : Perform a Singular Value Decomposition.
    SIMPLISMA : SIMPLe to use Interactive Self-modeling Mixture Analysis.
    """

    name = tr.Unicode("NMF")
    description = tr.Unicode("Scikit-learn NMF model")

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

    used_components = tr.Integer(
        default_value=None,
        allow_none=True,
        help="Number of components to use. If None is passed, all are used.",
    ).tag(config=True)

    init = tr.Enum(
        ["random", "nndsvd", "nndsvda", "nndsvdar", "custom"],
        default_value=None,
        allow_none=True,
        help="""Method used to initialize the procedure.
Valid options:
- `None`: 'nndsvda' if n_components <= min(n_samples, n_features),
  otherwise random.
- `'random'`: non-negative random matrices, scaled with:
  sqrt(X.mean() / n_components)
- `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
  initialization (better for sparseness)
- `'nndsvda'`: NNDSVD with zeros filled with the average of X
  (better when sparsity is not desired)
- `'nndsvdar'` NNDSVD with zeros filled with small random values
  (generally faster, less accurate alternative to NNDSVDa
  for when sparsity is not desired)
- `'custom'`: use custom matrices W and H""",
    ).tag(config=True)

    solver = tr.Enum(
        ["cd", "mu"],
        default_value="cd",
        help="""Numerical solver to use:
    - 'cd' is a Coordinate Descent solver.
    - 'mu' is a Multiplicative Update solver.""",
    ).tag(config=True)

    beta_loss = tr.Union(
        (tr.Float(), tr.Enum(["frobenius", "kullback-leibler", "itakura-saito"])),
        default_value="frobenius",
        help="""Beta divergence to be minimized, measuring the distance between X
and the dot product WH. Note that values different from 'frobenius' (or 2) and
'kullback-leibler' (or 1) lead to significantly slower fits.
Note that for beta_loss <= 0 (or 'itakura-saito'), the input matrix X cannot contain
zeros. Used only in 'mu' solver.""",
    ).tag(config=True)

    tol = tr.Float(default_value=1e-4, help="Tolerance of the stopping condition.").tag(
        config=True
    )

    maxit = tr.Integer(
        default_value=200, help="Maximum number of iterations before timing out."
    ).tag(config=True)

    random_state = tr.Union(
        (tr.Integer(), tr.Instance(RandomState)),
        allow_none=True,
        default_value=None,
        help="""Used for initialisation (when ``init`` == 'nndsvdar' or 'random'), and
in Coordinate Descent. Pass an int, for reproducible results across multiple function
calls.""",
    ).tag(config=True)

    alpha_W = tr.Float(
        default_value=0.0,
        help="Constant that multiplies the regularization terms of `W`. Set it to zero"
        "(default) to have no regularization on `W`.",
    ).tag(config=True)

    alpha_H = tr.Union(
        (tr.Float(), tr.Unicode("same")),
        default_value="same",
        help="""Constant that multiplies the regularization terms of `H`. Set it to zero
to have no regularization on `H`. If "same" (default), it takes the same value as
`alpha_W`.""",
    ).tag(config=True)

    l1_ratio = tr.Float(
        default_value=0.0,
        help="""The regularization mixing parameter, with 0 <= l1_ratio <= 1.
For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm).
For l1_ratio = 1 it is an elementwise L1 penalty.
For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.""",
    ).tag(config=True)

    # Probably log_level is enough
    # verbose = tr.Integer(default_value=0,
    #     help="Whether to be verbose.").tag(config=True)

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
        config=None,
        warm_start=False,
        copy=True,
        **kwargs,
    ):
        # we have changed the name n_components use in sklearn by
        # used_components (in order  to avoid conflict with the rest of the program)
        # warn th user:
        if "n_components" in kwargs:
            raise KeyError(
                "`n_components` is not a valid parameter. Did-you mean "
                "`used_components`?"
            )

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            config=config,
            copy=copy,
            **kwargs,
        )

        # initialize sklearn NMF
        self._nmf = decomposition.NMF(
            n_components=self.used_components,
            init=self.init,
            beta_loss=self.beta_loss,
            tol=self.tol,
            max_iter=self.maxit,
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

    # ----------------------------------------------------------------------------------
    # Public methods and properties specific to NMF
    # ----------------------------------------------------------------------------------
    @property
    def n_components(self):
        """
        Return the number of components that were fitted.
        """
        if self._fitted:
            return self._n_components
        else:
            raise NotFittedError("n_components")
