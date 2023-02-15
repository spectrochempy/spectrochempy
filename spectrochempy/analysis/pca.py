# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement several wrapper to scikit-learn model and estimators
"""
import numpy as np
import traitlets as tr
from numpy.random import RandomState
from sklearn import decomposition
from traittypes import Array

from spectrochempy.analysis.abstractanalysis import AnalysisConfigurable
from spectrochempy.utils import exceptions

__all__ = ["PCA"]
__configurables__ = __all__


class PCA(AnalysisConfigurable):
    """
    PCA analysis is here done using the sklearn PCA model.

    We just implement fit, reduce, reconstruct and fit_reconstruct
    """

    name = tr.Unicode("PCA")
    description = tr.Unicode("Scikit-learn PCA model")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    # _X already defined in the superclass
    # define here only the variable that you use in fit or transform functions
    _VT = Array(allow_none=True, help="Loadings")
    _S = Array(allow_none=True, help="Scores")
    _svd = tr.List(Array())

    _pca = tr.Instance(decomposition.PCA)

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    # centered = tr.Bool(
    #     default_value=True,
    #     help="If True the data are centered around the mean values: "
    #     ":math:`X' = X - mean(X)`",
    # ).tag(config=True)

    # sklearn PCA is always on centered data

    standardized = tr.Bool(
        default_value=False,
        help="If True the data are scaled to unit standard deviation: "
        ":math:`X' = X / \\sigma`",
    ).tag(config=True)

    scaled = tr.Bool(
        default_value=False,
        help="If True the data are scaled in the interval [0-1]: "
        ":math:`X' = (X - min(X)) / (max(X)-min(X))`",
    ).tag(config=True)

    n_components = tr.Union(
        (tr.Enum(["mle"]), tr.Int(), tr.Float()),
        allow_none=True,
        default_value=None,
        help="""Number of components to keep.
if n_components is not set all components are kept::
    n_components == min(n_observations, n_features)
If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's MLE is used to guess
the dimension. Use of ``n_components == 'mle'`` will interpret ``svd_solver == 'auto'``
as ``svd_solver == 'full'``.
If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the number of
components such that the amount of variance that needs to be explained is greater than
the percentage specified by n_components.
If ``svd_solver == 'arpack'``, the number of components must be strictly less than the
minimum of n_features and n_observations. Hence, the None case results in::
    n_components == min(n_observations, n_features) - 1""",
    ).tag(config=True)

    copy = tr.Bool(
        default_value=True,
        help="""If False, data passed to fit are overwritten and running
fit(X).transform(X) will not yield the expected results,
use fit_transform(X) instead.""",
    ).tag(config=True)

    whiten = tr.Bool(
        default_value=False,
        help="""When True (False by default) the `components_` vectors are multiplied
by the square root of n_observations and then divided by the singular values to ensure
uncorrelated outputs with unit component-wise variances. Whitening will remove some
information from the transformed signal (the relative variance scales of the components)
but can sometime improve the predictive accuracy of the downstream estimators by making
their data respect some hard-wired assumptions.""",
    ).tag(config=True)

    svd_solver = tr.Enum(
        ["auto", "full", "arpack", "randomized"],
        default_value="auto",
        help="""If auto :
The solver is selected by a default policy based on `X.shape`
and `n_components`: if the input data is larger than 500x500 and the number of
components to extract is lower than 80% of the smallest dimension of the data, then the
more efficient 'randomized' method is enabled. Otherwise the exact full SVD is computed
and optionally truncated afterwards.
If full :
run exact full SVD calling the standard LAPACK solver via `scipy.linalg.svd` and select
the components by postprocessing
If arpack :
run SVD truncated to n_components calling ARPACK solver via `scipy.sparse.linalg.svds`.
It requires strictly 0 < n_components < min(X.shape)
If randomized :
run randomized SVD by the method of Halko et al.""",
    ).tag(config=True)

    tol = tr.Float(
        default_value=0.0,
        help="""Tolerance for singular values computed by svd_solver == 'arpack'.
Must be of range [0.0, infinity).""",
    ).tag(config=True)

    iterated_power = tr.Union(
        (tr.Int(), tr.Enum(["auto"])),
        default_value="auto",
        help="""Number of iterations for the power method computed by
svd_solver == 'randomized'. Must be of range [0, infinity).""",
    ).tag(config=True)

    n_oversamples = tr.Int(
        default_value=10,
        help="""This parameter is only relevant when `svd_solver="randomized"`.
It corresponds to the additional number of random vectors to sample the range of `X` so
as to ensure proper conditioning. See :func:`~sklearn.utils.extmath.randomized_svd`
for more details.""",
    ).tag(config=True)

    power_iteration_normalizer = tr.Enum(
        ["auto", "QR", "LU", "none"],
        default_value="auto",
        help="""Power iteration normalizer for randomized SVD solver. Not used by
ARPACK. See :func:`~sklearn.utils.extmath.randomized_svd` for more details.""",
    ).tag(config=True)

    random_state = tr.Union(
        (tr.Int(), tr.Instance(RandomState)),
        allow_none=True,
        default_value=None,
        help="""Used when the 'arpack' or 'randomized' solvers are used. Pass an int
for reproducible results across multiple function calls.""",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        copy=True,
        log_level="WARNING",
        config=None,
        warm_start=False,
        **kwargs,
    ):
        # call the super class for initialisation
        super().__init__(
            copy=True,
            log_level=log_level,
            warm_start=warm_start,
            config=config,
            **kwargs,
        )

        # initialize sklearn PCA
        self._pca = decomposition.PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __str__(self, n_pc=5):

        if not self._fitted:
            raise exceptions.NotFittedError(
                f"The fit method must be used prior using the {self.name} model"
            )

        s = "\n"
        s += "PC\tEigenvalue\t\t%variance\t\t%cumulative\n"
        s += "  \t of cov(X)\t\t   per PC\t\t   variance\n"

        n_pc = min(n_pc, len(self.ev.data))
        for i in range(n_pc):
            tup = (
                i + 1,
                np.sqrt(self.ev.data[i]),
                self.ev_ratio.data[i],
                self.ev_cum.data[i],
            )
            s += "#{}\t{:10.3e}\t\t{:9.3f}\t\t{:11.3f}\n".format(*tup)

        return s

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):

        X = change.new

        # Standardization
        # ---------------
        if self.standardized:
            self._std = X.std(dim=0)
            X /= self._std
            X.name = f"standardized {X.name}"

        # Scaling
        # -------
        if self.scaled:
            self._min = X.min(dim=0)
            self._ampl = X.ptp(dim=0)
            X -= self._min
            X /= self._ampl
            X.name = "scaled %s" % X.name

        # we keep only the data
        self._X_preprocessed = X.data

    def _fit(self, X):
        # this method is called by the abstract class fit.
        # Input X is a np.ndarray

        # call the sklearn _fit function on data (it outputs SVD results)
        # _outfit is a tuple handle the eventual output of _fit for further processing.
        # The _outfit members are np.ndarrays
        self._outfit = self._pca._fit(X)

        # get the calculated attributes
        self.noise_variance_ = self._pca.noise_variance_
        self.n_observations_ = self._pca.n_samples_
        self.components_ = self._pca.components_
        self.n_components_ = self._pca.n_components_
        self.explained_variance_ = self._pca.explained_variance_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.singular_values_ = self._pca.singular_values_

    def _reduce(self, X, **kwargs):
        return self._pca.transform(X)

    def _reconstruct(self, X_reduced):
        return self._pca.inverse_transform(X_reduced)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from spectrochempy import MASKED, NDDataset
    from spectrochempy.utils import testing

    dataset = NDDataset.read("irdata/nh4y-activation.spg")
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca1 = PCA()
    pca1.fit(dataset)

    assert pca1._X.shape == (55, 5216), "missing row or col removed"
    assert testing.assert_dataset_equal(
        pca1.X, dataset
    ), "input dataset should be reflected in the internal variable X"

    # display scores
    scores1 = pca1.reduce(n_components=2)
    pca1.scoreplot(scores1, 1, 2)
    plt.show()

    # show loadings
    loadings1 = pca1.get_components(n_components=2)
    loadings1.plot(legend=True)
    plt.show()

    # reconstruct
    X_hat = pca1.reconstruct(scores1)
    pca1.plotmerit(dataset, X_hat)
    plt.show()

    # two other valid ways to get the reduction
    # 1
    scores2 = PCA().fit_reduce(dataset, n_components=2)
    assert testing.assert_dataset_equal(scores2, scores1)
    # 2
    scores3 = PCA().fit(dataset).reduce(n_components=2)
    assert testing.assert_dataset_equal(scores3, scores1)
