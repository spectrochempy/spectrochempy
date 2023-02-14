# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement several wrapper to scikit-learn model and estimators
"""
from functools import wraps

import numpy as np
import traitlets as tr
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.random import RandomState
from sklearn import decomposition

from spectrochempy.analysis.abstractanalysis import AnalysisConfigurable
from spectrochempy.core.dataset.coord import Coord

__all__ = ["SKL_PCA"]
__configurables__ = __all__


class _wrap_sklearn_output_to_nddataset:
    def __init__(self, method, **kwargs):
        self.method = method

    def __get__(self, instance, cls):
        @wraps(self.method)
        def wrapper(*args, **kwargs):
            # determine the input X dataset
            if isinstance(args[0], NDDataset):
                X = args[0]
            else:
                X = instance.X
            # get the sklearn data output
            data = self.method(instance, *args, **kwargs)
            # make a new dataset with this
            X_transf = NDDataset(data)
            # Now set the NDDataset attributes
            X_transf.units = X.units
            X_transf.name = f"{X.name}_{instance.name}.{self.method.__name__}"
            X_transf.history = (
                f"Created by method {instance.name}.{self.method.__name__}"
            )
            # title = kw.pop("title", f"{X.title} transformed by {self.method} method")
            # X_transf.title = title
            n_samples, n_features = X.shape

            # set coordinates
            if X_transf.shape == X.shape:
                X.transf.set_coordset(y=X.y, x=X.x)
            elif X_transf.shape == (instance.n_components, n_features):
                X_transf.set_coordset(
                    y=Coord(
                        None,
                        labels=["#%d" % (i + 1) for i in range(X_transf.shape[0])],
                        title="principal component",
                    ),
                    x=X.x,
                )
            elif X_transf.shape == (n_samples, instance.n_components):
                X_transf.set_coordset(
                    y=X.y,
                    x=Coord(
                        None,
                        labels=["#%d" % (i + 1) for i in range(X_transf.shape[1])],
                        title="principal component",
                    ),
                )
            return X_transf

        return wrapper

    def __call__(self, *args, **kwargs):
        # this is the case when using the decorator with property
        # (WARNING: must be placed after @property)
        return self.__get__(args[0], None)


class SKL_PCA(AnalysisConfigurable):
    """
    PCA analysis is here done using the sklearn PCA estimator.

    We just implement fit, reduce, reconstruct and fit_reconstruct
    """

    name = tr.Unicode("SKL_PCA")
    description = tr.Unicode("Scikit-learn PCA model")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    # _X already defined in the superclass
    # define here only the variable that you use in fit or transform functions

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
    n_components == min(n_samples, n_features)
If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's MLE is used to guess
the dimension. Use of ``n_components == 'mle'`` will interpret ``svd_solver == 'auto'``
as ``svd_solver == 'full'``.
If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the number of
components such that the amount of variance that needs to be explained is greater than
the percentage specified by n_components.
If ``svd_solver == 'arpack'``, the number of components must be strictly less than the
minimum of n_features and n_samples. Hence, the None case results in::
    n_components == min(n_samples, n_features) - 1""",
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
by the square root of n_samples and then divided by the singular values to ensure
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
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):

        Xsc = self.X

        # mean center the dataset
        # -----------------------
        if self.centered:
            self._center = center = Xsc.mean(dim=0)
            Xsc = Xsc - center
            Xsc.name = f"centered {Xsc.name}"

        # Standardization
        # ---------------
        if self.standardized:
            self._std = Xsc.std(dim=0)
            Xsc /= self._std
            Xsc.name = f"standardized {Xsc.name}"

        # Scaling
        # -------
        if self.scaled:
            self._min = Xsc.min(dim=0)
            self._ampl = Xsc.ptp(dim=0)
            Xsc -= self._min
            Xsc /= self._ampl
            Xsc.name = "scaled %s" % Xsc.name

        self._Xscaled = Xsc

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Fit the PCA model

        Parameters
        ----------
        X : |NDDataset| object
            The input dataset has shape (M, N). M is the number of
            observations (for examples a series of IR spectra) while N
            is the number of features (for example the wavenumbers measured
            in each IR spectrum).
        y : ignored
        """
        # fire the preprocessing validation
        self.X = X

        # Xscaled has been computed when X was set
        Xsc = self._Xscaled
        # call the sklearn _fit function on data
        # (it outputs SVD results)
        self._svd = self._pca._fit(Xsc.data)
        self._fitted = True
        return self

    @_wrap_sklearn_output_to_nddataset
    def transform(self, X):
        if not self._fitted:
            return
        return self._pca.transform(X.data)

    # @property
    # @_wrap_sklearn_output_to_nddataset
    # def LT(self):
    #     """
    #     LT.
    #     """
    #     if not self._pca._fitted:
    #         return
    #     LT = self._pca.components_
    #     return LT

    # @property
    # def S(self):
    #     """
    #     S.
    #     """
    #     if not self._pca._fitted:
    #         return
    #     S = self._pca.sc
    #

    def scoreplot(
        self, Xhat, *pcs, colormap="viridis", color_mapping="index", **kwargs
    ):
        """
        2D or 3D scoreplot of samples.

        Parameters
        ----------
        *pcs : a series of int argument or a list/tuple
            Must contain 2 or 3 elements.
        colormap : str
            A matplotlib colormap.
        color_mapping : 'index' or 'labels'
            If 'index', then the colors of each n_scores is mapped sequentially
            on the colormap. If labels, the labels of the n_observation are
            used for color mapping.
        """
        if not self._fitted:
            raise exceptions.NotFittedError(
                "The fit method must be used " "before using this method"
            )

        self.prefs = self.X.preferences

        if isinstance(pcs[0], (list, tuple, set)):
            pcs = pcs[0]

        # transform to internal index of component's index (1->0 etc...)
        pcs = np.array(pcs) - 1

        # colors
        if color_mapping == "index":

            if np.any(Xhat.y.data):
                colors = Xhat.y.data
            else:
                colors = np.array(range(Xhat.shape[0]))

        elif color_mapping == "labels":

            labels = list(set(Xhat.y.labels))
            colors = [labels.index(lab) for lab in Xhat.y.labels]

        if len(pcs) == 2:
            # bidimensional score plot

            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
            ax.set_title("Score plot")

            # ax.set_xlabel(
            #     "PC# {} ({:.3f}%)".format(pcs[0] + 1, self.ev_ratio.data[pcs[0]])
            # )
            # ax.set_ylabel(
            #     "PC# {} ({:.3f}%)".format(pcs[1] + 1, self.ev_ratio.data[pcs[1]])
            # )
            axsc = ax.scatter(
                Xhat.masked_data[:, pcs[0]],
                Xhat.masked_data[:, pcs[1]],
                s=30,
                c=colors,
                cmap=colormap,
            )

            number_x_labels = self.prefs.number_of_x_labels  # get from config
            number_y_labels = self.prefs.number_of_y_labels
            # the next two line are to avoid multipliers in axis scale
            y_formatter = ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
            ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

        if len(pcs) == 3:
            # tridimensional score plot
            plt.figure(**kwargs)
            ax = plt.axes(projection="3d")
            ax.set_title("Score plot")
            ax.set_xlabel(
                "PC# {} ({:.3f}%)".format(pcs[0] + 1, self.ev_ratio.data[pcs[0]])
            )
            ax.set_ylabel(
                "PC# {} ({:.3f}%)".format(pcs[1] + 1, self.ev_ratio.data[pcs[1]])
            )
            ax.set_zlabel(
                "PC# {} ({:.3f}%)".format(pcs[2] + 1, self.ev_ratio.data[pcs[2]])
            )
            axsc = ax.scatter(
                Xhat.masked_data[:, pcs[0]],
                Xhat.masked_data[:, pcs[1]],
                Xhat.masked_data[:, pcs[2]],
                zdir="z",
                s=30,
                c=colors,
                cmap=colormap,
                depthshade=True,
            )

        if color_mapping == "labels":
            import matplotlib.patches as mpatches

            leg = []
            for lab in labels:
                i = labels.index(lab)
                c = axsc.get_cmap().colors[int(255 / (len(labels) - 1) * i)]
                leg.append(mpatches.Patch(color=c, label=lab))

            ax.legend(handles=leg, loc="best")

        return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from spectrochempy import MASKED, NDDataset
    from spectrochempy.utils import exceptions

    dataset = NDDataset.read("irdata/nh4y-activation.spg")
    # dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca = SKL_PCA(n_components=2, svd_solver="full")
    pca.fit(dataset)
    x_hat = pca.transform(dataset)
    pca.scoreplot(x_hat, 1, 2)
    plt.show()

    # compare with our PCA
    from spectrochempy.analysis.pca import PCA

    pca1 = PCA()
    pca1.fit(dataset)
    x1_hat, _ = pca1.reduce(n_pc=2)
    pca1.scoreplot(1, 2)
    plt.show()

    from spectrochempy.utils import testing

    assert testing.assert_array_equal(x_hat.data, x1_hat[0].data)
