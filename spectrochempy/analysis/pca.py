# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of Principal Component Analysis (using scikit-learn library)
"""
import matplotlib.pyplot as plt
import numpy as np
import traitlets as tr
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.random import RandomState
from sklearn import decomposition

from spectrochempy.analysis._base import (
    DecompositionAnalysis,
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.plots import NBlue, NRed

__all__ = ["PCA"]
__configurables__ = ["PCA"]


# ======================================================================================
# class PCA
# ======================================================================================
@signature_has_configurable_traits
class PCA(DecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "PCA")

    __doc__ = _docstring.dedent(
        """
    Principal Component Anamysis (PCA).

    The Principal Component Analysis analysis is using the
    `sklearn.decomposition.PCA` model.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_PCA)s
    """
    )

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to PCA, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions
    _pca = tr.Instance(
        decomposition.PCA,
        help="The instance of sklearn.decomposition.PCA used in this model",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    # sklearn PCA is always on centered data
    standardized = tr.Bool(
        default_value=False,
        help=r"If True the data are scaled to unit standard deviation: "
        ":math:`X' = X / \sigma`\ .",
    ).tag(config=True)

    scaled = tr.Bool(
        default_value=False,
        help="If True the data are scaled in the interval ``[0-1]``\ : "
        ":math:`X' = (X - min(X)) / (max(X)-min(X))`\ .",
    ).tag(config=True)

    n_components = tr.Union(
        (tr.Enum(["mle"]), tr.Int(), tr.Float()),
        allow_none=True,
        default_value=None,
        help="""Number of components to keep.
if `n_components` is not set all components are kept::

    n_components == min(n_observations, n_features)

If ``n_components == 'mle'`` and ``svd_solver == 'full'`` , Minka's MLE is used to guess
the dimension. Use of ``n_components == 'mle'`` will interpret `svd_solver == 'auto'`
as ``svd_solver == 'full'`` .
If `0 < n_components < 1` and `svd_solver == 'full'` , select the number of
components such that the amount of variance that needs to be explained is greater than
the percentage specified by n_components.
If `svd_solver == 'arpack'` , the number of components must be strictly less than the
minimum of n_features and n_observations. Hence, the None case results in::

    n_components == min(n_observations, n_features) - 1.""",
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
run SVD truncated to n_components calling ARPACK solver via `scipy.sparse.linalg.svds` .
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
        help="""This parameter is only relevant when `svd_solver="randomized"` .
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

        # initialize sklearn PCA
        self._pca = decomposition.PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
            random_state=self.random_state,
        )

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

        self._X_preprocessed = X.data

        # final check on the configuration n_components parameter
        # (which can be done only when X is defined in fit arguments)
        n_observations, n_features = X.shape

        n_components = self.n_components
        if n_components is None:
            pass
        elif n_components == "mle":
            if n_observations < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_observations >= n_features"
                )
        elif not 0 <= n_components <= min(n_observations, n_features):
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_observations, n_features)=%r with "
                "svd_solver='full'" % (n_components, min(n_observations, n_features))
            )

    def _fit(self, X, Y=None):
        # this method is called by the abstract class fit.
        # Input X is a np.ndarray
        # Y is ignored in this model

        # call the sklearn _fit function on data (it outputs SVD results)
        # _outfit is a tuple handle the eventual output of _fit for further processing.

        # The _outfit members are np.ndarrays
        _outfit = self._pca.fit(X)

        # get the calculated attribute
        self._components = self._pca.components_

        self._noise_variance = self._pca.noise_variance_
        self._n_observations = self._pca.n_samples_
        self._explained_variance = self._pca.explained_variance_
        self._explained_variance_ratio = self._pca.explained_variance_ratio_
        self._singular_values = self._pca.singular_values_

        # unlike to sklearn, we will update the n_components value here with the
        # eventually calculated ones: this will simplify further process
        # indeed in sklearn, the value after processing is n_components_
        # with an underscore at the end

        self._n_components = int(
            self._pca.n_components_
        )  # cast the returned int64 to int
        self.n_components = self._n_components
        return _outfit

    def _transform(self, X):
        return self._pca.transform(X)

    def _inverse_transform(self, X_transform):
        # we need to  set self._pca.components_ to a compatible size but without
        # destroying the full matrix:
        store_components_ = self._pca.components_
        self._pca.components_ = self._pca.components_[: X_transform.shape[1]]
        X = self._pca.inverse_transform(X_transform)
        # restore
        self._pca.components_ = store_components_
        return X

    def _get_components(self):
        self._components = self._pca.components_
        return self._components

    # ----------------------------------------------------------------------------------
    # Public methods and properties specific to PCA
    # ----------------------------------------------------------------------------------
    _docstring.keep_params("analysis_fit.parameters", "X")

    @_docstring.dedent
    def fit(self, X):
        """
        Fit the PCA model on X.

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

    @property
    def loadings(self):
        """
        Return PCA loadings.
        """
        return self.get_components()

    @property
    def scores(self):
        """
        Returns PCA scores.
        """
        return self.transform(self.X)

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None, title="explained variance", typesingle="components"
    )
    def explained_variance(self):
        return self._pca.explained_variance_

    ev = explained_variance

    @property
    @_wrap_ndarray_output_to_nddataset(
        units="percent", title="explained variance ratio", typesingle="components"
    )
    def explained_variance_ratio(self):
        return self._pca.explained_variance_ratio_ * 100.0

    ev_ratio = explained_variance_ratio

    @property
    @_wrap_ndarray_output_to_nddataset(
        units="percent", title="cumulative explained variance", typesingle="components"
    )
    def cumulative_explained_variance(self):
        return np.cumsum(self._pca.explained_variance_ratio_) * 100.0

    ev_cum = cumulative_explained_variance

    # ----------------------------------------------------------------------------------
    # Reporting specific to PCA
    # ----------------------------------------------------------------------------------
    def __str__(self, n_components=5):

        if not self._fitted:
            raise NotFittedError(
                f"The fit method must be used prior using the {self.name} model"
            )

        s = "\n"
        s += "PC\tEigenvalue\t\t%variance\t\t%cumulative\n"
        s += "  \t of cov(X)\t\t   per PC\t\t   variance\n"

        if n_components is None or n_components > self.n_components:
            n_components = self.n_components
        for i in range(n_components):
            tup = (
                i + 1,
                np.sqrt(self.ev.data[i]),
                self.ev_ratio.data[i],
                self.ev_cum.data[i],
            )
            s += "#{}\t{:10.3e}\t\t{:9.3f}\t\t{:11.3f}\n".format(*tup)

        return s

    def printev(self, n_components=None):
        """
        Print PCA figures of merit.

        Prints eigenvalues and explained variance for all or first n_pc PC's.

        Parameters
        ----------
        n_components : int, optional
            The number of components to print.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used " "before using this method"
            )

        if n_components is None or n_components > self.n_components:
            n_components = self.n_components
        print((self.__str__(n_components)))

    # ----------------------------------------------------------------------------------
    # Plot methods specific to PCA
    # ----------------------------------------------------------------------------------
    def screeplot(self, n_components=None, **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA.

        Explained variance by each PC is plot as a bar graph (left y axis)
        and cumulative explained variance is plot as a scatter plot with lines
        (right y axis).

        Parameters
        ----------
        n_components : int
            Number of components to plot.
        **kwargs
            Extra arguments: `colors` (default: ``[NBlue, NRed]`` ) to set the colors
            of the bar plot and scatter plot; ``ylims`` (default ``[(0, 100), "auto"]``\ ).

        Returns
        -------
        `list` of `~matplotlib.axes.Axes`
            The list of axes.
        """
        # get n_components
        if n_components is None:
            n_components = self.n_components
        else:
            n_components = min(self.n_components, n_components)

        color1, color2 = kwargs.get("colors", [NBlue, NRed])
        # pen = kwargs.get('pen', True)
        ylim1, ylim2 = kwargs.get("ylims", [(0, 100), "auto"])

        if ylim2 == "auto":
            y1 = np.around(self.ev_ratio.data[0] * 0.95, -1)
            y2 = 101.0
            ylim2 = (y1, y2)

        ax1 = self.ev_ratio[:n_components].plot_bar(
            ylim=ylim1, color=color1, title="Scree plot"
        )
        ax2 = self.ev_cum[:n_components].plot_scatter(
            ylim=ylim2, color=color2, pen=True, markersize=7.0, twinx=ax1
        )
        ax1.set_title("Scree plot")
        return ax1, ax2

    def scoreplot(
        self,
        *args,
        colormap="viridis",
        color_mapping="index",
        show_labels=False,
        labels_column=0,
        labels_every=1,
        **kwargs,
    ):
        """
        2D or 3D scoreplot of observations.

        Plots the projection of each observation/spectrum onto the span of two or
        three selected principal components.

        Parameters
        ----------
        *args : `NDDataset` and/or series of 2 or 3 ints or iterabble of 2 or 3 int, optional
            The `NDDataset` contains the sores to plot. If not provided `PCA.scores`
            is used. The 2 or 3 int are the PC on which the projection is shown. If not
            provided, default to [1,2], i.e. bidimensional plot on PCs #1 and #2.
        colormap : str
            A matplotlib colormap.
        color_mapping : 'index' or 'labels'
            If 'index', then the colors of each n_scores is mapped sequentially
            on the colormap. If labels, the labels of the n_observations are
            used for color mapping.
        show_labels : bool, optional, default: False
            If True each observation will be annotated with its label.
        labels_column : int, optional, default:0
            If several columns of labels are present indicates which column has to be
            used to show labels.
        labels_every : int, optional, default: 1
            Do not label all points, but only every value indicated by this parameter.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes.
        """
        self.prefs = self.X.preferences

        # checks args
        if len(args) > 0:
            scores = args[0]
            if hasattr(scores, "_implements") and scores._implements("NDDataset"):
                if len(args) > 1:
                    pcs = args[1:]
                else:
                    pcs = 1, 2
            else:
                scores = self.scores
                pcs = args
        else:
            scores = self.scores
            pcs = 1, 2

        if isinstance(pcs[0], (list, tuple, set)):
            pcs = pcs[0]

        # transform to internal index of component's index (1->0 etc...)
        pcs = np.array(pcs) - 1

        # colors
        if color_mapping == "index":
            if np.any(scores.y.data):
                colors = scores.y.data
            else:
                colors = np.array(range(scores.shape[0]))

        elif color_mapping == "labels" and scores.y.labels is not None:
            if scores.y.labels.ndim == 1:
                labels = list(set(scores.y.labels))
            else:
                labels = list(set(scores.y.labels[:, labels_column]))
            colors = [labels.index(lab) for lab in scores.y.labels]

        # labels
        scatterlabels = None
        if show_labels:
            if scores.y.labels is None:
                raise ValueError("You set show_label to true but score.y has no label")
            elif scores.y.labels.ndim == 1:
                scatterlabels = scores.y.labels
            else:
                scatterlabels = scores.y.labels[:, labels_column]

        if len(pcs) == 2:
            # bidimensional score plot

            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
            ax.set_title("Score plot")

            ax.set_xlabel(
                "PC# {} ({:.3f}%)".format(pcs[0] + 1, self.ev_ratio.data[pcs[0]])
            )
            ax.set_ylabel(
                "PC# {} ({:.3f}%)".format(pcs[1] + 1, self.ev_ratio.data[pcs[1]])
            )
            x = scores.masked_data[:, pcs[0]]
            y = scores.masked_data[:, pcs[1]]
            axsc = ax.scatter(x, y, s=30, c=colors, cmap=colormap)

            if scatterlabels is not None:
                for idx, lab in enumerate(scatterlabels):
                    if idx % labels_every != 0:
                        continue
                    ax.annotate(
                        lab,
                        xy=(x[idx], y[idx]),
                        xytext=(-20, 20),
                        textcoords="offset pixels",
                        color=axsc.to_rgba(colors[idx]),
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
                scores.masked_data[:, pcs[0]],
                scores.masked_data[:, pcs[1]],
                scores.masked_data[:, pcs[2]],
                zdir="z",
                s=30,
                c=colors,
                cmap=colormap,
                depthshade=True,
            )

        if color_mapping == "labels" and scores.y.labels is not None:
            import matplotlib.patches as mpatches

            leg = []
            for lab in labels:
                i = labels.index(lab)
                c = axsc.get_cmap().colors[int(255 / (len(labels) - 1) * i)]
                leg.append(mpatches.Patch(color=c, label=lab))

            ax.legend(handles=leg, loc="best")

        return ax
