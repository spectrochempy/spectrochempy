# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Implementation of Principal Component Analysis (using scikit-learn library)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import traitlets as tr
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
from numpy.random import RandomState
from sklearn import decomposition

from spectrochempy import preferences as prefs
from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._analysisbase import _wrap_ndarray_output_to_nddataset
from spectrochempy.application.application import info_
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docutils import docprocess

__all__ = ["PCA"]
__configurables__ = ["PCA"]


# ======================================================================================
# class PCA
# ======================================================================================
@signature_has_configurable_traits
class PCA(DecompositionAnalysis):
    docprocess.delete_params("DecompositionAnalysis.see_also", "PCA")

    __doc__ = docprocess.dedent(
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
    """,
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
        r":math:`X' = X / \sigma`.",
    ).tag(config=True)

    scaled = tr.Bool(
        default_value=False,
        help=r"If True the data are scaled in the interval ``[0-1]``\ : "
        r":math:`X' = (X - min(X)) / (max(X)-min(X))`.",
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
            deprecated("used_components", replace="n_components", removed="0.7")
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
            X.name = f"scaled {X.name}"

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
                    "n_components='mle' is only supported if n_observations >= n_features",
                )
        elif not 0 <= n_components <= min(n_observations, n_features):
            raise ValueError(
                f"n_components={n_components!r} must be between 0 and "
                f"min(n_observations, n_features)={min(n_observations, n_features)!r} with "
                "svd_solver='full'",
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
            self._pca.n_components_,
        )  # cast the returned int64 to int
        self.n_components = self._n_components
        return _outfit

    def _transform(self, X):
        return self._pca.transform(X)

    def _inverse_transform(self, X_transform):
        # we need to  set self._pca.components_ to a compatible size but without
        # destroying the full matrix:
        store_components_ = self._pca.components_
        if X_transform.ndim == 1:
            X_transform = X_transform.reshape(-1, 1)
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
    docprocess.keep_params("analysis_fit.parameters", "X")

    @docprocess.dedent
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
        """Return PCA loadings."""
        return self.get_components()

    @property
    def scores(self):
        """Returns PCA scores."""
        return self.transform(self.X)

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title="explained variance",
        typesingle="components",
    )
    def explained_variance(self):
        return self._pca.explained_variance_

    ev = explained_variance

    @property
    @_wrap_ndarray_output_to_nddataset(
        units="percent",
        title="explained variance ratio",
        typesingle="components",
    )
    def explained_variance_ratio(self):
        return self._pca.explained_variance_ratio_ * 100.0

    ev_ratio = explained_variance_ratio

    @property
    @_wrap_ndarray_output_to_nddataset(
        units="percent",
        title="cumulative explained variance",
        typesingle="components",
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
                f"The fit method must be used prior using the {self.name} model",
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
            raise NotFittedError("The fit method must be used before using this method")

        if n_components is None or n_components > self.n_components:
            n_components = self.n_components
        info_(self.__str__(n_components))

    # ----------------------------------------------------------------------------------
    # Plot methods specific to PCA
    # ----------------------------------------------------------------------------------
    def plot_scree(self, n_components=None, **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA.

        Explained variance by each PC is plotted as a bar graph (left y axis)
        and cumulative explained variance is plotted as a line with markers
        (right y axis).

        Parameters
        ----------
        n_components : int, optional
            Number of components to plot. If None, plots all components.
        **kwargs
            Additional keyword arguments passed to :func:`plot_scree`.
            See :func:`~spectrochempy.plotting.composite.plotscree.plot_scree`
            for available options (e.g., ``bar_color``, ``line_color``,
            ``title``, ``ax``, ``show``).

        Returns
        -------
        matplotlib.axes.Axes
            The primary axes (left y-axis with bars).

        See Also
        --------
        plot_scree : Standalone scree plot function.

        Examples
        --------
        >>> import spectrochempy as scp
        >>> X = scp.read("irdata/nh4y-activation.spg")
        >>> pca = scp.PCA(n_components=5)
        >>> pca.fit(X)
        >>> ax = pca.plot_scree(show=False)  # doctest: +SKIP
        """
        from spectrochempy.plotting.composite import plot_scree

        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before using this method",
            )

        if n_components is None:
            n_components = self.n_components
        else:
            n_components = min(self.n_components, n_components)

        return plot_scree(
            self.ev_ratio.data[:n_components],
            cumulative=self.ev_cum.data[:n_components],
            **kwargs,
        )

    def screeplot(self, n_components=None, **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA.

        .. deprecated:: 0.7.4
            Use :meth:`plot_scree` instead.

        Parameters
        ----------
        n_components : int, optional
            Number of components to plot.
        **kwargs
            Additional keyword arguments (ignored, for backward compatibility).

        Returns
        -------
        matplotlib.axes.Axes
            The primary axes.
        """
        warnings.warn(
            "PCA.screeplot() is deprecated; use PCA.plot_scree() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot_scree(n_components=n_components, **kwargs)

    def plot_score(self, scores=None, components=(1, 2), **kwargs):
        """
        2D or 3D score plot of observations.

        Plots the projection of each observation/spectrum onto the span of
        two or three selected principal components.

        Parameters
        ----------
        scores : NDDataset or tuple, optional
            Scores dataset to plot. If None, uses `self.scores`.
            Pass a modified scores dataset (e.g., with custom labels)
            to use those labels in the plot.

            Note: If a tuple or list is passed as the first positional
            argument, it is interpreted as `components` for backward
            compatibility.
        components : tuple of int, optional
            Principal components to plot (1-based indexing).
            Length 2 for 2D plot, length 3 for 3D plot.
            Default: (1, 2).
        **kwargs
            Additional keyword arguments passed to :func:`plot_score`.
            See :func:`~spectrochempy.plotting.composite.plotscore.plot_score`
            for available options:

            - ``cmap`` : Colormap for coloring points.
            - ``color`` : Fixed color or color values for each point.
            - ``color_mapping`` : "index" (default) or "labels" - how to map colors.
            - ``show_labels`` : If True, annotate points with labels.
            - ``labels_column`` : Column index in scores.y.labels (0-based).
            - ``ax`` : Axes to plot on.
            - ``show`` : Whether to display the figure.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes.

        See Also
        --------
        plot_score : Standalone score plot function.

        Examples
        --------
        >>> import spectrochempy as scp
        >>> X = scp.read("irdata/nh4y-activation.spg")
        >>> pca = scp.PCA(n_components=5)
        >>> pca.fit(X)
        >>> ax = pca.plot_score((1, 2), show=False)  # doctest: +SKIP

        With custom labels:

        >>> scores = pca.transform()  # doctest: +SKIP
        >>> scores.y.labels = custom_labels  # doctest: +SKIP
        >>> ax = pca.plot_score(scores=scores, show_labels=True)  # doctest: +SKIP
        """
        from spectrochempy.plotting.composite import plot_score

        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before using this method",
            )

        if isinstance(scores, tuple | list) and all(isinstance(x, int) for x in scores):
            components = tuple(scores)
            scores = None

        if scores is None:
            scores = self.scores

        return plot_score(
            scores,
            components=components,
            **kwargs,
        )

    def scoreplot(
        self,
        *args,
        **kwargs,
    ):
        """
        2D or 3D scoreplot of observations.

        .. deprecated:: 0.7.4
            Use :meth:`plot_score` instead.

        Parameters
        ----------
        *args
            Positional arguments. Accepts:
            - (i, j) or (i, j, k): component indices (1-based)
            - (scores, i, j): NDDataset and component indices
        **kwargs
            Additional keyword arguments passed to :meth:`plot_score`.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes.
        """
        warnings.warn(
            "PCA.scoreplot() is deprecated; use PCA.plot_score() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        scores = None
        components = (1, 2)

        if len(args) == 0:
            pass
        elif len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                components = tuple(args[0])
            elif hasattr(args[0], "_implements") and args[0]._implements("NDDataset"):
                scores = args[0]
        elif len(args) == 2:
            if hasattr(args[0], "_implements") and args[0]._implements("NDDataset"):
                scores = args[0]
                components = (args[1],)
            else:
                components = (args[0], args[1])
        elif len(args) >= 3:
            if hasattr(args[0], "_implements") and args[0]._implements("NDDataset"):
                scores = args[0]
                components = (args[1], args[2])
                if len(args) > 3:
                    components = (args[1], args[2], args[3])
            else:
                components = tuple(args[:3])

        return self.plot_score(scores=scores, components=components, **kwargs)
