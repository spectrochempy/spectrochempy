# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the `Baseline` class for baseline corrections and related methods.
"""

import numpy as np
import scipy.interpolate
import scipy.signal
import traitlets as tr
from scipy import sparse
from scipy.sparse.linalg import spsolve

from spectrochempy.analysis._base import (
    AnalysisConfigurable,
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.analysis.preprocessing.utils import lls, lls_inv
from spectrochempy.application import info_, warning_
from spectrochempy.core.processors.concatenate import concatenate
from spectrochempy.utils.coordrange import trim_ranges
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.misc import TYPE_FLOAT, TYPE_INTEGER
from spectrochempy.utils.traits import NDDatasetType

__all__ = [
    "Baseline",
    "abc",
    "basc",
    "detrend",
    "als",
    "snip",
    "lls",
    "lls_inv",
]

_common_see_also = """
See Also
--------
Baseline : Manual baseline correction.
basc : NDDataset method performing a baseline correction using the `Baseline` class.
abc : NDDataset method performing an automatic baseline correction.
als : NDDataset method performing an Asymmetric Least Squares Smoothing baseline.
    correction.
snip : NDDataset method performing a Simple Non-Iterative Peak (SNIP) detection
    algorithm.
autosub: NDDataset method performing a subtraction of reference.
detrend : NDDataset method performing a remove polynomial trend along a dimension
    from dataset.
"""

_docstring.get_sections(
    _docstring.dedent(_common_see_also),
    base="Baseline",
    sections=["See Also"],
)
_docstring.delete_params("Baseline.see_also", "Baseline")
_docstring.delete_params("Baseline.see_also", "basc")
_docstring.delete_params("Baseline.see_also", "abc")
_docstring.delete_params("Baseline.see_also", "als")
_docstring.delete_params("Baseline.see_also", "snip")
_docstring.delete_params("Baseline.see_also", "autosub")
_docstring.delete_params("Baseline.see_also", "detrend")


# ======================================================================================
# Baseline class processor
# ======================================================================================
@signature_has_configurable_traits
# Note: with this decorator
# Configurable traits are added to the signature as keywords if they are not yet present.
class Baseline(AnalysisConfigurable):
    __doc__ = _docstring.dedent(
        """
    Baseline Correction processor.

    The baseline correction can be applied to 1D datasets consisting in a single row
    with :term:`n_features` or to a 2D dataset with shape (:term:`n_observation`,
    :term:`n_features`\ ).

    When dealing with 2D datasets, the baseline correction can be applied either sequentially (default) or using a multivariate approach (parameter
    `multivariate`set to `True).

    - The ``'sequential'`` approach which can be used for both 1D and 2D datasets
      consists in fitting the baseline sequentially for each observation row (spectrum).

    - The ``'multivariate'`` approach can only be applied to 2D datasets (at least 3
      observations).
      The 2D dataset is first dimensional reduced into several principal
      components using a conventional Singular Value Decomposition :term:`SVD` or a non-negative matrix factorization (`NMF`).
      Each component is then fitted before an inverse transform performed to recover
      the baseline correction.

    In both approaches, various models can be used to estimate the
    baseline.

    - ``'abc'`` : A linear baseline is automatically subtracted using the feature limit
      for reference.
    - ``'detrend'`` : remove trends from data. Depending on the ``order`` parameter,
      the detrend can be constant (mean removal), linear (order=1), quadratic (order=2)
      or `cubic`(order=3).
    - ``'als'`` : Asymmetric Least Squares Smoothing baseline correction. This method
      is based on the work of Eilers and Boelens (:cite:`eilers:2005`\ ).
    - ``'snip'`` : Simple Non-Iterative Peak (SNIP) detection algorithm
      (:cite:`ryan:1988`\ ).
    - ``'polynomial'`` : Fit a nth-degree polynomial to the data. The order of
      the polynomial is defined by the ``order`` parameter. The baseline is then obtained by evaluating the
      polynomial at each feature defined in predefined `ranges`\ .
    - ``'pchip'`` : Fit a piecewise cubic Hermite interpolating polynomial (PCHIP) to
      the data  (:cite:`fritsch:1980`\ ).

    # TODO: complete this description

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(Baseline.see_also.no_Baseline)s
    """
    )

    #     if not ranges and dataset.meta.regions is not None:
    #         # use the range stored in metadata
    #         ranges = dataset.meta.regions["baseline"]

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    multivariate = tr.Bool(
        default_value=False,
        help="For 2D datasets, if `True`, a multivariate method is used to fit a "
        "baseline on the principal components determined using a SVD decomposition"
        "followed by an inverse-transform to retrieve the baseline corrected "
        "dataset. If `False`, a sequential method is used which consists in fitting a "
        "baseline on each row (observations) of the dataset.",
    ).tag(config=True)

    model = tr.CaselessStrEnum(
        ["polynomial", "pchip", "abc", "detrend", "als", "snip"],
        default_value="pchip",
        help="""The model used to determine the baseline.

The following models are required that the `ranges` parameter is provided
(see `ranges` parameter for more details):

* 'polynomial': the baseline is determined by a nth-degree polynomial interpolation.
  It uses the `order` parameter to determine the degree of the polynomial.
* 'pchip': the baseline is determined by a piecewise cubic hermite interpolation.

The others models do not require the `ranges` parameter to be provided:

* 'detrend': the baseline is determined by a constant, linear or polynomial
  trend removal. The order of the trend is determined by the `order` parameter.
* 'als': the baseline is determined by an asymmetric least square algorithm.
* 'snip': the baseline is determined by a simple non-iterative peak detection
  algorithm (for th).
""",
    ).tag(config=True)

    lls = tr.Bool(
        default_value=False,
        help="If `True`, the baseline is determined on data transformed using the "
        "log-log-square transform. This compress the dynamic range of signal and thus "
        "emphasize smaller features. This parameter is always `True` for the 'snip' "
        "model.",
    ).tag(config=True)

    order = tr.Union(
        (
            tr.Integer(),
            tr.CaselessStrEnum(["constant", "linear", "quadratic", "cubic"]),
        ),
        default_value=6,
        help="Polynom order to use for polynomial interpolation or detrend.",
    ).tag(config=True, min=1)

    mu = tr.Float(
        default_value=1e5,
        help="The smoothness parameter for the ALS method. Larger values make the "
        "baseline stiffer. Values should be in the range (0, 1e9).",
    ).tag(config=True)

    asymmetry = tr.Float(
        default_value=0.05,
        help="The asymmetry parameter for the ALS method. It is typically between 0.001 "
        "and 0.1. 0.001 gives almost the same fit as the unconstrained least squares",
    ).tag(config=True)

    snip_width = tr.Integer(
        help="The width of the window used to determine the baseline using the SNIP "
        "algorithm."
    ).tag(config=True, min=0)

    tol = tr.Float(
        default_value=1e-3,
        help="The tolerance parameter for the ALS method. Smaller values make the "
        "fitting better but potentially increases the number of iterations and the "
        "running time. Values should be in the range (0, 1).",
    ).tag(config=True)

    max_iter = tr.Integer(50, help="Maximum number of :term:`ALS` iteration.").tag(
        config=True
    )
    n_components = tr.Integer(
        default_value=5,
        help="Number of components to use for the multivariate method "
        "(:term:`n_observation` >= `n_components`).",
    ).tag(config=True, min=1)

    ranges = tr.List(
        tr.Union((tr.List(minlen=2, maxlen=2), tr.Float())),
        default_value=[],
        help="A sequence of features values or feature's regions which are assumed to "
        "belong to the baseline. Feature ranges are defined as a list of 2 "
        "numerical values (start, end). Single values are internally converted to "
        "a pair (start=value, end=start). The limits of the spectra are "
        "automatically added during the fit process unless the `remove_limit` "
        "parameter is `True`\ ",
    ).tag(config=True)

    include_limits = tr.Bool(
        default_value=True,
        help="Whether to automatically include the features limits "
        "to the specified ranges.",
    ).tag(config=True)

    breakpoints = tr.List(
        default_value=[],
        help="""Breakpoints to define piecewise segments of the data,
specified as a vector containing coordinate values or indices indicating the location
of the breakpoints. Breakpoints are useful when you want to compute separate
baseline/trends for different segments of the data.
""",
    )
    # ----------------------------------------------------------------------------------
    # Runtime parameters
    # ----------------------------------------------------------------------------------
    _X_ranges = NDDatasetType(
        help="The dataset containing only the sections corresponding to _ranges"
    )
    _ranges = tr.List(help="The actual list of ranges after trim and clean-up")

    # ----------------------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):
        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.validate("breakpoints")
    def _validate_breakpoints(self, proposal):
        if proposal.value is None:
            return []
        else:
            return sorted(proposal.value)

    @tr.validate("n_components")
    def _validate_n_components(self, proposal):
        # n_components cannot be higher than the number of observations
        npc = proposal.value
        if self._X_is_missing:
            return npc
        return max(1, min(npc, self._X.shape[0]))

    @tr.validate("order")
    def _validate_order(self, proposal):
        # order provided as string must be transformed to int
        order = proposal.value
        if isinstance(order, str):
            order = {"constant": 0, "linear": 1, "quadratic": 2, "cubic": 3}[
                order.lower()
            ]
        return order

    @tr.validate("ranges")
    def _validate_ranges(self, proposal):
        # ranges must be transformed using trim_range
        ranges = proposal.value

        if isinstance(ranges, tuple) and len(ranges) == 1:
            ranges = ranges[0]  # probably passed with no start to the compute function
        if not isinstance(ranges, (list, tuple)):
            ranges = list(ranges)

        if len(ranges) == 2:
            if isinstance(ranges[0], TYPE_INTEGER + TYPE_FLOAT) and isinstance(
                ranges[1], TYPE_INTEGER + TYPE_FLOAT
            ):
                # a pair a values, we interpret this as a single range
                ranges = [[ranges[0], ranges[1]]]

        # prepare the final ranges : find the single values
        for i, item in enumerate(ranges[:]):
            if isinstance(item, TYPE_INTEGER + TYPE_FLOAT):
                # a single numerical value: interpret this as a single range
                item = [item, item]
            ranges[i] = item

        # clean the result (reorder and suppress overlap)
        return trim_ranges(*ranges)

    @tr.observe("_X", "ranges", "include_limits", "model", "multivariate")
    def _preprocess_as_X_or_ranges_changed(self, change):
        # set X and ranges using the new or current value
        X = None
        if change.name == "_X":
            X = change.new
        elif self._fitted:
            X = self._X

        if not self._fitted and X is None:
            # nothing to do
            return

        if self.model not in ["polynomial", "pchip", "abc"]:
            # such as detrend, or als we work on the full data so range is the
            # full feature range.
            self._X_ranges = X.copy()
            return

        ranges = change.new if change.name == "ranges" else self.ranges.copy()
        if X is None:
            # not a lot to do here
            # just return after cleaning ranges
            self._ranges = ranges = trim_ranges(*ranges)
            return

        if self.include_limits and X is not None:
            # be sure to extend ranges with the extrema of the x-axis
            # (it will not make a difference if this was already done before)
            lastcoord = X.coordset[X.dims[-1]]  # we have to take into account the
            # possibility of transposed data, so we can't use directly X.x
            x = lastcoord.data
            if self.model != "abc":
                ranges += [[x[0], x[2]], [x[-3], x[-1]]]
            else:
                r = (x[-1] - x[-1]) * 0.05
                ranges += [[x[0], x[0] + r], [x[-1] - r, x[-1]]]

        # trim, order and clean up ranges (save it in self._ranges)
        self._ranges = ranges = trim_ranges(*ranges)

        # Extract the dataset sections corresponding to the provided ranges
        # BUT warning, this does not work for masked data (as after removal of the
        # masked part, the  X.x coordinates is not more linear .The coordinate must have
        # been transformed to normal coordinate before. See AnalysisConfigurable.
        # _X_validate
        s = []
        for pair in ranges:
            # determine the slices
            sl = slice(*pair)
            sect = X[..., sl]
            if sect is None:
                continue
            s.append(sect)

        # determine _X_ranges (used by fit) by concatenating the sections
        self._X_ranges = concatenate(s)

        # we will also do necessary validation of other parameters:
        if self.multivariate:
            self.n_components = self.n_components  # fire the validation

    def _fit(self, xbase, ybase):
        # core _fit method:
        # calculate the baseline according to the current approch and model

        # get the last coordinate of the dataset
        lastcoord = self._X.coordset[self._X.dims[-1]]
        x = lastcoord.data

        # get the number of observations and features
        M, N = self._X.shape if self._X.ndim == 2 else (1, self._X.shape[0])

        # eventually transform the compress y data dynamic using log-log-square
        # operator. lls requires positive data.
        if self.model == "snip":
            self.lls = True
        offset_lls = ybase.min() if self.lls else 0
        Y = ybase if not self.lls else lls(ybase - offset_lls)

        # when using `nmf` multivariate factorization, it works much better when data
        # start at zero so we will remove the minimum value of the data and restore it
        # after the fit
        offset_nmf = Y.min() if self.multivariate == "nmf" else 0
        Y = Y - offset_nmf

        # Initialization varies according to the approach used (multivariate or not)
        # ---------------------------------------------------------------------------
        if not self.multivariate:
            # sequential method
            _store = np.zeros((M, N))
        else:
            # multivariate method
            U, s, Vt = np.linalg.svd(Y, full_matrices=False, compute_uv=True)
            M = self.n_components
            Y = Vt[0:M]
            _store = np.zeros((M, N))

        # -----------------------------------------
        # Polynomial interpolation, detrend or abc
        # -----------------------------------------
        if self.model in ["polynomial", "detrend", "abc"]:
            # polynomial interpolation or detrend process
            # using parameter `order` and predetermined ranges
            polycoef = np.polynomial.polynomial.polyfit(
                xbase, Y.T, deg=self.order, rcond=None, full=False
            )
            _store = np.polynomial.polynomial.polyval(x, polycoef)

        # -------------------------------------------------------------------------
        # PChip interpolation (piecewise cubic hermite interpolation) using ranges
        # -------------------------------------------------------------------------
        elif self.model == "pchip":
            # pchip interpolation
            for i in range(M):
                interp = scipy.interpolate.PchipInterpolator(xbase, Y[i])
                _store[i] = interp(x)

        # -----------------------------------------------------
        # Simple Non-Iterative Peak (SNIP) detection algorithm
        # -----------------------------------------------------
        # based on :cite:`Ryan1988` ,
        # and  https://stackoverflow.com/questions/57350711/
        # baseline-correction-for-spectroscopic-data
        elif self.model == "snip":
            # SNIP baseline correction

            # First phase: transform the data Y -> LLS(Y - offset)
            # This has been done already, so proceed to the next phase

            # Second phase: multipass peak clipping loop
            # on the scanned window
            for w in range(self.snip_width - 8):
                mean = (np.roll(Y, -w, axis=1) + np.roll(Y, w, axis=1)) / 2
                Y[:, w : N - w] = np.minimum(Y[:, w : N - w], mean[:, w : N - w])

            # Third phase: reduce progressively the snip_width for the last passes by a
            # factor sqrt(2) (cf. ryan:1988)
            f = np.sqrt(2)
            for w in range(self.snip_width - 8, self.snip_width):
                w = int(np.ceil(w / f))  # decrease the window size by factor f
                mean = (np.roll(Y, -w, axis=1) + np.roll(Y, w, axis=1)) / 2
                Y[:, w : N - w] = np.minimum(Y[:, w : N - w], mean[:, w : N - w])
                f = f * np.sqrt(2)  # in next iteration the window size will
                # be decreased by another factor sqrt(2)

            # Last phase: do an inverse transform Y = LLS^-1(G) + offset
            # will be done later
            _store = Y

        # ------------------------
        # ALS baseline correction
        # ------------------------
        # see
        # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
        elif self.model == "als":
            # ALS fitted baseline
            # For now, this doesn't work with masked data
            mu = self.mu
            p = self.asymmetry
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(N, N - 2))

            for i in range(M):
                w = np.ones(N)
                w_old = 1e5
                y = Y[i].squeeze()
                # make the data positive (to allow the use of NNMF instead of SVD)
                mi = y.min()
                y -= mi
                iter = 0
                while True:
                    W = sparse.spdiags(w, 0, N, N)
                    C = W + mu * D.dot(D.transpose())
                    z = spsolve(C, w * y)
                    w = p * (y > z) + (1 - p) * (y < z)
                    change = np.sum(np.abs(w_old - w)) / N
                    info_(change)
                    if change <= self.tol:
                        info_(f"Convergence reached in {iter} iterations")
                        break
                    if iter >= self.max_iter:
                        info_(f"Maximum number of iterations {self.max_iter} reached")
                        break
                    w_old = w
                    iter += 1
                # do not forget to add to mi to get the original data back
                z += mi
                _store[i] = z[::-1] if self._X.x.is_descendant else z

        # inverse transform to get the baseline in the original data space
        # this depends on the approach used (multivariate or not)
        if self.multivariate:
            T = U[:, 0:M] @ np.diag(s)[0:M, 0:M]
            baseline = T @ _store
        else:
            baseline = _store

        # eventually inverse lls transform
        if self.lls:
            baseline = lls_inv(baseline) + offset_lls

        # eventually add the offset for nmf
        if self.multivariate == "nmf":
            baseline += offset_nmf

        # unsqueeze data (needed to restore masks properly)
        if baseline.ndim == 1:
            baseline = baseline[np.newaxis]

        # now return the baseline
        return baseline

    # ----------------------------------------------------------------------------------
    # Public methods/properties
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def fit(self, X):
        """
        Fit a baseline model on a `X` dataset.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s

        Returns
        -------
        %(analysis_fit.returns)s
        """
        self._fitted = False  # reinit this flag

        # Set X
        # -----
        X = X.copy()
        if self.model == "als":  # ALS doesn't work with masked data (see _fit)
            # so we will remove the mask and restore it after the fit
            self.Xmasked = X.copy()
            X.remove_masks()

        # fire the X and _ranges validation and preprocessing.
        self._X = X

        # _X_ranges has been computed when X and _ranges were set,
        # but we need increasing order of the coordinates
        self._X_ranges.sort(inplace=True, descend=False)

        # _X_ranges is now ready, we can fit. _Xranges contains
        # only the baseline data to fit
        ybase = self._X_ranges.data  # baseline data
        lastcoord = self._X_ranges.coordset[self._X_ranges.dims[-1]]
        xbase = lastcoord.data  # baseline x-axis

        # # Handling breakpoints  # TODO: to make it work
        # # --------------------
        # # include the extrema of the x-axis as breakpoints
        # bpil = [0, self._X.shape[-1] - 1]
        # # breakpoints can be provided as indices or as values.
        # # we convert them to indices.
        # for bp in self.breakpoints:
        #     bpil = self._X.x.loc2index(bp) if isinstance(bp, TYPE_FLOAT) else bp
        #     bpil.append(bpil)
        # # sort and remove duplicates
        # bpil = sorted(list(set(bpil)))
        #
        # # loop on breakpoints pairs
        # baseline = np.zeros_like(self._X)
        # bpstart = 0
        # for bpend in bpil[1:]:
        #     # fit the baseline on each segment
        #     xb = xbase[bpstart : bpend + 1]
        #     yb = ybase[bpstart : bpend + 1]
        #     baseline[bpstart : bpend + 1] = self._fit(xb, yb)
        # self._outfit = [
        #     baseline,
        #     bpil,
        # ]

        baseline = self._fit(xbase, ybase)
        self._outfit = [
            baseline,
        ]

        # if the process was successful, _fitted is set to True so that other method
        # which needs fit will be possibly used.
        self._fitted = True
        return self

    @property
    @_wrap_ndarray_output_to_nddataset
    def baseline(self):
        """Computed baseline."""
        if not self._fitted:
            raise NotFittedError
        baseline = self._outfit[0]
        return baseline

    def transform(self):
        """Return a dataset with baseline removed."""
        if self.model == "als" and hasattr(self, "Xmasked"):
            corrected = self.Xmasked - self.baseline
        else:
            corrected = self.X - self.baseline
        return corrected

    @property
    def corrected(self):
        """Dataset with baseline removed."""
        return self.transform()

    @_docstring.dedent
    def parameters(self, default=False):
        """
        %(MetaConfigurable.parameters_doc)s
        """
        d = super().parameters(default)
        if not default:
            d.ranges = self._ranges
        return d

    @property
    def used_ranges(self):
        """
        The actual ranges used during fitting

        Eventually the features limits are included and the list returned is trimmed,
        cleaned and ordered.
        """
        return self._ranges

    def show_regions(self, ax):
        if hasattr(self, "_sps") and self._sps:
            for sp in self._sps:
                sp.remove()
        # self._sps = []
        for range in self._ranges:
            range.sort()
            sp = ax.axvspan(range[0], range[1], facecolor="#2ca02c", alpha=0.5)
            # self._sps.append(sp)


# ======================================================================================
# API / NDDataset functions
# ======================================================================================
# Instead of using directly the Baseline class, we provide here some functions
# which are eventually more user friendly and which can be used directly on NDDataset or
# called from the API.


@_docstring.dedent
def basc(dataset, *ranges, **kwargs):
    r"""
    Compute a baseline correction using the Baseline class processor.

    See `Baseline` for detailled information on the parameters.

    Parameters
    ----------
    dataset : a `NDDataset` instance
        The dataset where to calculate the baseline.
    *ranges : a variable number of pair-tuples
        The regions taken into account for the manual baseline correction.
    **kwargs
        Optional keyword parameters (see `Baseline` Parameters for a detailled
        information).

    Returns
    -------
    `NDDataset`
        The baseline corrected dataset

    See Also
    --------
    %(Baseline.see_also.no_basc)s

    Notes
    -----
    For more flexibility and functionality, it is advised to use the Baseline class
    processor instead.
    """

    blc = Baseline()
    if ranges:
        blc.ranges = ranges
    for key in kwargs:
        setattr(blc, key, kwargs[key])
    blc.fit(dataset)
    return blc.transform()


@_docstring.dedent
def detrend(dataset, order="linear", breakpoints=[], **kwargs):
    """
    Remove polynomial trend along a dimension from dataset.

    Depending on the ``order``parameter, `detrend` removes the best-fit polynomial line
    (in the least squares sense) from the data and returns the remaining data.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    order : non-negative `int` or a `str` among ['constant', 'linear', 'quadratic', 'cubic'], optional, default='linear'
        The order of the polynomial trend.

        * If ``order=0`` or ``'constant'``\ , the mean of data is subtracted to remove
          a shift trend.
        * If ``order=1`` or ``'linear'`` (default), the best straith-fit line is
          subtracted from data to remove a linear trend (drift).
        * If order=2 or ``order=quadratic``\ ,  the best fitted nth-degree polynomial
          line is subtracted from data to remove a quadratic polynomial trend.
        * ``order=n`` can also be used to remove any nth-degree polynomial trend.

    breakpoints : :term:`array_like`\ , optional
        Breakpoints to define piecewise segments of the data, specified as a vector
        containing coordinate values or indices indicating the location of the
        breakpoints. Breakpoints are useful when you want to compute separate trends
        for different segments of the data.

    Returns
    -------
    `NDDataset`
        The detrended dataset.

    See Also
    --------
    %(Baseline.see_also.no_detrend)s
    """

    # kwargs will be removed in version 0.8
    inplace = kwargs.pop("inplace", None)
    if inplace is not None:
        warning_("inplace parameter was removed in version 0.7 and has no more effect.")

    type = kwargs.pop("type", None)
    if type is not None:
        deprecated("type", replace="order", removed="0.8")
        order = type

    dim = kwargs.pop("dim", None)
    if dim is not None:
        deprecated(
            "dim",
            extra_msg="Transpose your data before processing if needed.",
            removed="0.8",
        )

    blc = Baseline()
    blc.model = "detrend"
    blc.order = order
    blc.breakpoints = breakpoints
    blc.fit(dataset)

    return blc.transform()


@_docstring.dedent
def als(dataset, mu=1e5, asymmetry=0.05, tol=1e-3, max_iter=50):
    """
    Asymmetric Least Squares Smoothing baseline correction.

    This method is based on the work of Eilers and Boelens (:cite:`eilers:2005`\ ).

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    mu : `float`, optional, default:1e5
        The smoothness parameter for the ALS method. Larger values make the
        baseline stiffer. Values should be in the range (0, 1e9)
    asymmetry : `float`, optional, default:0.05,
        The asymmetry parameter for the ALS method. It is typically between 0.001
        and 0.1. 0.001 gives almost the same fit as the unconstrained least squares.
    tol = `float`, optional, default:1e-3
        The tolerance parameter for the ALS method. Smaller values make the fitting better but potentially increases the number of iterations and the running time. Values should be in the range (0, 1).
    max_iter = `int`, optional, default:50
        Maximum number of :term:`ALS` iteration.

    Returns
    -------
    `NDDataset`
        The baseline corrected dataset.

    See Also
    --------
    %(Baseline.see_also.no_als)s
    """
    blc = Baseline()
    blc.model = "als"
    blc.asymmetry = asymmetry
    blc.tol = tol
    blc.max_iter = max_iter
    blc.fit(dataset)

    return blc.transform()


@_docstring.dedent
def snip(dataset, snip_width=50):
    """
    Simple Non-Iterative Peak (SNIP) detection algorithm

    See :cite:t:`ryan:1988` .

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    snip_width : `int`, optional, default:50
        The width of the window used to determine the baseline using the SNIP algorithm.

    Returns
    -------
    `NDDataset`
        The baseline corrected dataset.

    See Also
    --------
    %(Baseline.see_also.no_snip)s
    """
    blc = Baseline()
    blc.model = "snip"
    blc.snip_width = snip_width
    blc.fit(dataset)

    return blc.transform()


@_docstring.dedent
def abc(dataset, model="linear", window=0.05, nbzone=32, mult=4, order=5, **kwargs):
    """
    Automatic baseline correction.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    model : `str`, optional, default: 'linear'
        The baseline correction model to use. Available models are:

        * ``'linear'``: linear baseline correction using the limits of the dataset.

    See Also
    --------
    %(Baseline.see_also.no_abc)s

    # TODO add other methods
    """

    blc = Baseline()
    blc.model = "abc"
    if model == "linear":
        blc.ranges = []
        blc.include_limits = True
        blc.order = 1
    else:
        raise ValueError(f"Unknown model {model}")
    blc.fit(dataset)

    return blc.transform()