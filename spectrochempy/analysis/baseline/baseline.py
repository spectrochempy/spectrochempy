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
from scipy.spatial import ConvexHull

from spectrochempy.analysis._base._analysisbase import AnalysisConfigurable
from spectrochempy.analysis.baseline.baselineutils import lls, lls_inv
from spectrochempy.application import info_, warning_
from spectrochempy.processing.transformation.concatenate import concatenate
from spectrochempy.utils.coordrange import trim_ranges
from spectrochempy.utils.decorators import (
    _wrap_ndarray_output_to_nddataset,
    deprecated,
    signature_has_configurable_traits,
)
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.misc import TYPE_FLOAT, TYPE_INTEGER
from spectrochempy.utils.plots import NBlue, NGreen, NRed
from spectrochempy.utils.traits import NDDatasetType

__all__ = [
    "Baseline",
    "get_baseline",
    "basc",
    "detrend",
    "asls",
    "snip",
    "rubberband",
    "lls",
    "lls_inv",
]
__configurables__ = ["Baseline"]
__dataset_methods__ = ["get_baseline", "basc", "detrend", "asls", "snip", "rubberband"]

_common_see_also = """
See Also
--------
Baseline : Manual baseline correction processor.
get_baseline : Compuute a baseline using the `Baseline` class.
basc : Make a baseline correction using the `Baseline` class.
asls : Perform an Asymmetric Least Squares Smoothing baseline correction.
snip : Perform a Simple Non-Iterative Peak (SNIP) detection algorithm.
rubberband : Perform a Rubberband baseline correction.
autosub: Perform an automatic subtraction of reference.
detrend : Remove polynomial trend along a dimension from dataset.
"""

_docstring.get_sections(
    _docstring.dedent(_common_see_also),
    base="Baseline",
    sections=["See Also"],
)
_docstring.delete_params("Baseline.see_also", "Baseline")
_docstring.delete_params("Baseline.see_also", "get_baseline")
_docstring.delete_params("Baseline.see_also", "basc")
_docstring.delete_params("Baseline.see_also", "asls")
_docstring.delete_params("Baseline.see_also", "snip")
_docstring.delete_params("Baseline.see_also", "autosub")
_docstring.delete_params("Baseline.see_also", "detrend")
_docstring.delete_params("Baseline.see_also", "rubberband")


# ======================================================================================
# Baseline class processor
# ======================================================================================
@signature_has_configurable_traits
# Note: with this decorator
# Configurable traits are added to the signature as keywords
# if they are not yet present.
class Baseline(AnalysisConfigurable):
    __doc__ = _docstring.dedent(
        r"""
    Baseline Correction processor.

    The baseline correction can be applied to 1D datasets consisting in a single row
    with :term:`n_features` or to a 2D dataset with shape (:term:`n_observations`\ ,
    :term:`n_features`\ ).

    When dealing with 2D datasets, the baseline correction can be applied either
    sequentially (default) or using a multivariate approach (parameter
    `multivariate`set to `True).

    - The ``'sequential'`` approach which can be used for both 1D and 2D datasets
      consists in fitting the baseline sequentially for each observation row (spectrum).

    - The ``'multivariate'`` approach can only be applied to 2D datasets (at least 3
      observations).
      The 2D dataset is first dimensionally reduced into several principal
      components using a conventional Singular Value Decomposition :term:`SVD` or a
      non-negative matrix factorization (`NMF`\ ).
      Each component is then fitted before an inverse transform performed to recover
      the baseline correction.

    In both approaches, various models can be used to estimate the
    baseline.

    - ``'detrend'`` : remove trends from data. Depending on the ``order`` parameter,
      the detrend can be constant (mean removal), linear (order=1), quadratic (order=2)
      or `cubic`(order=3).
    - ``'asls'`` : Asymmetric Least Squares Smoothing baseline correction. This method
      is based on the work of Eilers and Boelens (:cite:`eilers:2005`\ ).
    - ``'snip'`` : Simple Non-Iterative Peak (SNIP) detection algorithm
      (:cite:`ryan:1988`\ ).
    - ``'rubberband'`` : Rubberband baseline correction.
    - ``'polynomial'`` : Fit a nth-degree polynomial to the data. The order of
      the polynomial is defined by the ``order`` parameter. The baseline is then
      obtained by evaluating the polynomial at each feature defined in predefined
      `ranges`\ .

    By default, `ranges` is set to the feature limits (i.e. `ranges=[features[0],
    features[-1]]`\ )

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
    multivariate = tr.Union(
        (tr.Bool(), tr.CaselessStrEnum(["nmf", "svd"])),
        default_value=False,
        help="For 2D datasets, if `True` or if multivariate='svd' or 'nmf' , a "
        "multivariate method is used to fit a "
        "baseline on the principal components determined using a SVD decomposition "
        " if `multivariate='svd'`\ or `True`, or a NMF factorization if "
        "`multivariate='nmf'`\ ,"
        "followed by an inverse-transform to retrieve the baseline corrected "
        "dataset. If `False` , a sequential method is used which consists in fitting a "
        "baseline on each row (observations) of the dataset.",
    ).tag(config=True)

    model = tr.CaselessStrEnum(
        ["polynomial", "detrend", "asls", "snip", "rubberband"],
        default_value="polynomial",
        help="""The model used to determine the baseline.

* 'polynomial': the baseline correction is determined by a nth-degree polynomial fitted
  on the data belonging to the selected `ranges`. The `order` parameter to determine the
  degree of the polynomial.
* 'detrend': removes a constant, linear or polynomial trend to the data. The order of
  the trend is determined by the `order` parameter.
* 'asls': the baseline is determined by an asymmetric least square algorithm.
* 'snip': the baseline is determined by a simple non-iterative peak detection
  algorithm.
* 'rubberband': the baseline is determined by a rubberband algorithm.
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
            tr.CaselessStrEnum(["constant", "linear", "quadratic", "cubic", "pchip"]),
        ),
        default_value=1,
        help="""Polynom order to use for polynomial/pchip interpolation or detrend.

* If an integer is provided, it is the order of the polynom to fit, i.e. 1 for linear,
* If a string if provided among  'constant', 'linear', 'quadratic' and 'cubic',
  it is equivalent to order O (constant) to 3 (cubic).
* If a string equal to `pchip` is provided, the polynomial interpolation is replaced
  by a piecewise cubic hermite interpolation
  (see `scipy.interpolate.PchipInterpolator`\ """,
    ).tag(config=True, min=1)

    lamb = tr.Float(
        default_value=1e5,
        help="The smoothness parameter for the AsLS method. Larger values make the "
        "baseline stiffer. Values should be in the range (0, 1e9).",
    ).tag(config=True)

    asymmetry = tr.Float(
        default_value=0.05,
        help="The asymmetry parameter for the AsLS method. "
        "It is typically between 0.001 "
        "and 0.1. 0.001 gives almost the same fit as the unconstrained least squares",
    ).tag(config=True, min=0.001)

    snip_width = tr.Integer(
        help="The width of the window used to determine the baseline using the SNIP "
        "algorithm."
    ).tag(config=True, min=0)

    tol = tr.Float(
        default_value=1e-3,
        help="The tolerance parameter for the AsLS method. Smaller values make the "
        "fitting better but potentially increases the number of iterations and the "
        "running time. Values should be in the range (0, 1).",
    ).tag(config=True, min=0, max=1)

    max_iter = tr.Integer(50, help="Maximum number of :term:`AsLS` iteration.").tag(
        config=True, min=1
    )
    n_components = tr.Integer(
        default_value=5,
        help="Number of components to use for the multivariate method "
        "(:term:`n_observations` >= `n_components`).",
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
        if isinstance(order, str) and order != "pchip":
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

        # clean the result (reorder and suppress overlaps)
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

        if self.model not in [
            "polynomial",
        ]:
            # such as detrend, or asls we work on the full data so range is the
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
            ranges += [[x[0], x[2]], [x[-3], x[-1]]]

        if self.breakpoints:
            # we should also include breakpoints in the ranges
            inc = lastcoord.spacing
            inc = inc.m if hasattr(inc, "magnitude") else inc
            ranges += [[bp - inc, bp + inc] for bp in self.breakpoints]

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

    def _fit(self, xbase, ybase, Xpart):
        # core _fit method:
        # calculate the baseline according to the current approach and model

        # get the last coordinate of the dataset
        lastcoord = Xpart.coordset[Xpart.dims[-1]]
        x = lastcoord.data

        # get the number of observations and features
        M, N = Xpart.shape if Xpart.ndim == 2 else (1, Xpart.shape[0])

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
            M = self.n_components
            _store = np.zeros((M, N))
            if self.multivariate == "nmf":
                # mutivariate NMF factorization
                from sklearn.decomposition import NMF

                nmf = NMF(n_components=M, init="random", random_state=0)
                T = nmf.fit_transform(Y)
                Y = nmf.components_
            else:
                # multivariate SVD method
                U, s, Vt = np.linalg.svd(Y, full_matrices=False, compute_uv=True)
                T = U[:, 0:M] @ np.diag(s)[0:M, 0:M]
                Y = Vt[0:M]

        # -----------------------------------------
        # Polynomial interpolation, detrend
        # -----------------------------------------
        if self.model in ["polynomial", "detrend"] and self.order != "pchip":
            # polynomial interpolation or detrend process
            # using parameter `order` and predetermined ranges
            polycoef = np.polynomial.polynomial.polyfit(
                xbase, Y.T, deg=self.order, rcond=None, full=False
            )
            _store = np.polynomial.polynomial.polyval(x, polycoef)

        # -------------------------------------------------------------------------
        # PChip interpolation (piecewise cubic hermite interpolation) using ranges
        # -------------------------------------------------------------------------
        elif self.model == "polynomial" and self.order == "pchip":
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
            _store = Y[:, ::-1] if self._X.coord(1).is_descendant else Y

        # ------------------------
        # AsLS baseline correction
        # ------------------------
        # see
        # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
        elif self.model == "asls":
            # AsLS fitted baseline
            # For now, this doesn't work with masked data
            lamb = self.lamb
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
                    C = W + lamb * D.dot(D.transpose())
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

        elif self.model == "rubberband":
            # Rubberband baseline correction
            # (do not work with multivariate approach)
            # based on a solution found here:
            # https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
            x = np.arange(N)
            for i in range(M):
                y = Y[i].squeeze()
                # Find the convex hull
                v = ConvexHull(np.array(np.column_stack((x, y)))).vertices
                # Array v contains indices of the vertex points,
                # arranged in the counterclockwise direction,
                # e.g. [892, 125, 93, 0, 4, 89, 701, 1023].
                # We have to extract part where v is ascending, e.g. 0–1023.
                # Rotate convex hull vertices until they start from the lowest one
                v = np.roll(v, -v.argmin())
                # Leave only the ascending part
                v = v[: v.argmax() + 1]
                # Create baseline using linear interpolation between vertices
                _store[i] = np.interp(x, x[v], y[v])

        # inverse transform to get the baseline in the original data space
        # this depends on the approach used (multivariate or not)
        if self.multivariate:
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

        if (
            self.model == "asls"
        ):  # AsLS doesn't work for now with masked data (see _fit)
            # so we will remove the mask and restore it after the fit
            self.Xmasked = X.copy()
            X.remove_masks()

        # fire the X and _ranges validation and preprocessing.
        self._X = X

        # get the last axis coordinates
        Xx = self._X.coordset[self._X.dims[-1]]

        # _X_ranges has been computed when X and _ranges were set,
        # but we need increasing order of the coordinates
        self._X_ranges.sort(inplace=True, descend=False)

        # to simplify further operation we also sort the self._X data
        descend = Xx.is_descendant
        self._X.sort(inplace=True, descend=False)
        Xx = self._X.coordset[self._X.dims[-1]]  # get it again after sorting

        # _X_ranges is now ready, we can fit. _Xranges contains
        # only the baseline data to fit
        ybase = self._X_ranges.data  # baseline data
        lastcoord = self._X_ranges.coordset[self._X_ranges.dims[-1]]
        xbase = lastcoord.data  # baseline x-axis

        # Handling breakpoints
        # --------------------
        # include the extrema of the x-axis as breakpoints
        bplist = [Xx.data[0], Xx.data[-1]]
        # breakpoints can be provided as indices or as values.
        # we convert them to values
        for bp in self.breakpoints:
            bp = Xx.data[bp] if not isinstance(bp, TYPE_FLOAT) else bp
            bplist.append(bp)
        # sort and remove duplicates
        bplist = sorted(list(set(bplist)))

        # loop on breakpoints pairs
        baseline = np.zeros_like(self._X.data)
        istart = lastcoord.loc2index(bplist[0])
        ixstart = Xx.loc2index(bplist[0])
        for end in bplist[1:]:
            iend = lastcoord.loc2index(end)
            ixend = Xx.loc2index(end)
            # fit the baseline on each segment
            xb = xbase[istart : iend + 1]
            yb = ybase[..., istart : iend + 1]
            Xpart = self._X[..., ixstart : ixend + 1]
            baseline[..., ixstart : ixend + 1] = self._fit(xb, yb, Xpart)
            istart = iend + 1
            ixstart = ixend + 1

        # sort back o the original order
        if descend:
            baseline = baseline[..., ::-1]
            self._X.sort(inplace=True, descend=True)

        if self.model == "asls":  # restore the mask
            self._X._mask = self.Xmasked.mask

        self._outfit = (baseline, bplist)  # store the result

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
        if self.model == "asls" and hasattr(self, "Xmasked"):
            corrected = self.Xmasked - self.baseline
        else:
            corrected = self.X - self.baseline
        return corrected

    @property
    def corrected(self):
        """Dataset with baseline removed."""
        return self.transform()

    @_docstring.dedent
    def params(self, default=False):
        """
        %(MetaConfigurable.parameters_doc)s
        """
        d = super().params(default)
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
        # if not self._fitted:
        #    raise NotFittedError
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

    # ----------------------------------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def plot(self, **kwargs):
        r"""
        Plot the original, baseline and corrected dataset.

        Parameters
        ----------
        %(kwargs)s

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib subplot axe.

        Other Parameters
        ----------------
        colors : `tuple` or `~numpy.ndarray` of 3 colors, optional
            Colors for original , baseline and corrected data.
            in the case of 2D, The default colormap is used for the original data.
            By default, the three colors are :const:`NBlue` , :const:`NGreen`
            and :const:`NRed`  (which are colorblind friendly).
        offset : `float`, optional, default: `None`
            Specify the separation (in percent) between the
            original and corrected data.
        nb_traces : `int` or ``'all'``\ , optional
            Number of lines to display. Default is ``'all'``\ .
        **others : Other keywords parameters
            Parameters passed to the internal `plot` method of the datasets.
        """
        colX, colXhat, colRes = kwargs.pop("colors", [NBlue, NGreen, NRed])

        X = self.X  # we need to use self.X here not self._X because the mask
        # are restored automatically
        Xc = self.transform()
        bas = self.baseline

        if X._squeeze_ndim == 1:
            X = X.squeeze()
            Xc = Xc.squeeze()
            bas = bas.squeeze()

        # Number of traces to keep
        nb_traces = kwargs.pop("nb_traces", "all")
        if X.ndim == 2 and nb_traces != "all":
            inc = int(X.shape[0] / nb_traces)
            X = X[::inc]
            Xc = Xc[::inc]
            bas = bas[::inc]

        # separation between traces
        offset = kwargs.pop("offset", None)
        if offset is None:
            offset = 0
        ma = max(X.max(), Xc.max())
        mao = ma * offset / 100
        _ = (X - X.min()).plot(color=colX, **kwargs)
        _ = (Xc - Xc.min() - mao).plot(
            clear=False, ls="dashed", cmap=None, color=colXhat
        )
        ax = (bas - X.min()).plot(clear=False, cmap=None, color=colRes)
        ax.autoscale(enable=True, axis="y")
        ax.set_title(f"{self.name} plot")
        ax.yaxis.set_visible(False)
        return ax


# ======================================================================================
# API / NDDataset functions
# ======================================================================================
# Instead of using directly the Baseline class, we provide here some functions
# which are eventually more user-friendly and which can be used directly on NDDataset or
# called from the API.


@_docstring.dedent
def get_baseline(dataset, *ranges, **kwargs):
    r"""
    Compute a baseline using the Baseline class processor.

    If no ranges is provided, the features limits are used.
    See `Baseline` for detailed information on the parameters.

    Parameters
    ----------
    dataset : a `NDDataset` instance
        The dataset where to calculate the baseline.
    *ranges : a variable number of pair-tuples
        The regions taken into account for the manual baseline correction.
    **kwargs
        Optional keyword parameters (see `Baseline` Parameters for a detailed
        information).

    Returns
    -------
    `NDDataset`
        The computed baseline

    See Also
    --------
    %(Baseline.see_also.no_get_baseline)s

    Notes
    -----
    For more flexibility and functionality, it is advised to use the Baseline class
    processor instead.
    """

    blc = Baseline()
    # by default, model is 'polynomial' and order is 1.
    # kwargs can overwrite these default values
    for key in kwargs:
        setattr(blc, key, kwargs[key])

    # if model is 'polynomial' and no ranges is provided, we use the features limits
    # and order='linear'
    if blc.model == "polynomial":
        if not ranges and blc.order != 1:
            warning_(
                f"As no ranges was provided, baseline() uses the features limit "
                f"with order='linear'. Provided order={blc.order} is ignored"
            )
            blc.order = "linear"
        blc.ranges = ranges

    blc.fit(dataset)
    return blc.baseline


@_docstring.dedent
def basc(dataset, *ranges, **kwargs):
    r"""
    Compute a baseline corrected dataset using the Baseline class processor.

    If no ranges is provided, the features limits are used.
    See `Baseline` for detailed information on the parameters.

    Parameters
    ----------
    dataset : a `NDDataset` instance
        The dataset where to correct the baseline.
    *ranges : a variable number of pair-tuples
        The regions taken into account for the manual baseline correction.
    **kwargs
        Optional keyword parameters (see `Baseline` Parameters for a detailed
        information).

    Returns
    -------
    `NDDataset`
        The computed baseline corrected dataset

    See Also
    --------
    %(Baseline.see_also.no_basc)s

    Notes
    -----
    For more flexibility and functionality, it is advised to use the Baseline class
    processor instead.
    """
    return dataset - get_baseline(dataset, *ranges, **kwargs)


@_docstring.dedent
def detrend(dataset, order="linear", breakpoints=[], **kwargs):
    r"""
    Remove polynomial trend along a dimension from dataset.

    Depending on the ``order``parameter, `detrend` removes the best-fit polynomial line
    (in the least squares sense) from the data and returns the remaining data.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    order : non-negative `int` or a `str` among ['constant', 'linear', 'quadratic', 'cubic'], optional, default:'linear'
        The order of the polynomial trend.

        * If ``order=0`` or ``'constant'`` , the mean of data is subtracted to remove
          a shift trend.
        * If ``order=1`` or ``'linear'`` (default), the best straight-fit line is
          subtracted from data to remove a linear trend (drift).
        * If order=2 or ``order=quadratic`` ,  the best fitted nth-degree polynomial
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
def asls(dataset, lamb=1e5, asymmetry=0.05, tol=1e-3, max_iter=50):
    """
    Asymmetric Least Squares Smoothing baseline correction.

    This method is based on the work of Eilers and Boelens (:cite:`eilers:2005`\ ).

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    lamb : `float`, optional, default:1e5
        The smoothness parameter for the AsLS method. Larger values make the
        baseline stiffer. Values should be in the range (0, 1e9)
    asymmetry : `float`, optional, default:0.05,
        The asymmetry parameter for the AsLS method. It is typically between 0.001
        and 0.1. 0.001 gives almost the same fit as the unconstrained least squares.
    tol = `float`, optional, default:1e-3
        The tolerance parameter for the AsLS method. Smaller values make the fitting better but potentially increases the number of iterations and the running time. Values should be in the range (0, 1).
    max_iter = `int`, optional, default:50
        Maximum number of :term:`AsLS` iteration.

    Returns
    -------
    `NDDataset`
        The baseline corrected dataset.

    See Also
    --------
    %(Baseline.see_also.no_asls)s
    """
    blc = Baseline()
    blc.model = "asls"
    blc.lamb = lamb
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
def rubberband(dataset):
    """
    Rubberband baseline correction.

    The algorithm is faster than the AsLS method, but it is less controllable as there
    is no parameter to tune.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.

    Returns
    -------
    `NDDataset`
        The baseline corrected dataset.

    See Also
    --------
    %(Baseline.see_also.no_rubberband)s
    """
    blc = Baseline()
    blc.model = "rubberband"
    blc.fit(dataset)

    return blc.transform()
