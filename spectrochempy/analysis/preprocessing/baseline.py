# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the `Baseline` class for baseline corrections.
"""
__dataset_methods__ = ["ab", "abc", "dc", "basc"]

import numpy as np
import scipy.interpolate
import scipy.signal
import traitlets as tr

from spectrochempy.analysis._base import (
    AnalysisConfigurable,
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.core.processors.concatenate import concatenate
from spectrochempy.utils.coordrange import trim_ranges
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.misc import TYPE_FLOAT, TYPE_INTEGER
from spectrochempy.utils.traits import NDDatasetType


@signature_has_configurable_traits
# Note: with this decorator
# Configurable traits are added to the signature as keywords if they are not yet present.
class Baseline(AnalysisConfigurable):
    __doc__ = _docstring.dedent(
        """
    Baseline Correction processor.

    The baseline correction can be applied to 1D datasets consisting in a single row
    with :term:`n_features` or to 2D dataset with shape (:term:`n_observation`,
    :term:`n_features`\ ).

    Several methods are proposed:

    - A general ``'sequential'`` method which can be used for both 1D and 2D datasets.
      with separate fitting of each observation row (spectrum).

    - A ``'multivariate'`` method which can only be applied to 2D datasets. With this
      method the 2D dataset is first dimensionnaly reduced into several principal
      components using the classical :term:`SVD` algorithm.
      Each components is thus fitted before an inverse transform to recover the
      original data with baseline correction.


    For both methods, various interpolation algorithm can be used to estimate the
    baseline.

    In the case of a ``'sequential'`` correction, the interpolation options
    are:

    - ``'abc'`` : A linear baseline is automatically subtracted using the feature limit
      for reference.
    - ``'detrend'`` : remove trend from data. Depending on the ``order`` parameter,
      the detrend can be constant (mean removal), linear (order 1)
      or quadratic (order 2).
    - ``'als'`` :
    - ``'polynomial'`` :
    - ``'pchip'`` :


    # TODO: complete this description

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    detrend : `NDDataset` method equivalent to the ``'Baseline.detrend'`` using
        `scipy.signal.detrend` for linear or constant trend removal.
    abc : `NDDataset` method to automatically suppress a linear baseline correction
        equivalent to the ``'Baseline.abc'`` method.
    """
    )

    #     if not ranges and dataset.meta.regions is not None:
    #         # use the range stored in metadata
    #         ranges = dataset.meta.regions["baseline"]

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    method = tr.CaselessStrEnum(
        ["sequential", "multivariate"],
        default_value="sequential",
        help="Method used for baseline resolution.",
    ).tag(config=True)

    interpolation = tr.CaselessStrEnum(
        ["polynomial", "pchip", "abc", "detrend"],
        default_value="pchip",
        help="Interpolation method.",
    ).tag(config=True)

    order = tr.Union(
        (
            tr.Integer(),
            tr.CaselessStrEnum(["constant", "linear", "quadratic", "cubic"]),
        ),
        default_value=6,
        min=1,
        help="Polynom order to use for polynomial interpolation or detrend.",
    ).tag(config=True)

    n_components = tr.Integer(
        default_value=5,
        min=1,
        help="Number of components to use for the multivariate method "
        "(:term:`n_observation` >= `n_components`).",
    ).tag(config=True)

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

    bp = tr.List(
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
        help="The dataset containing only the sections " "corresponding to _ranges"
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
    @tr.validate("bp")
    def _validate_bp(self, proposal):
        if proposal.value is None:
            return []
        else:
            return sorted(proposal.value)

    @tr.validate("n_components")
    def _validate_n_components(self, proposal):
        # n cannot be higher than the size of s
        npc = proposal.value
        if self._X_is_missing:
            return npc
        return min(npc, self._X.shape[0])

    @tr.validate("order")
    def _validate_order(self, proposal):
        # string must be transformed to int
        order = proposal.value
        if isinstance(order, str):
            order = {"constant": 0, "linear": 1, "quadratic": 2, "cubic": 3}[
                order.lower()
            ]
        return order

    @tr.validate("ranges")
    def _validate_ranges(self, proposal):
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

    @tr.observe("_X", "ranges", "include_limits")
    def _preprocess_as_X_or_ranges_changed(self, change):
        # set X and ranges using the new or current value
        X = None
        if change.name == "_X":
            X = change.new
        elif self._fitted:
            X = self._X

        if self.interpolation not in ["polynomial", "pchip"]:
            # such as detrend, we work on the full data so range is the full feature
            # range.
            self._X_ranges = self._X.copy()
            return

        ranges = change.new if change.name == "ranges" else self.ranges.copy()
        if X is None:
            # not a lot to here
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
        if self.method == "multivariate":
            self.n_components = self.n_components  # fire the validation

    def _fit(self, xbase, ybase):
        # determine the baseline according to the current method and interpolation

        lastcoord = self._X.coordset[self._X.dims[-1]]
        x = lastcoord.data
        M, N = self._X.shape

        # sequential
        # ----------
        if self.method == "sequential":

            if self.interpolation in ["polynomial", "detrend"]:
                polycoef = np.polynomial.polynomial.polyfit(
                    xbase, ybase.T, deg=self.order, rcond=None, full=False
                )
                baseline = np.polynomial.polynomial.polyval(x, polycoef)

            elif self.interpolation == "pchip":
                baseline = np.zeros_like(self._X)
                for i in range(M):
                    interp = scipy.interpolate.PchipInterpolator(xbase, ybase[i])
                    baseline[i] = interp(x)

        elif self.method == "multivariate":
            # SVD of ybase
            U, s, Vt = np.linalg.svd(ybase, full_matrices=False, compute_uv=True)

            # select npc loadings & compute scores
            npc = self.n_components
            baseline_loadings = np.zeros((npc, N))

            Pt = Vt[0:npc]
            T = U[:, 0:npc] @ np.diag(s)[0:npc, 0:npc]

            if self.interpolation == "pchip":
                for i in range(npc):
                    interp = scipy.interpolate.PchipInterpolator(xbase, Pt[i])
                    baseline_loadings[i] = interp(x)

            elif self.interpolation == "polynomial":
                polycoef = np.polynomial.polynomial.polyfit(
                    xbase, Pt.T, deg=self.order, rcond=None, full=False
                )

                baseline_loadings = np.polynomial.polynomial.polyval(x, polycoef)

            baseline = T @ baseline_loadings

        # unsqueeze data (needed to restore masks properly)
        if baseline.ndim == 1:
            baseline = baseline[np.newaxis]

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
        # fire the X and _ranges validation and preprocessing.
        self._X = X

        # _X_ranges has been computed when X and _ranges were set,
        # but we need increasing order of the coordinates
        self._X_ranges.sort(inplace=True, descend=False)

        # _X_ranges is now ready, we can fit. _Xranges contains
        # only the baseline data to fit.
        ybase = self._X_ranges.data  # baseline data
        lastcoord = self._X_ranges.coordset[self._X_ranges.dims[-1]]
        xbase = lastcoord.data  # baseline x-axis

        # Handling breakpoints
        # --------------------
        # inclde the extrama of the x-axis as breakpoints
        bpil = [0, self._X.shape[-1] - 1]
        # breakpoints can be provided as indices or as values.
        # we convert them to indices.
        for bp in self.bp:
            bpil = self._X.x.loc2index(bp) if isinstance(bp, TYPE_FLOAT) else bp
            bpil.append(bpil)
        # sort and remove duplicates
        bpil = sorted(list(set(bpil)))

        # loop on breakpoints pairs
        baseline = np.zeros_like(self._X)
        bpstart = 0
        for bpend in bpil[1:]:
            # fit the baseline on each segment
            xb = xbase[bpstart : bpend + 1]
            yb = ybase[bpstart : bpend + 1]
            baseline[bpstart : bpend + 1] = self._fit(xb, yb)
        self._outfit = [baseline, bpil, ]

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
        return self.X - self.baseline

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
        self._sps = []
        self.ranges = list(trim_ranges(*self.ranges))
        for range in self.ranges:
            range.sort()
            sp = ax.axvspan(range[0], range[1], facecolor="#2ca02c", alpha=0.5)
            self._sps.append(sp)


#     def run(self, *ranges, **kwargs):
#         """
#         Interactive version of the baseline correction.
#
#         Parameters
#         ----------
#         *ranges : a variable number of pair-tuples
#             The regions taken into account for the manual baseline correction.
#         **kwargs
#             See other parameter of method compute.
#         """
#         self._setup(**kwargs)
#         self.sps = []
#
#         # output dataset
#         new = self.corrected
#         origin = self.dataset.copy()
#
#         # we assume that the last dimension if always the dimension to which we want to subtract the baseline.
#         # Swap the axes to be sure to be in this situation
#         axis, dim = new.get_axis(**kwargs, negative_axis=True)
#
#         # swapped = False
#         if axis != -1:
#             new.swapdims(axis, -1, inplace=True)
#             origin.swapdims(axis, -1, inplace=True)
#             # swapped = True
#
#         lastcoord = new.coordset[dim]
#
#         # most of the time we need sorted axis, so let's do it now
#
#         if lastcoord.reversed:
#             new.sort(dim=dim, inplace=True, descend=False)
#             lastcoord = new.coordset[dim]
#
#         x = lastcoord.data
#         self.ranges = [[x[0], x[2]], [x[-3], x[-1]]]
#         self._extendranges(*ranges, **kwargs)
#         self.ranges = ranges = trim_ranges(*self.ranges)
#
#         new = self.compute(*ranges, **kwargs)
#
#         # display
#         datasets = [origin, new]
#         labels = [
#             "Click on left button & Span to set regions. Click on right button on a region to remove it.",
#             "Baseline corrected dataset preview",
#         ]
#         axes = multiplot(
#             datasets,
#             labels,
#             method="stack",
#             sharex=True,
#             nrow=2,
#             ncol=1,
#             figsize=self.figsize,
#             suptitle="INTERACTIVE BASELINE CORRECTION",
#         )
#
#         fig = plt.gcf()
#         fig.canvas.draw()
#
#         ax1 = axes["axe11"]
#         ax2 = axes["axe21"]
#
#         self.show_regions(ax1)
#
#         def show_basecor(ax2):
#
#             corrected = self.compute(*ranges, **kwargs)
#
#             ax2.clear()
#             ax2.set_title(
#                 "Baseline corrected dataset preview", fontweight="bold", fontsize=8
#             )
#             if self.zoompreview > 1:
#                 zb = 1.0  # self.zoompreview
#                 zlim = [corrected.data.min() / zb, corrected.data.max() / zb]
#                 _ = corrected.plot_stack(ax=ax2, colorbar=False, zlim=zlim, clear=False)
#             else:
#                 _ = corrected.plot_stack(ax=ax2, colorbar=False, clear=False)
#
#         show_basecor(ax2)
#
#         def onselect(xmin, xmax):
#             self.ranges.append([xmin, xmax])
#             self.show_regions(ax1)
#             show_basecor(ax2)
#             fig.canvas.draw()
#
#         def onclick(event):
#             if event.button == 3:
#                 for i, r in enumerate(self.ranges):
#                     if r[0] > event.xdata or r[1] < event.xdata:
#                         continue
#                     else:
#                         self.ranges.remove(r)
#                         self.show_regions(ax1)
#                         show_basecor(ax2)
#                         fig.canvas.draw()  # _idle
#
#         _ = fig.canvas.mpl_connect("button_press_event", onclick)
#
#         _ = SpanSelector(
#             ax1,
#             onselect,
#             "horizontal",
#             minspan=5,
#             button=[1],
#             useblit=True,
#             props=dict(alpha=0.5, facecolor="blue"),
#         )
#
#         fig.canvas.draw()
#
#         return
#
#
# def basc(dataset, *ranges, **kwargs):
#     """
#     Compute a baseline correction using the Baseline processor.
#
#     2 methods are proposed :
#
#     * `sequential` (default) = classical polynom fit or spline
#       interpolation with separate fitting of each row (spectrum)
#     * `multivariate` = SVD modeling of baseline, polynomial fit of PC's
#       and calculation of the modelled baseline spectra.
#
#     Parameters
#     ----------
#     dataset : a [NDDataset| instance
#         The dataset where to calculate the baseline.
#     \*ranges : a variable number of pair-tuples
#         The regions taken into account for the manual baseline correction.
#     **kwargs
#         Optional keyword parameters (see Other Parameters).
#
#     Other Parameters
#     ----------------
#     dim : str or int, keyword parameter, optional, default: 'x'.
#         Specify on which dimension to apply the apodization method.
#         If `dim` is specified as an integer
#         it is equivalent  to the usual `axis` numpy parameter.
#     method : str, keyword parameter, optional, default: 'sequential'
#         Correction method among ['multivariate','sequential']
#     interpolation : string, keyword parameter, optional, default='polynomial'
#         Interpolation method for the computation of the baseline,
#         among ['polynomial','pchip']
#     order : int, keyword parameter, optional, default: 6
#         If the correction method polynomial, this give the polynomial order to use.
#     npc : int, keyword parameter, optional, default: 5
#         Number of components to keep for the `multivariate` method
#
#     See Also
#     --------
#     Baseline : Manual baseline corrections.
#     abc : Automatic baseline correction.
#
#     Notes
#     -----
#     For more flexibility and functionality, it is advised to use the Baseline
#     processor instead.
#     """
#     blc = Baseline(dataset)
#     if not ranges and dataset.meta.regions is not None:
#         # use the range stored in metadata
#         ranges = dataset.meta.regions["baseline"]
#     return blc.compute(*ranges, **kwargs)
#
#
# # ======================================================================================
# # abc # TODO: some work to perform on this
# # ======================================================================================
# def abc(dataset, dim=-1, **kwargs):
#     """
#     Automatic baseline correction.
#
#     Various algorithms are provided to calculate the baseline automatically.
#
#     Parameters
#     ----------
#     dataset : a [NDDataset| instance
#         The dataset where to calculate the baseline.
#     dim : str or int, optional
#         The dataset dimentsion where to calculate the baseline. Default is -1.
#     **kwargs
#         Optional keyword parameters (see Other Parameters).
#
#     Returns
#     -------
#     baseline_corrected
#         A baseline corrected dataset.
#     baseline_only
#         Only the baseline (apply must be set to False).
#     baseline_points
#         Points where the baseline is calculated (return_points must be set to True).
#
#     Other Parameters
#     ----------------
#     basetype : string, optional, default: 'linear'
#         See notes - available = linear, basf, ...
#     window : float/int, optional, default is 0.05
#         If float <1 then the corresponding percentage of the axis size is taken as window.
#     nbzone : int, optional, default is 32
#         Number of zones. We will divide the size of the last axis by this number
#         to determine the number of points in each zone (nw).
#     mult : int
#         A multiplicator. determine the number of point for the database calculation (nw*mult<n base points).
#     nstd : int, optional, default is 2 times the standard error
#         Another multiplicator. Multiply the standard error to determine the region in which points are from the
#         baseline.
#     polynom : bool, optional, default is True
#         If True a polynom is computed for the base line, else an interpolation is achieved betwwen points.
#     porder : int, default is 6
#         Order of the polynom to fit on the baseline points
#     return_points : bool, optional, default is False
#         If True, the points abscissa used to determine the baseline are returned.
#     apply : bool, optional, default is True
#         If apply is False, the data are not modified only the baseline is returned.
#     return_pts : bool, optional, default is False
#         If True, the baseline reference points are returned.
#
#     See Also
#     --------
#     Baseline : Manual baseline corrections.
#     basc : Manual baseline correction.
#
#     Notes
#     -----
#     #TODO: description of these algorithms
#     * linear -
#     * basf -
#
#     Examples
#     --------
#     To be done
#     """
#     # # options evaluation
#     # parser = argparse.ArgumentParser(description='BC processing.', usage="""
#     # ab [-h] [--mode {linear,poly, svd}] [--dryrun]
#     #                        [--window WINDOW] [--step STEP] [--nbzone NBZONE]
#     #                        [--mult MULT] [--order ORDER] [--verbose]
#     # """)
#     # # positional arguments
#     # parser.add_argument('--mode', '-mo', default='linear',
#     #                      choices=['linear', 'poly', 'svd'], help="mode of correction")
#     # parser.add_argument('--dryrun', action='store_true', help='dry flag')
#     #
#     # parser.add_argument('--window', '-wi', default=0.05, type=float, help='selected window for linear and svd bc')
#     # parser.add_argument('--step', '-st', default=5, type=int, help='step for svd bc')
#     # parser.add_argument('--nbzone', '-nz', default=32, type=int, help='number of zone for poly')
#     # parser.add_argument('--mult', '-mt', default=4, type=int, help='multiplicator of zone for poly')
#     # parser.add_argument('--order', '-or', default=5, type=int, help='polynom order for poly')
#     #
#     # parser.add_argument('--verbose', action='store_true', help='verbose flag')
#     # args = parser.parse_args(options.split())
#     #
#     # source.history.append('baseline correction mode:%s' % args.mode)
#
#     inplace = kwargs.pop("inplace", False)
#     dryrun = kwargs.pop("dryrun", False)
#
#     # output dataset inplace or not
#     if not inplace or dryrun:  # default
#         new = dataset.copy()
#     else:
#         new = dataset
#
#     axis, dim = new.get_axis(dim, negative_axis=True)
#     swapped = False
#     if axis != -1:
#         new.swapdims(axis, -1, inplace=True)  # must be done in  place
#         swapped = True
#
#     base = _basecor(new.data.real, **kwargs)
#
#     if not dryrun:
#         new.data -= base  # return the corrected spectra
#     else:
#         new.data = base  # return the baseline
#
#     # restore original data order if it was swapped
#     if swapped:
#         new.swapdims(axis, -1, inplace=True)  # must be done inplace
#
#     new.history = "`abc` Baseline correction applied."
#     return new
#
#
# def ab(dataset, dim=-1, **kwargs):
#     """
#     Alias of `abc` .
#     """
#     return abs(dataset, dim, **kwargs)
#
#
# @_units_agnostic_method
# def dc(dataset, **kwargs):
#     """
#     Time domain baseline correction.
#
#     Parameters
#     ----------
#     dataset : nddataset
#         The time domain daatset to be corrected.
#     kwargs : dict, optional
#         Additional parameters.
#
#     Returns
#     -------
#     dc
#         DC corrected array.
#
#     Other Parameters
#     ----------------
#     len : float, optional
#         Proportion in percent of the data at the end of the dataset to take into account. By default, 25%.
#     """
#
#     len = int(kwargs.pop("len", 0.25) * dataset.shape[-1])
#     dc = np.mean(np.atleast_2d(dataset)[..., -len:])
#     dataset -= dc
#
#     return dataset
#
#
# # ======================================================================================
# # private functions
# # ======================================================================================
# def _basecor(data, **kwargs):
#     mode = kwargs.pop("mode", "linear")
#
#     if mode == "linear":
#         return _linearbase(data, **kwargs)
#
#     if mode == "svd":
#         return _svdbase(data, **kwargs)
#
#     if mode == "poly":
#         return _polybase(data, **kwargs)
#     else:
#         raise ValueError(f"`ab` mode = `{mode}`  not known")
#
#
# #
# # _linear mode
# #
# def _linearbase(data, **kwargs):
#     # Apply a linear baseline correction
#     # Very simple and naive procedure that compute a straight baseline from side to the other
#     # (averging on a window given by the window parameters : 5% of the total width on each side by default)
#
#     window = kwargs.pop("window", 0.05)
#
#     if window <= 1.0:
#         # percent
#         window = int(data.shape[-1] * window)
#
#     if len(data.shape) == 1:
#         npts = float(data.shape[-1])
#         a = (data[-window:].mean() - data[:window].mean()) / (npts - 1.0)
#         b = data[:window].mean()
#         baseline = a * np.arange(npts) + b
#
#     else:
#         npts = float(data.shape[-1])
#         a = (data[:, -window:].mean(axis=-1) - data[:, :window].mean(axis=-1)) / (
#             npts - 1.0
#         )
#         b = data[:, :window].mean(axis=-1)
#         baseline = (((np.ones_like(data).T * a).T * np.arange(float(npts))).T + b).T
#
#     return baseline
#
#
# def _planeFit(points):
#     # p, n = planeFit(points)  # copied from https://stackoverflow.com/a/18968498
#     #
#     # Fit an multi-dimensional plane to the points.
#     # Return a point on the plane and the normal.
#     #
#     # Parameters
#     # ----------
#     # points :
#     #
#     # Notes
#     # -----
#     #     Replace the nonlinear optimization with an SVD.
#     #     The following creates the moment of inertia tensor, M, and then
#     #     SVD's it to get the normal to the plane.
#     #     This should be a close approximation to the least-squares fit
#     #     and be much faster and more predictable.
#     #     It returns the point-cloud center and the normal.
#
#     from numpy.linalg import svd
#
#     npts = points.shape[0]
#     points = np.reshape(points, (npts, -1))
#     assert points.shape[0] < points.shape[1]
#     ctr = points.mean(axis=1)
#     x = points - ctr[:, None]
#     M = np.dot(x, x.T)
#     return ctr, svd(M)[0][:, -1]
#
#
# def _svdbase(data, args=None, retw=False):
#     # Apply a planar baseline correction to 2D data
#     import pandas as pd  # TODO: suppress this need
#
#     if not args:
#         window = 0.05
#         step = 5
#     else:
#         window = args.window
#         step = args.step
#
#     if window <= 1.0:
#         # percent
#         window = int(data.shape[-1] * window)
#
#     data = pd.DataFrame(
#         data
#     )  # TODO: facilitate the manipulation (but to think about further)
#     a = pd.concat([data.iloc[:window], data.iloc[-window:]])
#     b = pd.concat(
#         [data.iloc[window:-window, :window], data.iloc[window:-window, -window:]],
#         axis=1,
#     )
#     bs = pd.concat([a, b])
#     bs = bs.stack()
#     bs.sort()
#     x = []
#     y = []
#     z = []
#     for item in bs.index[::step]:
#         x.append(item[0])
#         y.append(item[1])
#         z.append(bs[item].real)
#
#     norm = np.max(np.abs(z))
#     z = np.array(z)
#     z = z / norm
#     XYZ = np.array((x, y, z))
#     p, n = _planeFit(XYZ)
#     d = np.dot(p, n)
#
#     col = data.columns
#     row = data.index
#     X, Y = np.meshgrid(col, row)
#     Z = -norm * (n[0] * X + n[1] * Y - d) / n[2]
#
#     if retw:
#         return Z, None  # TODO: return something
#     return Z
#
#
# def _polybase(data, **kwargs):
#     # Automatic baseline correction
#
#     if data.ndim == 1:
#         dat = np.array(
#             [
#                 data,
#             ]
#         )
#
#     nbzone = kwargs.pop("nbzone", 64)
#     mult = kwargs.pop("mult", 4)
#     order = kwargs.pop("order", 6)
#
#     npts = data.shape[-1]
#     w = np.arange(npts)
#
#     baseline = np.ma.masked_array(dat, mask=True)
#     sigma = 1.0e6
#     nw = int(npts / nbzone)
#
#     # print (nw)
#     # unmask extremities of the baseline
#     baseline[:, :nw].mask = False
#     baseline[:, -nw:].mask = False
#
#     for j in range(nbzone):
#         s = dat[:, nw * j : min(nw * (j + 1), npts + 1)]
#         sigma = min(s.std(), sigma)
#
#     nw = nw * 2  # bigger window
#     nw2 = int(nw / 2)
#
#     found = False
#     nb = 0
#     nstd = 2.0
#     while (not found) or (nb < nw * mult):
#         nb = 0
#         for i in range(nw2, npts - nw2 + 1, 1):
#             s1 = dat[:, max(i - 1 - nw2, 0) : min(i - 1 + nw2, npts + 1)]
#             s2 = dat[:, max(i - nw2, 0) : min(i + nw2, npts + 1)]
#             s3 = dat[:, max(i + 1 - nw2, 0) : min(i + 1 + nw2, npts + 1)]
#             mi1, mi2, mi3 = s1.min(), s2.min(), s3.min()
#             ma1, ma2, ma3 = s1.max(), s2.max(), s3.max()
#
#             if (
#                 abs(ma1 - mi1) < float(nstd) * sigma
#                 and abs(ma2 - mi2) < float(nstd) * sigma
#                 and abs(ma3 - mi3) < float(nstd) * sigma
#             ):
#                 found = True
#                 nb += 1
#                 baseline[:1, i].mask = False  # baseline points
#
#         # increase nstd
#         nstd = nstd * 1.1
#     debug_("basf optimized nstd: %.2F mult: %.2f" % (nstd, mult))
#
#     wm = np.array(list(zip(*np.argwhere(~baseline[:1].mask)))[1])
#     bm = baseline[:, wm]
#     if data.ndim > 1:
#         bm = smooth(bm.T, max(int(dat.shape[0] / 10), 3)).T
#     bm = smooth(bm, max(int(dat.shape[-1] / 10), 3))
#
#     # if not polynom:
#     #    sr = pchip(wm, bm.real)
#     #    si = pchip(wm, bm.imag)
#     #    baseline = sr(w) + si(w) * 1.0j
#     #    baseline = smooth(baseline, window_len=int(nw / 4))
#     # else:
#     # fit a polynom
#     pf = np.polyfit(wm, bm.T, order).T
#     for i, row in enumerate(pf[:]):
#         poly = np.poly1d(row)
#         baseline[i] = poly(w)
#
#     if data.ndim == 1:
#         baseline = baseline[0]
#
#     return baseline
