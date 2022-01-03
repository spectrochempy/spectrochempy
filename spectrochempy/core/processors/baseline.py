# -*- coding: utf-8 -*-

# =============================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# =============================================================================
"""
This module implements the `BaselineCorrection` class for baseline corrections.
"""
__all__ = ["BaselineCorrection", "ab", "abc", "dc", "basc"]

__dataset_methods__ = ["ab", "abc", "dc", "basc"]

import numpy as np
import scipy.interpolate
from traitlets import Int, Instance, HasTraits, Float, Unicode, Tuple, List
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

from spectrochempy.core.dataset.coordrange import trim_ranges
from spectrochempy.core.plotters.multiplot import multiplot
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import TYPE_INTEGER, TYPE_FLOAT
from spectrochempy.core import debug_, warning_
from spectrochempy.core.processors.utils import _units_agnostic_method
from .smooth import smooth


class BaselineCorrection(HasTraits):
    """
    Baseline Correction processor.

    2 methods are proposed :

    * ``sequential`` (default) = classical polynom fit or spline
      interpolation with separate fitting of each row (spectrum)
    * ``multivariate`` = SVD modeling of baseline, polynomial fit of PC's
      and calculation of the modelled baseline spectra.

    Interactive mode is proposed using the interactive function : :meth:`run`.

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to be transformed.

    See Also
    --------
    abc : Automatic baseline correction.

    Examples
    --------
    .. plot::
        :include-source:

        from spectrochempy import *
        nd = NDDataset.read_omnic(os.path.join('irdata',
                                    'nh4y-activation.spg'))
        ndp = nd[:, 1291.0:5999.0]
        bc = BaselineCorrection(ndp)
        ranges=[[5996., 5998.], [1290., 1300.],
                [2205., 2301.], [5380., 5979.],
                [3736., 5125.]]
        span = bc.compute(*ranges,method='multivariate',
                          interpolation='pchip', npc=8)
        _ = bc.corrected.plot_stack()
        show()
    """

    dataset = Instance(NDDataset)
    corrected = Instance(NDDataset)
    method = Unicode("sequential")
    interpolation = Unicode("polynomial")
    axis = Int(-1)
    dim = Unicode("")
    order = Int(1, min=1, allow_none=True)
    npc = Int(5, min=1, allow_none=True)
    zoompreview = Float(1.0)
    figsize = Tuple((7, 5))
    sps = List()

    # ..........................................................................
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset
        self.corrected = self.dataset.copy()
        if args or kwargs:
            warning_(
                "DEPRECATION WARNING: Pass all arguments such range, and method definition in the "
                "``compute`` method, not during the initialisation of the BaselineCorrection instance.\n"
                "Here they are ignored."
            )

    # ..........................................................................
    def _extendranges(self, *ranges, **kwargs):
        if not ranges:
            # look in the kwargs
            ranges = kwargs.pop("ranges", ())
        if isinstance(ranges, tuple) and len(ranges) == 1:
            ranges = ranges[0]  # probably passed with no start to the compute function
        if not isinstance(ranges, (list, tuple)):
            ranges = list(ranges)
        if not ranges:
            return

        if len(ranges) == 2:
            if isinstance(ranges[0], TYPE_INTEGER + TYPE_FLOAT) and isinstance(
                ranges[1], TYPE_INTEGER + TYPE_FLOAT
            ):
                # a pair a values, we intepret this as a single range
                ranges = [[ranges[0], ranges[1]]]
        # find the single values
        for item in ranges:
            if isinstance(item, TYPE_INTEGER + TYPE_FLOAT):
                # a single numerical value: intepret this as a single range
                item = [item, item]
            self.ranges.append(item)

    # ..........................................................................
    def _setup(self, **kwargs):

        self.method = kwargs.get("method", self.method)
        self.interpolation = kwargs.get("interpolation", self.interpolation)
        if self.interpolation == "polynomial":
            self.order = int(kwargs.get("order", self.order))
        if self.method == "multivariate":
            self.npc = int(kwargs.get("npc", self.npc))
        self.zoompreview = kwargs.get("zoompreview", self.zoompreview)
        self.figsize = kwargs.get("figsize", self.figsize)

    # ..........................................................................
    def __call__(self, *ranges, **kwargs):

        return self.compute(*ranges, **kwargs)

    # ..........................................................................
    def compute(self, *ranges, **kwargs):
        """
        Base function for dataset baseline correction.

        Parameters
        ----------
        *ranges : a variable number of pair-tuples
            The regions taken into account for the manual baseline correction.
        **kwargs : dict
            See other parameters.

        Other Parameters
        ----------------
        dim : str or int, keyword parameter, optional, default='x'.
            Specify on which dimension to apply the apodization method.
            If `dim` is specified as an integer
            it is equivalent  to the usual `axis` numpy parameter.
        method : str, keyword parameter, optional, default='sequential'
            Correction method among ['multivariate','sequential']
        interpolation : string, keyword parameter, optional, default='polynomial'
            Interpolation method for the computation of the baseline,
            among ['polynomial','pchip']
        order : int, keyword parameter, optional, default=1
            If the correction method polynomial,
            this give the polynomial order to use.
        npc : int, keyword parameter, optional, default=5
            Number of components to keep for the ``multivariate`` method
        zoompreview : float, keyword parameter, optional, default=1.0
            The zoom factor for the preview in interactive mode
        figsize : tuple, keyword parameter, optional, default=(8, 6)
            Size of the figure to display in inch
        """

        self._setup(**kwargs)

        # output dataset
        new = self.corrected

        # we assume that the last dimension
        # if always the dimension to which we want to subtract the baseline.
        # Swap the axes to be sure to be in this situation
        axis, dim = new.get_axis(**kwargs, negative_axis=True)

        swaped = False
        if axis != -1:
            new.swapdims(axis, -1, inplace=True)
            swaped = True

        lastcoord = new.coordset[dim]

        # most of the time we need sorted axis, so let's do it now
        is_descendant = False
        if lastcoord.descendant:
            new.sort(dim=dim, inplace=True, descend=False)
            is_descendant = True
            lastcoord = new.coordset[dim]

        x = lastcoord.data
        self.ranges = [[x[0], x[2]], [x[-3], x[-1]]]
        self._extendranges(*ranges, **kwargs)
        self.ranges = ranges = trim_ranges(*self.ranges)

        baseline = np.zeros_like(new)

        # Extract: Sbase: the matrix of data corresponding to ranges
        #          xbase: the xaxis values corresponding to ranges

        s = []
        for pair in ranges:
            # determine the slices

            sl = slice(*pair)
            sect = new[..., sl]
            if sect is None:
                continue

            s.append(sect)

        sbase = NDDataset.concatenate(s, axis=-1)
        # TODO: probably we could use masked data instead of concatenating - could be faster
        xbase = sbase.coordset(dim)

        if self.method == "sequential":

            if self.interpolation == "polynomial":
                # # bad fit when NaN values => are replaced by 0      # NO reason we have Nan -> suppressed
                # if np.any(np.isnan(sbase)):
                #     sbase[np.isnan(sbase)] = 0

                polycoef = np.polynomial.polynomial.polyfit(
                    xbase.data, sbase.data.T, deg=self.order, rcond=None, full=False
                )
                baseline = np.polynomial.polynomial.polyval(x, polycoef)

            elif self.interpolation == "pchip":
                for i in range(new.shape[0]):
                    interp = scipy.interpolate.PchipInterpolator(
                        xbase.data, sbase.data[i]
                    )
                    baseline[i] = interp(x)

        elif self.method == "multivariate":

            # SVD of Sbase
            U, s, Vt = np.linalg.svd(sbase.data, full_matrices=False, compute_uv=True)

            # npc cannot be higher than the size of s
            npc = min(self.npc, s.shape[0])

            # select npc loadings & compute scores
            Pt = Vt[0:npc]
            T = np.dot(U[:, 0:npc], np.diag(s)[0:npc, 0:npc])

            baseline_loadings = np.zeros((npc, new.shape[-1]))

            if self.interpolation == "pchip":
                for i in range(npc):
                    interp = scipy.interpolate.PchipInterpolator(xbase.data, Pt[i])
                    baseline_loadings[i] = interp(x)

            elif self.interpolation == "polynomial":
                polycoef = np.polynomial.polynomial.polyfit(
                    xbase.data, Pt.T, deg=self.order, rcond=None, full=False
                )

                baseline_loadings = np.polynomial.polynomial.polyval(x, polycoef)

            baseline = np.dot(T, baseline_loadings)

        new.data = new.data - baseline

        # eventually sort back to the original order
        if is_descendant:
            new.sort(axis=-1, inplace=True, descend=True)

        new.history = str(new.modified) + ": " + "Baseline correction." + " Method: "
        if self.method == "Multivariate":
            new.history = "Multivariate (" + str(self.npc) + " PCs)."
        else:
            new.history = "Sequential."

        if self.interpolation == "polynomial":
            new.history = "Interpolation: Polynomial, order=" + str(self.order) + ".\n"
        else:
            new.history = "Interpolation: Pchip. \n"

        if swaped:
            new = new.swapdims(axis, -1)

        self.corrected = new
        return new

    # ..........................................................................
    def show_regions(self, ax):
        if self.sps:
            for sp in self.sps:
                sp.remove()
        self.sps = []
        self.ranges = list(trim_ranges(*self.ranges))
        for x in self.ranges:
            x.sort()
            sp = ax.axvspan(x[0], x[1], facecolor="#2ca02c", alpha=0.5)
            self.sps.append(sp)

    # ..........................................................................
    def run(self, *ranges, **kwargs):
        """
        Interactive version of the baseline correction.

        Parameters
        ----------
        *ranges : a variable number of pair-tuples
            The regions taken into account for the manual baseline correction.
        **kwargs : dict
            See other parameter of method compute.
        """
        self._setup(**kwargs)
        self.sps = []

        # output dataset
        new = self.corrected
        origin = self.dataset.copy()

        # we assume that the last dimension if always the dimension to which we want to subtract the baseline.
        # Swap the axes to be sure to be in this situation
        axis, dim = new.get_axis(**kwargs, negative_axis=True)

        # swaped = False
        if axis != -1:
            new.swapdims(axis, -1, inplace=True)
            origin.swapdims(axis, -1, inplace=True)
            # swaped = True

        lastcoord = new.coordset[dim]

        # most of the time we need sorted axis, so let's do it now

        if lastcoord.reversed:
            new.sort(dim=dim, inplace=True, descend=False)
            lastcoord = new.coordset[dim]

        x = lastcoord.data
        self.ranges = [[x[0], x[2]], [x[-3], x[-1]]]
        self._extendranges(*ranges, **kwargs)
        self.ranges = ranges = trim_ranges(*self.ranges)

        new = self.compute(*ranges, **kwargs)

        # display
        datasets = [origin, new]
        labels = [
            "Click on left button & Span to set regions. Click on right button on a region to remove it.",
            "Baseline corrected dataset preview",
        ]
        axes = multiplot(
            datasets,
            labels,
            method="stack",
            sharex=True,
            nrow=2,
            ncol=1,
            figsize=self.figsize,
            suptitle="INTERACTIVE BASELINE CORRECTION",
        )

        fig = plt.gcf()
        fig.canvas.draw()

        ax1 = axes["axe11"]
        ax2 = axes["axe21"]

        self.show_regions(ax1)

        def show_basecor(ax2):

            corrected = self.compute(*ranges, **kwargs)

            ax2.clear()
            ax2.set_title(
                "Baseline corrected dataset preview", fontweight="bold", fontsize=8
            )
            if self.zoompreview > 1:
                zb = 1.0  # self.zoompreview
                zlim = [corrected.data.min() / zb, corrected.data.max() / zb]
                _ = corrected.plot_stack(ax=ax2, colorbar=False, zlim=zlim, clear=False)
            else:
                _ = corrected.plot_stack(ax=ax2, colorbar=False, clear=False)

        show_basecor(ax2)

        def onselect(xmin, xmax):
            self.ranges.append([xmin, xmax])
            self.show_regions(ax1)
            show_basecor(ax2)
            fig.canvas.draw()

        def onclick(event):
            if event.button == 3:
                for i, r in enumerate(self.ranges):
                    if r[0] > event.xdata or r[1] < event.xdata:
                        continue
                    else:
                        self.ranges.remove(r)
                        self.show_regions(ax1)
                        show_basecor(ax2)
                        fig.canvas.draw()  # _idle

        _ = fig.canvas.mpl_connect("button_press_event", onclick)

        _ = SpanSelector(
            ax1,
            onselect,
            "horizontal",
            minspan=5,
            button=[1],
            useblit=True,
            rectprops=dict(alpha=0.5, facecolor="blue"),
        )

        fig.canvas.draw()

        return


# ..............................................................................
def basc(dataset, *ranges, **kwargs):
    """
    Compute a baseline correction using the BaselineCorrection processor.

    2 methods are proposed :

    * ``sequential`` (default) = classical polynom fit or spline
      interpolation with separate fitting of each row (spectrum)
    * ``multivariate`` = SVD modeling of baseline, polynomial fit of PC's
      and calculation of the modelled baseline spectra.

    Parameters
    ----------
    dataset : a [NDDataset| instance
        The dataset where to calculate the baseline.
    *ranges : a variable number of pair-tuples
        The regions taken into account for the manual baseline correction.
    **kwargs : dict
        See other parameters.

    Other Parameters
    ----------------
    dim : str or int, keyword parameter, optional, default='x'.
        Specify on which dimension to apply the apodization method. If `dim` is specified as an integer
        it is equivalent  to the usual `axis` numpy parameter.
    method : str, keyword parameter, optional, default='sequential'
        Correction method among ['multivariate','sequential']
    interpolation : string, keyword parameter, optional, default='polynomial'
        Interpolation method for the computation of the baseline, among ['polynomial','pchip']
    order : int, keyword parameter, optional, default=6
        If the correction method polynomial, this give the polynomial order to use.
    npc : int, keyword parameter, optional, default=5
        Number of components to keep for the ``multivariate`` method

    See Also
    --------
    BaselineCorrection : Manual baseline corrections.
    abc : Automatic baseline correction.

    Notes
    -----
    For more flexibility and functionality, it is advised to use the BaselineCorrection processor instead.

    Examples
    --------
    .. plot::
        :include-source:

        import spectrochempy as scp
        nd = scp.read('irdata/nh4y-activation.spg')
        ndp = nd[:, 1291.0:5999.0]

        ranges=[[5996., 5998.], [1290., 1300.],
                [2205., 2301.], [5380., 5979.],
                [3736., 5125.]]

        ndcorr = scp.basc(ndp, *ranges,method='multivariate', interpolation='pchip', npc=8)
        ndcorr.plot()
        scp.show()
    """
    blc = BaselineCorrection(dataset)
    if not ranges and dataset.meta.regions is not None:
        # use the range stored in metadata
        ranges = dataset.meta.regions["baseline"]
    return blc.compute(*ranges, **kwargs)


# ======================================================================================================================
# abc # TODO: some work to perform on this
# ======================================================================================================================
def abc(dataset, dim=-1, **kwargs):
    """
    Automatic baseline correction.

    Various algorithms are provided to calculate the baseline automatically.

    Parameters
    ----------
    dataset : a [NDDataset| instance
        The dataset where to calculate the baseline.
    dim : str or int, optional
        The dataset dimentsion where to calculate the baseline. Default is -1.
    **kwargs : dict
        See other parameters.

    Returns
    -------
    baseline_corrected
        A baseline corrected dataset.
    baseline_only
        Only the baseline (apply must be set to False).
    baseline_points
        Points where the baseline is calculated (return_points must be set to True).

    Other Parameters
    ----------------
    basetype : string, optional, default: 'linear'
        See notes - available = linear, basf, ...
    window : float/int, optional, default is 0.05
        If float <1 then the corresponding percentage ot the axis size is taken as window.
    nbzone : int, optional, default is 32
        Number of zones. We will divide the size of the last axis by this number
        to determine the number of points in each zone (nw).
    mult : int
        A multiplicator. determine the number of point for the database calculation (nw*mult<n base points).
    nstd : int, optional, default is 2 times the standard error
        Another multiplicator. Multiply the standard error to determine the region in which points are from the
        baseline.
    polynom : bool, optional, default is True
        If True a polynom is computed for the base line, else an interpolation is achieved betwwen points.
    porder : int, default is 6
        Order of the polynom to fit on the baseline points
    return_points : bool, optional, default is False
        If True, the points abscissa used to determine the baseline are returned.
    apply : bool, optional, default is True
        If apply is False, the data are not modified only the baseline is returned.
    return_pts : bool, optional, default is False
        If True, the baseline reference points are returned.

    See Also
    --------
    BaselineCorrection : Manual baseline corrections.
    basc : Manual baseline correction.

    Notes
    -----
    #TODO: description of these algorithms
    * linear -
    * basf -

    Examples
    --------
    To be done
    """
    # # options evaluation
    # parser = argparse.ArgumentParser(description='BC processing.', usage="""
    # ab [-h] [--mode {linear,poly, svd}] [--dryrun]
    #                        [--window WINDOW] [--step STEP] [--nbzone NBZONE]
    #                        [--mult MULT] [--order ORDER] [--verbose]
    # """)
    # # positional arguments
    # parser.add_argument('--mode', '-mo', default='linear',
    #                      choices=['linear', 'poly', 'svd'], help="mode of correction")
    # parser.add_argument('--dryrun', action='store_true', help='dry flag')
    #
    # parser.add_argument('--window', '-wi', default=0.05, type=float, help='selected window for linear and svd bc')
    # parser.add_argument('--step', '-st', default=5, type=int, help='step for svd bc')
    # parser.add_argument('--nbzone', '-nz', default=32, type=int, help='number of zone for poly')
    # parser.add_argument('--mult', '-mt', default=4, type=int, help='multiplicator of zone for poly')
    # parser.add_argument('--order', '-or', default=5, type=int, help='polynom order for poly')
    #
    # parser.add_argument('--verbose', action='store_true', help='verbose flag')
    # args = parser.parse_args(options.split())
    #
    # source.history.append('baseline correction mode:%s' % args.mode)

    inplace = kwargs.pop("inplace", False)
    dryrun = kwargs.pop("dryrun", False)

    # output dataset inplace or not
    if not inplace or dryrun:  # default
        new = dataset.copy()
    else:
        new = dataset

    axis, dim = new.get_axis(dim, negative_axis=True)
    swaped = False
    if axis != -1:
        new.swapdims(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    base = _basecor(new.data.real, **kwargs)

    if not dryrun:
        new.data -= base  # return the corrected spectra
    else:
        new.data = base  # return the baseline

    # restore original data order if it was swaped
    if swaped:
        new.swapdims(axis, -1, inplace=True)  # must be done inplace

    new.history = "`abc` Baseline correction applied."
    return new


# ..............................................................................
def ab(dataset, dim=-1, **kwargs):
    """
    Alias of `abc`.
    """
    return abs(dataset, dim, **kwargs)


# ..............................................................................
@_units_agnostic_method
def dc(dataset, **kwargs):
    """
    Time domain baseline correction.

    Parameters
    ----------
    dataset : nddataset
        The time domain daatset to be corrected.
    kwargs : dict, optional
        Additional parameters.

    Returns
    -------
    dc
        DC corrected array.

    Other Parameters
    ----------------
    len : float, optional
        Proportion in percent of the data at the end of the dataset to take into account. By default, 25%.
    """

    len = int(kwargs.pop("len", 0.25) * dataset.shape[-1])
    dc = np.mean(np.atleast_2d(dataset)[..., -len:])
    dataset -= dc

    return dataset


# =======================================================================================================================
# private functions
# =======================================================================================================================
def _basecor(data, **kwargs):
    mode = kwargs.pop("mode", "linear")

    if mode == "linear":
        return _linearbase(data, **kwargs)

    if mode == "svd":
        return _svdbase(data, **kwargs)

    if mode == "poly":
        return _polybase(data, **kwargs)
    else:
        raise ValueError(f"`ab` mode = `{mode}`  not known")


#
# _linear mode
#
def _linearbase(data, **kwargs):
    # Apply a linear baseline correction
    # Very simple and naive procedure that compute a straight baseline from side to the other
    # (averging on a window given by the window parameters : 5% of the total width on each side by default)

    window = kwargs.pop("window", 0.05)

    if window <= 1.0:
        # percent
        window = int(data.shape[-1] * window)

    if len(data.shape) == 1:
        npts = float(data.shape[-1])
        a = (data[-window:].mean() - data[:window].mean()) / (npts - 1.0)
        b = data[:window].mean()
        baseline = a * np.arange(npts) + b

    else:
        npts = float(data.shape[-1])
        a = (data[:, -window:].mean(axis=-1) - data[:, :window].mean(axis=-1)) / (
            npts - 1.0
        )
        b = data[:, :window].mean(axis=-1)
        baseline = (((np.ones_like(data).T * a).T * np.arange(float(npts))).T + b).T

    return baseline


def _planeFit(points):
    # p, n = planeFit(points)  # copied from https://stackoverflow.com/a/18968498
    #
    # Fit an multi-dimensional plane to the points.
    # Return a point on the plane and the normal.
    #
    # Parameters
    # ----------
    # points :
    #
    # Notes
    # -----
    #     Replace the nonlinear optimization with an SVD.
    #     The following creates the moment of inertia tensor, M, and then
    #     SVD's it to get the normal to the plane.
    #     This should be a close approximation to the least-squares fit
    #     and be much faster and more predictable.
    #     It returns the point-cloud center and the normal.

    from numpy.linalg import svd

    npts = points.shape[0]
    points = np.reshape(points, (npts, -1))
    assert points.shape[0] < points.shape[1]
    ctr = points.mean(axis=1)
    x = points - ctr[:, None]
    M = np.dot(x, x.T)
    return ctr, svd(M)[0][:, -1]


def _svdbase(data, args=None, retw=False):
    # Apply a planar baseline correction to 2D data
    import pandas as pd  # TODO: suppress this need

    if not args:
        window = 0.05
        step = 5
    else:
        window = args.window
        step = args.step

    if window <= 1.0:
        # percent
        window = int(data.shape[-1] * window)

    data = pd.DataFrame(
        data
    )  # TODO: facilitate the manipulation (but to think about further)
    a = pd.concat([data.iloc[:window], data.iloc[-window:]])
    b = pd.concat(
        [data.iloc[window:-window, :window], data.iloc[window:-window, -window:]],
        axis=1,
    )
    bs = pd.concat([a, b])
    bs = bs.stack()
    bs.sort()
    x = []
    y = []
    z = []
    for item in bs.index[::step]:
        x.append(item[0])
        y.append(item[1])
        z.append(bs[item].real)

    norm = np.max(np.abs(z))
    z = np.array(z)
    z = z / norm
    XYZ = np.array((x, y, z))
    p, n = _planeFit(XYZ)
    d = np.dot(p, n)

    col = data.columns
    row = data.index
    X, Y = np.meshgrid(col, row)
    Z = -norm * (n[0] * X + n[1] * Y - d) / n[2]

    if retw:
        return Z, None  # TODO: return something
    return Z


def _polybase(data, **kwargs):
    # Automatic baseline correction

    if data.ndim == 1:
        dat = np.array(
            [
                data,
            ]
        )

    nbzone = kwargs.pop("nbzone", 64)
    mult = kwargs.pop("mult", 4)
    order = kwargs.pop("order", 6)

    npts = data.shape[-1]
    w = np.arange(npts)

    baseline = np.ma.masked_array(dat, mask=True)
    sigma = 1.0e6
    nw = int(npts / nbzone)

    # print (nw)
    # unmask extremities of the baseline
    baseline[:, :nw].mask = False
    baseline[:, -nw:].mask = False

    for j in range(nbzone):
        s = dat[:, nw * j : min(nw * (j + 1), npts + 1)]
        sigma = min(s.std(), sigma)

    nw = nw * 2  # bigger window
    nw2 = int(nw / 2)

    found = False
    nb = 0
    nstd = 2.0
    while (not found) or (nb < nw * mult):
        nb = 0
        for i in range(nw2, npts - nw2 + 1, 1):
            s1 = dat[:, max(i - 1 - nw2, 0) : min(i - 1 + nw2, npts + 1)]
            s2 = dat[:, max(i - nw2, 0) : min(i + nw2, npts + 1)]
            s3 = dat[:, max(i + 1 - nw2, 0) : min(i + 1 + nw2, npts + 1)]
            mi1, mi2, mi3 = s1.min(), s2.min(), s3.min()
            ma1, ma2, ma3 = s1.max(), s2.max(), s3.max()

            if (
                abs(ma1 - mi1) < float(nstd) * sigma
                and abs(ma2 - mi2) < float(nstd) * sigma
                and abs(ma3 - mi3) < float(nstd) * sigma
            ):
                found = True
                nb += 1
                baseline[:1, i].mask = False  # baseline points

        # increase nstd
        nstd = nstd * 1.1
    debug_("basf optimized nstd: %.2F mult: %.2f" % (nstd, mult))

    wm = np.array(list(zip(*np.argwhere(~baseline[:1].mask)))[1])
    bm = baseline[:, wm]
    if data.ndim > 1:
        bm = smooth(bm.T, max(int(dat.shape[0] / 10), 3)).T
    bm = smooth(bm, max(int(dat.shape[-1] / 10), 3))

    # if not polynom:
    #    sr = pchip(wm, bm.real)
    #    si = pchip(wm, bm.imag)
    #    baseline = sr(w) + si(w) * 1.0j
    #    baseline = smooth(baseline, window_len=int(nw / 4))
    # else:
    # fit a polynom
    pf = np.polyfit(wm, bm.T, order).T
    for i, row in enumerate(pf[:]):
        poly = np.poly1d(row)
        baseline[i] = poly(w)

    if data.ndim == 1:
        baseline = baseline[0]

    return baseline


if __name__ == "__main__":
    pass
