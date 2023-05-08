# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the `BaselineCorrection` class for baseline corrections.
"""
__all__ = ["ab", "abc", "dc", "basc", "detrend"]

__dataset_methods__ = __all__

import numpy as np

from spectrochempy.core import debug_
from spectrochempy.core.processors.filter import smooth
from spectrochempy.core.processors.utils import _units_agnostic_method
from spectrochempy.utils.decorators import deprecated


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
    Baseline : Manual baseline corrections.
    abc : Automatic baseline correction.

    Notes
    -----
    For more flexibility and functionality, it is advised to use the Baseline class
    processor instead.
    """
    from spectrochempy.analysis.preprocessing.baseline import Baseline

    blc = Baseline()
    if ranges:
        blc.ranges = ranges
    for key in kwargs:
        setattr(blc, key, kwargs[key])
    blc.fit(dataset)
    return blc.transform()


def detrend(dataset, order="linear", bp=None, **kwargs):
    """
    Remove polynomial trend along a dimension from dataset.

    Depending on the ``order``parameter, `detrend` removes the best-fit polynomial line
    (in the least squares sense) from the data and returns the remaining data.

    See examples section for the different syntax.

    Parameters
    ----------
    dataset : `NDDataset`
        The input data.
    order : non-negative `int` or a `str` among ['constant', 'linear', 'quadratic'], optional, default='linear'
        The order of the polynomial trend.

        * If ``order=0`` or ``'constant'``\ , the mean of data is subtracted to remove
          a shift trend.
        * If ``order=1`` or ``'linear'`` (default), the best straith-fit line is
          subtracted from data to remove a linear trend (drift).
        * If order=2 or ``order=quadratic``\ ,  the best fitted nth-degree polynomial
          line is subtracted from data to remove a quadratic polynomial trend.
        * ``order=n`` can also be used to remove any nth-degree polynomial trend.

    bp : :term:`array_like`\ , optional
        Breakpoints to define piecewise segments of the data, specified as a vector
        containing coordinate values or indices indicating the location of the
        breakpoints. Breakpoints are useful when you want to compute separate trends
        for different segments of the data.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    `NDDataset`
        The detrended dataset.

    Other Parameters
    ----------------
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new
        object

    See Also
    --------
    Baseline : Manual baseline correction.
    abc : Automatic baseline correction.
    autosub : Subtraction of reference.

    Examples
    --------

    >>> dataset = scp.read("irdata/nh4y-activation.spg")
    >>> dataset.detrend(order='constant')
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))
    """
    if not kwargs.pop("inplace", False):
        # default
        new = dataset.copy()
    else:
        new = dataset

    type = kwargs.pop("type", None)
    if type is not None:
        deprecated("type", replace="order", removed="0.7")
        kwargs["order"] = kwargs.pop("type")

    dim = kwargs.pop("dim", None)
    if dim is not None:
        deprecated(
            "dim",
            extra_msg="Transpose your data before processing if needed.",
            removed="0.7",
        )

    from spectrochempy.analysis.preprocessing.baseline import Baseline

    blc = Baseline()
    blc.interpolation = "detrend"
    blc.order = order
    blc.bp = bp
    blc.fit(new)

    return new


def ab(dataset, dim=-1, **kwargs):
    """
    Alias of `abc` .
    """
    return abs(dataset, dim, **kwargs)


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


# ======================================================================================
# abc # TODO: some work to perform on this
# ======================================================================================
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
    **kwargs
        Optional keyword parameters (see Other Parameters).

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
        If float <1 then the corresponding percentage of the axis size is taken as window.
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
    swapped = False
    if axis != -1:
        new.swapdims(axis, -1, inplace=True)  # must be done in  place
        swapped = True

    base = _basecor(new.data.real, **kwargs)

    if not dryrun:
        new.data -= base  # return the corrected spectra
    else:
        new.data = base  # return the baseline

    # restore original data order if it was swapped
    if swapped:
        new.swapdims(axis, -1, inplace=True)  # must be done inplace

    new.history = "`abc` Baseline correction applied."
    return new


# ======================================================================================
# private functions
# ======================================================================================
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
