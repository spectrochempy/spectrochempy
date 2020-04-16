# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module implements the `BaselineCorrection` class for baseline corrections.

"""
__all__ = ['BaselineCorrection', 'ab']
__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import scipy.interpolate
from traitlets import (Int, Instance, HasTraits, Float, Unicode, Tuple)
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------

from ..dataset.ndcoordrange import CoordRange
from ..plotters.multiplot import multiplot
from ..dataset.nddataset import NDDataset
from ...utils import docstrings
from .smooth import smooth
from .. import error_, debug_

class BaselineCorrection(HasTraits):
    """
    Baseline Correction processor

    2 methods are proposed :

    * ``sequential`` (default) = classical polynom fit or spline
      interpolation with separate fitting of each row (spectrum)
    * ``multivariate`` = SVD modeling of baseline, polynomial fit of PC's
      and calculation of the modelled baseline spectra.

    Interactive mode is proposed using the interactive function : :meth:`run`.
    
    """
    dataset = Instance(NDDataset)
    corrected = Instance(NDDataset)
    method = Unicode('multivariate')
    interpolation = Unicode('pchip')
    axis = Int(-1)
    dim = Unicode('')
    order = Int(6, min=1, allow_none=True)
    npc = Int(5, min=1, allow_none=True)
    zoompreview = Float(1.)
    figsize = Tuple((8, 6))

    @docstrings.get_sectionsf('BaselineCorrection', sections=['Parameters', 'Other Parameters'])
    @docstrings.dedent
    def __init__(self,
                 dataset,
                 *ranges,
                 **kwargs):
        """
        Parameters
        ----------
        dataset : |NDDataset|
            The dataset to be transformed
        *ranges : a variable number of pair-tuples
            The regions taken into account for the manual baseline correction.
        
        Other Parameters
        ----------------
        dim : str or int, keyword parameter, optional, default='x'.
            Specify on which dimension to apply the apodization method. If `dim` is specified as an integer it is equivalent
            to the usual `axis` numpy parameter.
        method : str, keyword parameter, optional, default='multivariate'
            Correction method among ['multivariate','sequential']
        interpolation : string, keyword parameter, optional, default='polynomial'
            Interpolation method for the computation of the baseline, among ['polynomial','pchip']
        order : int, keyword parameter, optional, default=6
            If the correction method polynomial, this give the polynomial order to use.
        npc : int, keyword parameter, optional, default=5
            Number of components to keep for the ``multivariate`` method
        zoompreview : float, keyword parameter, optional, default=1.0
            The zoom factor for the preview in interactive mode
        figsize : tuple, keyword parameter, optional, default=(8, 6)
            Size of the figure to display in inch
            
        Examples
        --------

        .. plot::
            :include-source:

            from spectrochempy import *
            nd = NDDataset.read_omnic(os.path.join('irdata',
                                        'nh4y-activation.spg'))
            ndp = nd[:, 1291.0:5999.0]
            bc = BaselineCorrection(ndp,method='multivariate',
                                        interpolation='pchip',
                                        npc=8)
            ranges=[[5996., 5998.], [1290., 1300.],
                    [2205., 2301.], [5380., 5979.],
                    [3736., 5125.]]
            span = bc.compute(*ranges)
            _ = bc.corrected.plot_stack()
            show()

        """
        self.dataset = dataset
        self.corrected = self.dataset.copy()

        self._setup(**kwargs)

        x = getattr(dataset, self.dim).data

        self.ranges = [[x[0], x[2]], [x[-3], x[-1]]]
        
        if ranges:
            self.ranges.extend(ranges)

    # ..................................................................................................................
    def _setup(self, **kwargs):
        
        if 'axis' in kwargs.keys() or 'dim' in kwargs.keys():
            self.axis, self.dim = self.dataset.get_axis(**kwargs)  # using dim, dims or axis keyword arguments
            
        if not self.dim:
            self.dim = self.dataset.dims[self.axis]
            
        self.method = kwargs.get('method', self.method)
        self.interpolation = kwargs.get('interpolation', self.interpolation)
        if self.interpolation == 'polynomial':
            self.order = int(kwargs.get('order', self.order))
        if self.method == 'multivariate':
            self.npc = int(kwargs.get('npc', self.npc))
        self.zoompreview = kwargs.get('zoompreview', self.zoompreview)
        self.figsize = kwargs.get('figsize', self.figsize)

    # ..................................................................................................................
    def __call__(self, *ranges, **kwargs):

        return self.compute(*ranges, **kwargs)

    # ..................................................................................................................
    docstrings.delete_params('BaselineCorrection.other_parameters', 'dataset')
    @docstrings.dedent
    def compute(self, *ranges, **kwargs):
        """
        Base function for dataset baseline correction.

        Parameters
        ----------
        %(BaselineCorrection.parameters.no_dataset)s
        
        Other Parameters
        ----------------
        %(BaselineCorrection.other_parameters)s

        """

        self._setup(**kwargs)

        # output dataset
        new = self.dataset.copy()

        # we assume that the first dimension if always the dimension to which we want to subtract the baseline.
        # Swap the axes to be sure to be in this situation

        swaped = False
        if self.axis != -1:
            new.swapaxes(self.axis, -1, inplace=True)
            swaped = True

        # most of the time we need sorted axis, so let's do it now
        is_sorted = False
        if new.coords[self.dim].reversed:
            new.sort(dim=self.dim, inplace=True, descend=False)
            is_sorted = True

        coords = new.coords[self.dim]
        baseline = np.zeros_like(new)

        if ranges:
            self.ranges.extend(ranges)

        ranges = CoordRange(*self.ranges)

        # Extract: Sbase: the matrix of data corresponding to ranges
        #          xbase: the xaxis values corresponding to ranges

        s = []
        for pair in ranges:
            # determine the slices

            sl = slice(*pair)
            s.append(new[..., sl])

        sbase = NDDataset.concatenate(s, axis=-1)
        # TODO: probably we could use masked data instead of concatenating - could be faster
        xbase = sbase.coords(self.dim)

        if self.method == 'sequential':

            if self.interpolation == 'polynomial':
                # bad fit when NaN values => are replaced by 0
                if np.any(np.isnan(sbase)):
                    sbase[np.isnan(sbase)] = 0

                polycoef = np.polynomial.polynomial.polyfit(xbase.data, sbase.data.T,
                                                            deg=self.order, rcond=None, full=False)
                baseline = np.polynomial.polynomial.polyval(coords.data, polycoef)

            elif self.interpolation == 'pchip':
                for i in range(new.shape[0]):
                    y = scipy.interpolate.PchipInterpolator(xbase.data, sbase.data[i])
                    baseline[i] = y(coords)

        elif self.method == 'multivariate':

            # SVD of Sbase
            U, s, Vt = np.linalg.svd(sbase.data, full_matrices=False, compute_uv=True)

            # npc cannot be higher than the size of s
            npc = min(self.npc, s.shape[0])

            # select npc loadings & compute scores
            Pt = (Vt[0:npc])
            T = np.dot(U[:, 0:npc], np.diag(s)[0:npc, 0:npc])

            baseline_loadings = np.zeros((npc, new.shape[-1]))

            if self.interpolation == 'pchip':
                for i in range(npc):
                    y = scipy.interpolate.PchipInterpolator(xbase.data, Pt[i])
                    baseline_loadings[i] = y(coords)

            elif self.interpolation == 'polynomial':
                polycoef = np.polynomial.polynomial.polyfit(xbase.data, Pt.T,
                                                            deg=self.order, rcond=None, full=False)

                baseline_loadings = np.polynomial.polynomial.polyval(coords.data, polycoef)

            baseline = np.dot(T, baseline_loadings)

        new.data = new.data - baseline

        # eventually sort back to the original order
        if is_sorted:
            new.sort(axis=-1, inplace=True, descend=True)

        new.history = str(new.modified) + \
                      ': ' + 'Baseline correction.' + ' Method: '
        if self.method == 'Multivariate':
            new.history = 'Multivariate (' + str(self.npc) + ' PCs).'
        else:
            new.history = 'Sequential.'

        new.history = 'Interpolation: '

        if self.interpolation == 'polynomial':
            new.history = 'Polynomial, order=' + str(self.order) + '.\n'
        else:
            new.history = 'Pchip. \n'

        if swaped:
            new = new.swapaxes(self.axis, -1)

        self.corrected = new
        return new

    @docstrings.dedent
    def run(self, *ranges, **kwargs):
        """
        Interactive version of the baseline correction.

        Parameters
        ----------
        %(BaselineCorrection.parameters.no_dataset)s

        """
        self._setup(**kwargs)

        datasets = [self.dataset, self.dataset]
        labels = ['\nClick & span with left mouse button to set a baseline region.'
                  '\nClick on right button on a region to remove it.',
                  'Baseline corrected dataset preview']
        axes = multiplot(datasets, labels,
                         method='stack',
                         sharex=True,
                         nrow=2,
                         ncol=1,
                         figsize=self.figsize,
                         suptitle='INTERACTIVE BASELINE CORRECTION')

        fig = plt.gcf()
        fig.canvas.draw()

        ax1 = axes['axe11']
        ax2 = axes['axe21']

        sps = []

        if ranges:
            self.ranges.extend(ranges)

        self.ranges = list(self.ranges)

        for x in self.ranges:
            x.sort()
            y = list(ax1.get_ylim())
            sp = ax1.axvspan(x[0], x[1], facecolor='#2ca02c', alpha=0.5)
            sps.append(sp)

        def show_basecor(ax2):

            corrected = self.compute()

            ax2.clear()
            ax2.set_title('Baseline corrected dataset preview',
                          fontweight='bold')
            if self.zoompreview > 1:
                zb = 1.  # self.zoompreview
                zlim = [corrected.data.min() / zb, corrected.data.max() / zb]
                ax2 = corrected.plot_stack(ax=ax2, colorbar=False,
                                           zlim=zlim, clear=False)
            else:
                ax2 = corrected.plot_stack(ax=ax2, colorbar=False, clear=False)

        show_basecor(ax2)

        def onselect(xmin, xmax):
            self.ranges.append([xmin, xmax])
            sp = ax1.axvspan(xmin, xmax, facecolor='#2ca02c', alpha=0.5)
            sps.append(sp)
            show_basecor(ax2)
            fig.canvas.draw()

        def onclick(event):
            if event.button == 3:
                for i, r in enumerate(self.ranges):
                    if r[0] > event.xdata or r[1] < event.xdata:
                        continue
                    else:
                        self.ranges.remove(r)
                        sp = sps[i]
                        sps.remove(sp)
                        sp.remove()
                        show_basecor(ax2)
                        fig.canvas.draw_idle()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        span = SpanSelector(ax1, onselect, 'horizontal', minspan=5,
                            button=[1], useblit=True,
                            rectprops=dict(alpha=0.5, facecolor='blue'))

        fig.canvas.draw()

        return

# =======================================================================================================================
# ab
#=======================================================================================================================
def ab(dataset, dim=-1, **kwargs):
    """
    Automatic baseline correction

    Parameters
    ----------
    source: a source instance
    basetype: string, optional, default: 'linear'
        see notes - available = linear, basf, ...
    window: float/int, optional, default is 0.05
        if float <1 then the corresponding percentage ot the axis size is taken as window
    nbzone: int, optional, default is 32
        Number of zones. We will divide the size of the last axis by this number
        to determine the number of points in each zone (nw)
    mult: int
        A multiplicator. determine the number of point for the database calculation (nw*mult<n base points)
    nstd: int, optional, default is 2 times the standard error
        Another multiplicator. Multiply the standard error to determine the region in which points are from the baseline
    polynom: bool, optional, default is True
        If True a polynom is computed for the base line, else an interpolation is achieved betwwen points.
    porder: int, default is 6
        Order of the polynom to fit on the baseline points
    return_pts: bool, optional, default is False
        If True, the points abscissa used to determine the baseline are returned
    apply: bool, optional, default is True
        if apply is False, the data are not modified only the baseline is returned
    return_pts: bool, optional, default is False
        if True, the baseline reference points are returned
    axis: optional, default is -1

    Notes
    -----
    #TODO: description of these algorithms
    * linear -
    * basf -

    """
    # # options evaluation
    # parser = argparse.ArgumentParser(description='BC processing.', usage="""
    # ab [-h] [--mode {linear,poly, svd}] [--dryrun]
    #                        [--window WINDOW] [--step STEP] [--nbzone NBZONE]
    #                        [--mult MULT] [--order ORDER] [--verbose]
    # """)
    # # positional arguments
    # parser.add_argument('--mode', '-mo', default='linear', choices=['linear', 'poly', 'svd'], help="mode of correction")
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

    debug_('Automatic baseline correction')

    inplace = kwargs.pop('inplace', False)
    dryrun = kwargs.pop('dryrun', False)
    
    # output dataset inplace or not
    if not inplace or dryrun:           # default
        new = dataset.copy()
    else:
        new = dataset

    axis, dim = new.get_axis(dim, negative_axis=True)
    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    # select the last coordinates and check the unit validity
    lastcoord = new.coords[dim]
    if (lastcoord.units.dimensionality != '1/[time]' and lastcoord.units != 'ppm'):
        error_('`ab` apply only to dimensions with [frequency] dimensionality or with ppm units\n'
               'Baseline correction processing was thus cancelled')
        return new
    
    base = _basecor(new.data.real, **kwargs)
    
    if not dryrun:
        new.data -= base # return the corrected spectra
    else:
        new.data = base # return the baseline

    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace

    return new

# =======================================================================================================================
# private functions
#=======================================================================================================================
def _basecor(data, **kwargs):
    
    mode = kwargs.pop('mode', 'linear')
    
    if mode == 'linear':
        return _linearbase(data, **kwargs)
    
    if mode == 'svd':
        return _svdbase(data, **kwargs)
    
    if mode == 'poly':
        return _polybase(data, **kwargs)
    else:
        raise ValueError(f'`ab` mode = `{mode}`  not known')

#
# _linear mode
#
def _linearbase(data, **kwargs):
    # Apply a linear baseline correction
    # Very simple and naive procedure that compute a straight baseline from side to the other
    # (averging on a window given by the window parameters : 5% of the total width on each side by default)
    
    window = kwargs.pop('window', 0.05)
    
    if window <= 1.0:
        # percent
        window = int(data.shape[-1] * window)
    
    debug_(f"Linear base correction window : {window}")
    
    if len(data.shape) == 1:
        npts = float(data.shape[-1])
        a = (data[-window:].mean() - data[:window].mean()) / (npts - 1.)
        b = data[:window].mean()
        baseline = a * np.arange(npts) + b
    
    else:
        npts = float(data.shape[-1])
        a = (data[:, -window:].mean(axis=-1) - data[:, :window].mean(axis=-1)) / (npts - 1.)
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
    #Apply a planar baseline correction to 2D data
    
    if not args:
        window = 0.05
        step = 5
    else:
        window = args.window
        step = args.step
    
    if window <= 1.0:
        # percent
        window = int(data.shape[-1] * window)
    
    data = pd.DataFrame(data)  # TODO: facilitate the manipulation (but to think about further)
    a = pd.concat([data.iloc[:window], data.iloc[-window:]])
    b = pd.concat([data.iloc[window:-window, :window], data.iloc[window:-window, -window:]], axis=1)
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
    debug_(" origin baseline plane: ", p)
    debug_(" normal vector component:", n)
    
    col = data.columns
    row = data.index
    X, Y = np.meshgrid(col, row)
    Z = -norm * (n[0] * X + n[1] * Y - d) / n[2]
    
    if retw:
        return Z, None  #TODO: return something
    return Z


def _polybase(data, **kwargs):
    # Automatic baseline correction
    
    if data.ndim == 1:
        dat = np.array([data, ])
    
    nbzone = kwargs.pop('nbzone', 64)
    mult = kwargs.pop('mult', 4)
    order = kwargs.pop('order', 6)
    
    npts = data.shape[-1]
    w = np.arange(npts)
    
    nw = npts / nbzone
    baseline = np.ma.masked_array(dat, mask=True)
    
    sigma = 1.e6
    nw = int(npts / nbzone)
    
    # print (nw)
    # unmask extremities of the baseline
    baseline[:, :nw].mask = False
    baseline[:, -nw:].mask = False
    
    for j in range(nbzone):
        s = dat[:, nw * j:min(nw * (j + 1), npts + 1)]
        sigma = min(s.std(), sigma)
    
    nw = nw * 2  # bigger window
    nw2 = int(nw / 2)
    
    found = False
    nb = 0
    nstd = 2.
    while (not found) or (nb < nw * mult):
        nb = 0
        for i in range(nw2, npts - nw2 + 1, 1):
            s1 = dat[:, max(i - 1 - nw2, 0):min(i - 1 + nw2, npts + 1)]
            s2 = dat[:, max(i - nw2, 0):min(i + nw2, npts + 1)]
            s3 = dat[:, max(i + 1 - nw2, 0):min(i + 1 + nw2, npts + 1)]
            mi1, mi2, mi3 = s1.min(), s2.min(), s3.min()
            ma1, ma2, ma3 = s1.max(), s2.max(), s3.max()

            if abs(ma1 - mi1) < float(nstd) * sigma and abs(ma2 - mi2) < float(nstd) * sigma and abs(ma3 - mi3) < float(
                    nstd) * sigma:
                found = True
                nb += 1
                baseline[:1, i].mask = False  # baseline points
        
        # increase nstd
        nstd = nstd * 1.1
    debug_('basf optimized nstd: %.2F mult: %.2f' % (nstd, mult))
    
    wm = np.array(list(zip(*np.argwhere(~baseline[:1].mask)))[1])
    bm = baseline[:, wm]
    if data.ndim>1:
        bm = smooth(bm.T, length=max(int(dat.shape[0] / 10), 3)).T
    bm = smooth(bm, length=max(int(dat.shape[-1] / 10), 3))
    
    #if not polynom:
    #    sr = pchip(wm, bm.real)
    #    si = pchip(wm, bm.imag)
    #    baseline = sr(w) + si(w) * 1.0j
    #    baseline = smooth(baseline, window_len=int(nw / 4))
    #else:
    # fit a polynom
    pf = np.polyfit(wm, bm.T, order).T
    for i, row in enumerate(pf[:]):
        poly = np.poly1d(row)
        baseline[i] = poly(w)

    if data.ndim == 1:
        baseline = baseline[0]
        
    return baseline

if __name__ == '__main__':
    pass
