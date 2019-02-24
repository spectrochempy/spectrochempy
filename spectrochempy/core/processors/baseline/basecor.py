# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module implements the `BaselineCorrection` class for baseline corrections.

"""
__all__ = ['BaselineCorrection']

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

from spectrochempy.core.dataset.ndcoords import CoordRange
from spectrochempy.core.plotters.multiplot import multiplot
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import docstrings


class BaselineCorrection(HasTraits):
    """
    Baseline Correction processor

    2 methods are proposed:

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
    axis = Int(allow_none=True)
    order = Int(6, min=1, allow_none=True)
    npc = Int(5, min=1, allow_none=True)
    zoompreview = Float(1.)
    figsize = Tuple((8, 6))

    @docstrings.get_sectionsf('BaselineCorrection')
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
            The regions taken into account for the baseline correction.
        **kwargs : keywords arguments
            Known keywords are given below

            * dim : the dimension along which to apply the baseline correction usually 'x', *i.e.*, along .
            * method : ``multivariate`` or ``sequential``
            * interpolation : ``polynomial`` or ``pchip``
            * order : polynomial order, default=6
            * npc : number of components for the ``multivariate`` method

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

        dimx = dataset.dims[self.axis]
        x = getattr(dataset, dimx).data

        self.ranges = [
            [x[0], x[2]],
            [x[-3], x[-1]]
        ]
        if ranges:
            self.ranges.extend(ranges)

    def _setup(self, **kwargs):
        self.axis = self.dataset.get_axis(**kwargs)  # using dim, dims or axis keyword arguments
        if self.axis is None:
            self.axis = -1
        self.method = kwargs.get('method', self.method)
        self.interpolation = kwargs.get('interpolation', self.interpolation)
        if self.interpolation == 'polynomial':
            self.order = int(kwargs.get('order', self.order))
        if self.method == 'multivariate':
            self.npc = int(kwargs.get('npc', self.npc))
        self.zoompreview = kwargs.get('zoompreview', self.zoompreview)
        self.figsize = kwargs.get('figsize', self.figsize)

    def __call__(self, *ranges, **kwargs):

        return self.compute(*ranges, **kwargs)

    docstrings.delete_params('BaselineCorrection.parameters', 'dataset')

    @docstrings.dedent
    def compute(self, *ranges, **kwargs):
        """
        Base function for dataset baseline correction.

        Parameters
        ----------
        %(BaselineCorrection.parameters.no_dataset)s
        zoompreview : the zoom factor for the preview in interactive mode
        figsize : Size of the figure to display

        """

        self._setup(**kwargs)

        # output dataset
        new = self.dataset.copy()

        # we assume that the first dimension if always the dimension to which
        # we want to subtract the baseline.
        # Swap the axes to be sure to be in this situation

        swaped = False
        if self.axis != -1:
            new.swapaxes(self.axis, -1, inplace=True)
            swaped = True

        # most of the time we need sorted axis, so let's do it now
        coords = new.coords(0)

        is_sorted = False
        if new.coords(-1).reversed:
            new.sort(axis=-1, inplace=True, descend=False)
            is_sorted = True

        coords = new.coords(-1)
        baseline = np.zeros_like(new)

        if ranges:
            self.ranges.extend(ranges)

        ranges = CoordRange(*self.ranges)

        # Extract: Sbase: the matrix of data correponding to ranges
        #          xbase: the xaxis values corresponding to ranges

        s = []
        for pair in ranges:
            # determine the slices

            sl = slice(*pair)
            s.append(new[..., sl])

        sbase = s[0].concatenate(*s[1:], axis=-1)

        # TODO: probably we could use masked data instead of concatenating - could be faster
        xbase = sbase.coords(-1)

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
        if self.method == 'multivariate':
            new.history = 'multivariate (' + str(self.npc) + ' PCs).'
        else:
            new.history = 'sequential.'

        new.history = 'Interpolation: '

        if self.interpolation == 'polynomial':
            new.history = 'polynomial, order=' + str(self.order) + '.\n'
        else:
            new.history = 'pchip. \n'

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


if __name__ == '__main__':
    pass
