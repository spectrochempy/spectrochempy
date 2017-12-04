# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================
# author : A.Travert & C.Fernandez
# =============================================================================

import numpy as np
import scipy.interpolate
from traitlets import (Int, Instance, HasTraits, Float, Unicode, Tuple)
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

from spectrochempy.dataset.nddataset import NDDataset

# local imports
# -----------------------------------------------------------------------------
from spectrochempy.dataset.ndcoords import CoordRange
from spectrochempy.plotters.multiplot import multiplot

__all__ = ['BaselineCorrection']


class BaselineCorrection(HasTraits):
    """
    Baseline Correction processor

    2 methods are proposed:

        * ``sequential`` (default) = classical polynom fit or spline \
            interpolation with one fit per spectrum
        * ``multivariate`` = SVD modeling of baseline, polynomial fit of PC's \
        and calculation of the modelled baseline spectra.

    Interactive mode is proposed using the `interactive function.`

    """
    source = Instance(NDDataset)
    corrected_source = Instance(NDDataset)
    method = Unicode('multivariate')
    interpolation = Unicode('pchip')
    axis = Int(1)
    order = Int(6, min=1, allow_none=True)
    npc = Int(5, min=1, allow_none=True)
    zoompreview = Float(1.)
    figsize=Tuple((8,6))

    def __init__(self,
                 source,
                 **kwargs):

        self.source = source
        self.corrected = self.source.copy()

        self._setup(**kwargs)

        x = source.x.data
        self.ranges = [
                [x[0], x[2]],
                [x[-3], x[-1]]
            ]

    def _setup(self, **kwargs):

        self.axis = kwargs.get('axis', self.axis)
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

    def compute(self, *ranges, **kwargs):
        """Base function for dataset baseline correction.
        """

        self._setup(**kwargs)

        # output dataset
        new = self.source.copy()

        # we assume that the last dimension if always the dimension to which
        # we want to subtract the baseline.
        # Swap the axes to be sure to be in this situation

        if self.axis == new.ndim - 1:
            self.axis = -1

        swaped = False
        if self.axis != -1:
            new.swapaxes(self.axis, -1, inplace=True)
            swaped = True

        # most of the time we need sorted axis, so let's do it now
        coords = new.coordset(-1)

        sorted = False
        if new.coordset(-1).is_reversed:
            new.sort(axis=-1, inplace=True)
            sorted = True

        coords = new.coordset(-1)
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

        sbase = s[0].concatenate(*s[1:],
                                 axis=-1)  # TODO: probably we could use masked
        # data instead of concatenating
        xbase = sbase.coordset(-1)

        if self.method == 'sequential':

            if self.interpolation == 'polynomial':
                # bad fit when NaN values => are replaced by 0
                if np.any(np.isnan(sbase)):
                    sbase[np.isnan(sbase)] = 0

                polycoef = np.polynomial.polynomial.polyfit(xbase.data,
                                                            sbase.data.T,
                                                            deg=self.order,
                                                            rcond=None,
                                                            full=False)
                baseline = np.polynomial.polynomial.polyval(coords.data,
                                                            polycoef)

            elif self.interpolation == 'pchip':
                for i in range(new.shape[0]):
                    y = scipy.interpolate.PchipInterpolator(
                            xbase.data, sbase.data[i, :])
                    baseline[i, :] = y(coords)

        elif self.method == 'multivariate':

            # SVD of Sbase
            U, s, Vt = np.linalg.svd(sbase.data, full_matrices=False,
                                     compute_uv=True)

            # npc cannot be higher than the size of s
            npc = min(self.npc, s.shape[0])

            # select npc loadings & compute scores
            Pt = (Vt[0:npc, :])
            T = np.dot(U[:, 0:npc], np.diag(s)[0:npc, 0:npc])

            baseline_loadings = np.zeros((npc, new.shape[1]))

            if self.interpolation == 'pchip':
                for i in range(npc):
                    y = scipy.interpolate.PchipInterpolator(xbase.data,
                                                            Pt[i, :])
                    baseline_loadings[i, :] = y(coords)

            elif self.interpolation == 'polynomial':
                polycoef = np.polynomial.polynomial.polyfit(xbase.data,
                                                            Pt.T, deg=self.order,
                                                            rcond=None,
                                                            full=False)

                baseline_loadings = np.polynomial.polynomial.polyval(
                    coords.data,
                    polycoef)

            baseline = np.dot(T, baseline_loadings)

        new.name = '*' + self.source.name
        new.data = new.data - baseline

        # eventually sort back to the original order
        if sorted:
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

    def run(self, *ranges, **kwargs):
        """
        Interactive version of the basecor function.

        """
        self._setup(**kwargs)
        sources = [self.source, self.source]
        labels = ['\nClick & span with left mouse button to set a baseline region.'
                      '\nClick on right button on a region to remove it.',
                  'Baseline corrected dataset preview']
        axes = multiplot(sources, labels,
                  method='stack',
                  sharex=True,
                  nrow = 2,
                  ncol = 1,
                  figsize=self.figsize,
                  suptitle = 'INTERACTIVE BASELINE CORRECTION')

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
            if self.zoompreview>1:
                zb = 1. # self.zoompreview
                zlim = [corrected.data.min()/zb,corrected.data.max()/zb]
                ax2 = corrected.plot_stack(ax=ax2, colorbar=False,
                                           zlim=zlim, hold=True)
            else:
                ax2 = corrected.plot_stack(ax=ax2, colorbar=False, hold=True)

        show_basecor(ax2)

        def onselect(xmin, xmax):
            self.ranges.append([xmin, xmax])
            sp = ax1.axvspan(xmin, xmax, facecolor='#2ca02c', alpha=0.5)
            sps.append(sp)
            show_basecor(ax2)
            fig.canvas.draw()

        def onclick(event):
            if event.button==3:
                for i, r in enumerate(self.ranges):
                    if r[0]>event.xdata or r[1]<event.xdata:
                        continue
                    else:
                        self.ranges.remove(r)
                        sp = sps[i]
                        sps.remove(sp)
                        sp.remove()
                        show_basecor(ax2)
                        fig.canvas.draw_idle()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        span =  SpanSelector(ax1, onselect, 'horizontal',  minspan=5,
                             button=[1], useblit=True,
                             rectprops=dict(alpha=0.5, facecolor='blue'))


        fig.canvas.draw()

        return



if __name__ == '__main__':

    from spectrochempy.api import *
    import os
    nd = NDDataset.read_omnic(
        os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))

    ndp = (nd - nd[-1])[:,
          1290.0:5999.0]  # Important note that we use foating point number
    # integer would mean points, not wavenumbers!


    ibc = BaselineCorrection(ndp, axis=-1,
                             method='multivariate',
                             interpolation='pchip',
                             order=3,
                             npc=5,
                             zoompreview=3)
    ranges=[]
    span = ibc.run(*ranges)
    plt.show()

    print (ibc.corrected)