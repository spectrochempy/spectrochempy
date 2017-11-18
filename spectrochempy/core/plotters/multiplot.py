# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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


"""
Module containing multiplot function(s)

"""
import  numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.tight_layout import (get_renderer, get_tight_layout_figure,
                                     get_subplotspec_list)
from spectrochempy.core.dataset.ndio import set_figure_style

__all__ = ['multiplot', 'multiplot_map', 'multiplot_stack',
           'multiplot_image', 'multiplot_lines', 'multiplot_scatter']

_methods = __all__[:]

# .............................................................................
def multiplot_scatter(sources, **kwargs):
    """
    Plot a multiplot with 1D scatter type plots.

    Alias of multiplot (with `kind` argument set to ``scatter``.

    """
    kwargs['kind'] = 'scatter'
    multiplot(sources, **kwargs)

# .............................................................................
def multiplot_lines(sources, **kwargs):
    """
    Plot a multiplot with 1D linetype plots.

    Alias of multiplot (with `kind` argument set to ``lines``.

    """
    kwargs['kind'] = 'lines'
    multiplot(sources, **kwargs)

# .............................................................................
def multiplot_stack(sources, **kwargs):
    """
    Plot a multiplot with 2D stack type plots.

    Alias of multiplot (with `kind` argument set to ``stack``.

    """
    kwargs['kind'] = 'stack'
    multiplot(sources, **kwargs)

# .............................................................................
def multiplot_map(sources, **kwargs):
    """
    Plot a multiplot with 2D map type plots.

    Alias of multiplot (with `kind` argument set to ``map``.

    """
    kwargs['kind'] = 'map'
    multiplot(sources, **kwargs)


# .............................................................................
def multiplot_image(sources, **kwargs):
    """
    Plot a multiplot with 2D image type plots.

    Alias of multiplot (with `kind` argument set to ``image``.

    """
    kwargs['kind'] = 'image'
    multiplot(sources, **kwargs)

# .............................................................................
def multiplot( sources=[], labels=[],
                    kind='stack', nrow=1, ncol=1, figsize=None,
                    sharex=True, sharey=False,
                    pad=0.8, wpad=0, hpad=0, **kwargs):
    if len(sources) < nrow * ncol:
        # not enough sources given in this list.
        raise ValueError('Not enough sources given in this list')

    if labels and len(labels) != len(sources):
        # not enough labels given in this list.
        raise ValueError('Not enough labels given in this list')

    if nrow == ncol and nrow == 1:
        # obviously a single plot
        return source.plot_stack(figsize=figsize)

    # create the suplots
    set_figure_style(**kwargs)

    axes = subplots(nrow=nrow, ncol=ncol, figsize=figsize, sharex=sharex,
                    sharey=sharey)

    fig = plt.figure(plt.get_fignums()[-1])
    # axes is dictionary with keys such as 'axe12', where  the fist number
    # is the row and the second the column

    ylims = []

    for (axkey, ax), s, label in zip(axes.items(), sources, labels):
        s = s * 2000.
        s.plot(kind=kind, ax=ax, colorbar=False, hold=True, autolayout=False,
               **kwargs)
        ax.set_title(label, fontsize=12)
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        ylims.append(ax.get_ylim())
    axes['axe11'].yaxis.label.set_visible(True)
    axes['axe{}{}'.format(nrow, ncol)].xaxis.label.set_visible(True)
    # TODO: add a common color bar (set vmin and vmax)

    ylim = [np.min(np.array(ylims)), np.max(np.array(ylims))]
    for ax in axes.values():
        ax.set_ylim(ylim)

    # tight_layout
    renderer = get_renderer(fig)
    axeslist = list(axes.values())
    subplots_list = list(get_subplotspec_list(axeslist))
    kw = get_tight_layout_figure(fig, axeslist, subplots_list, renderer,
                                 pad=pad, h_pad=hpad, w_pad=wpad, rect=None)

    plt.subplots_adjust(left=kw['left'], bottom=kw['bottom'],
                        right=kw['right'], top=kw['top'],
                        wspace=kw.get('wspace',0), hspace=kw.get('hspace',0))

    axes['axe11'].yaxis.set_label_coords(0.03, 0.5, transform=fig.transFigure)
    axes['axe{}{}'.format(nrow, ncol)].xaxis.set_label_coords(
                           0.5, 0.08, transform=fig.transFigure)


    mpl.rcParams['figure.autolayout'] = True # restore the default param

def subplots(nrow=1, ncol=1, figsize=None, sharex=False, sharey=False):

    mpl.rcParams['figure.autolayout'] = False
    fig = plt.figure(figsize=figsize)
    axes = {}

    for i in range(nrow):
        for j in range(ncol):
            if i == j and i == 0:
                ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1)
            else:
                if sharex:
                    sharex = axes['axe11']
                else:
                    sharex = None
                if sharey:
                    sharey = axes['axe11']
                else:
                    sharey = None
                ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1,
                                     sharex=sharex, sharey=sharey)
            ax.name = 'axe{}{}'.format(i + 1, j + 1)
            axes[ax.name] = ax
            if j > 0 and sharey is not None:
                # hide the redondant ticklabels on left side of interior figures
                plt.setp(axes[ax.name].get_yticklabels(), visible=False)
            if i < nrow - 1 and sharex:
                # hide the bottom ticklabels of interior rows
                plt.setp(axes[ax.name].get_xticklabels(), visible=False)

    return axes


if __name__ == '__main__':

    from spectrochempy.api import *

    source = NDDataset.read_omnic(
         os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))

    sources=[source, source*1.1, source*1.2, source*1.3]
    labels = ['sample {}'.format(label) for label in
              ["1", "2", "3", "4"]]
    multiplot(sources=sources, kind='stack', labels=labels, nrow=2, ncol=2,
                    figsize=(9, 5), sharex=True, sharey=True)

    multiplot(sources=sources, kind='stack', labels=labels, nrow=2, ncol=2,
                    figsize=(9, 5), sharex=True, sharey=True)

    sources = [source * 1.2, source * 1.3,
               source, source * 1.1, source * 1.2, source * 1.3]
    labels = ['sample {}'.format(label) for label in
                                 ["1", "2", "3", "4", "5", "6"]]
    multiplot_map(sources=sources, labels=labels, nrow=2, ncol=3,
              figsize=(9, 5), sharex=False, sharey=False)

    multiplot_map(sources=sources, labels=labels, nrow=2, ncol=3,
              figsize=(9, 5), sharex=True, sharey=True)

    sources = [source * 1.2, source * 1.3, source, ]
    labels = ['sample {}'.format(label) for label in
              ["1", "2", "3"]]
    multiplot_stack(sources=sources, labels=labels, nrow=1, ncol=3,
                    figsize=(9, 5), sharex=True,
                    sharey=True)

    multiplot_stack(sources=sources, labels=labels, nrow=3, ncol=1,
                    figsize=(9, 5), sharex=True,
                    sharey=True)

    plt.show()