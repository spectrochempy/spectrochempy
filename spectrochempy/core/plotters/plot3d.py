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
This module should be able to handle a large set of plot types.



"""
__all__ = ['plot_3D']
_methods = __all__[:]

def plot_3D(source,
           **kwargs):
    """

    Parameters
    ----------
    source : :class:`~spectrochempy.core.ddataset.nddataset.NDDataset` to plot

    contour : `bool`
    axeslabels : None,

    transpose : False,
    step : `int` or `str` [optional, default=``all``]


    lw = None,
    color=None,
    colormap=None,
    normalize=None,
    zlim=None,
    colorbar=True,


    Returns
    -------
    fig : figure handler or None
        None is returned if something went wrong during the plotting

    """

    if source.ndim==2:
        S = source.data.copy()
        w = source.coords(-1).copy()
        t = source.coords(-2).copy()

    elif source.ndim==1:
        return source.plot_1D(**kwargs)




    # full limits of the spectra
    wlim = (w[-1], w[0])
    tlim = (t[0], t[-1])

    lab = labels[:]  # protect
    if not transpose:
        s = S.copy()
        xlim = wlim
        x = w
        y = t
    else:
        s = S.copy().T
        xlim = tlim
        x = t
        y = w
        if lab is not None:
            lab[0:2] = lab[0:2][::-1]

    if labels is not None:
        xlabel, ylabel, zlabel = lab
    else:
        xlabel = 'x'
        ylabel = 'y'
        zlabel = 'z'

    if transpose and y.size > 100:
        step = y.size // 50

    if not isinstance(step, str):
        showed = np.arange(y[0], y[-1], float(step))
        ishowed = np.searchsorted(y, showed, 'left')
    elif step == 'all':
        ishowed = slice(None)
    else:
        raise ValueError(
     'step parameter was not recognized. Should be: an int, "all"')

    s = s[:, ishowed]
    y = y[ishowed]

    # now plot the collection of lines
    if color == None:
        ax.plot(x, s, lw=lw )
    elif color != 'map':
        ax.plot(x, s, c=color, lw=lw)
    elif color == 'map':
        if not transpose:
            if zlim is not None:
                vmin, vmax = zlim
            else:
                vmin, vmax = y[0], y[-1]
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # we normalize to the max time
            if normalize is not None:
                norm.vmax = normalize
        else:
            norm = mpl.colors.Normalize(vmin=y[0], vmax=y[-1])

        line_segments = LineCollection([list(zip(x, s[:, i].tolist())) for i in xrange(len(y))][::-1],
                                    # Make a sequence of x,s[i] pairs
                                    # linewidths    = (0.5,1,1.5,2),
                                    linewidths=lw,
                                    linestyles='solid',
                                    # alpha=.5,
                         )
        line_segments.set_array(y[::-1])
        line_segments.set_cmap(colormap)
        line_segments.set_norm(norm)

        ax.add_collection(line_segments)

        if colorbar:
            fig = pl.gcf()
            axcb = fig.colorbar(line_segments, ax=ax)
            axcb.set_label(ylabel)
            axcb.set_ticks(np.linspace(int(vmin), int(vmax), 5 ))

    ax.set_xlim(xlim)
    ax.set_ylim(-.2, 3.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)
    return ax, line_segments