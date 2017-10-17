
# TODO: This is an old file from another project - need to adapt some of the technique to scp

import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt

#TODO : plt.style.use('agir_default')

# ******************************************************************************
# PLOTTING
# ******************************************************************************

def drawabs(ax, x, y, dy, head_width = 5, xshift=10 ):
    ax.arrow(x, y, 0, dy, head_width=head_width, head_length=dy/5.,
                  length_includes_head=True, fc='k', ec='k')
    ax.arrow(x, y+dy, 0, -dy, head_width=head_width, head_length=dy/5.,
                  length_includes_head=True, fc='k', ec='k')
    txt = "%.2f a.u." % (dy)
    ax.text(x-xshift, (2.*y+dy)/2., txt, va='center', fontsize=6)


def plotregions(ax, data, regions=[], inc=20):
    _nr, nc = data.shape
    incr = max(int(nc // inc), 1)
    w = data.index.values
    S = data.copy().values
    # reducing
    S = S[:, ::incr]
    p = ax.plot(w, S)
    colors = []
    for i, l in enumerate(p):
        l.set_label(data.columns[i])
        colors.append(l.get_color())
    if regions:
        for item in regions:
            wmin, wmax = sorted(item)
            ax.axvspan(wmin, wmax, facecolor='lightgreen', alpha=0.3)
    ax.set_xlim(w[-1], w[0])
    return colors

def plots(ax, specs, t=None, w=None,
             labels=['wavenumbers (cm$^{-1}$)', 'TOS (h)', 'absorbance (a.u.)'],
             transpose=False,
             step='all',
             lw = 1.,
             color='map',
             colormap='jet',
             normalize=None,
             zlim=None,
             colorbar=True):

    if isinstance(specs, pd.DataFrame):
        # prepare data to plot
        S = specs.values.copy()
        w = specs.index.values.copy()
        t = specs.columns.values.copy()

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

def plot_param(fig, fignumb, sharey=None, sharex=None):
    #print int(fignumb)
    ax = fig.add_subplot(int(fignumb), sharex=sharex, sharey=sharey)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    return ax

def plotdata(SL,
             colormap='jet',
             plot_transposed=False,
             xlim=None,
             ylim=None,
             zlim=None,
             lw = 0.5,
             boldlines=None,
             interrupted=None,
             #width = 3.4,
             #height = 3.4*0.75,
             horizontal=False,
             figsize = None,
             middle_ticks=True,
             barshow=True,
             step='all',
             xshared=True,
             yshared=True,
             yticks = True,
             title=None,
             titlecolor = 'k',
             hspace=0,
             wspace=0,
             bottom = 0.24,
             top=0.95,
             left = 0.08,
             right=0.97,
             ):

    row = 1
    col = 1
    axes = []
    if isinstance(SL, list):
        # several plots
        if horizontal:
            row=1
            col=len(SL)
        else:
            row = len(SL)
        if not isinstance(title, list):
            title = [title] * row
        if not isinstance(titlecolor, list):
            titlecolor = [titlecolor] * row
    else:
        SL = [SL, ]
        title = [title, ]
        titlecolor = [titlecolor, ]

    if interrupted:
        # we limit to only one plot in this case
        if len(SL) > 1:
            print("ERROR: limited to one plot (not transposed)")
            return None, None
        plot_transposed = False
        row = 1
        col = 2

    if plot_transposed:
        col = 2

    if figsize:
        fig = pl.figure(figsize=figsize)
    else:
        fig = pl.figure()

    for idx, S in enumerate(SL):

        if xlim is None and S is not None:
            xlim_ = (min(S.index), max(S.index))[::-1]
            if interrupted:
                xlim2_ = (min(S.index), interrupted[0])[::-1]
                xlim_ = (interrupted[-1], max(S.index))[::-1]
        else:
            xlim_=xlim

        if ylim is None and S is not None:
            ylim_ = (S.values.min(), S.values.max())
        else:
            ylim_ = ylim

        if not(xshared or yshared) or idx==0:
            ax = plot_param(fig, "%d%d%d" % (row, col, (idx + 1)))
        elif xshared and not yshared:
            ax = plot_param(fig, "%d%d%d" % (row, col, (idx + 1)), sharex=axes[0])
        elif yshared and not xshared:
            ax = plot_param(fig, "%d%d%d" % (row, col, (idx + 1)), sharey=axes[0])
        else:
            ax = plot_param(fig, "%d%d%d" % (row, col, (idx + 1)),
                            sharex=axes[0], sharey=axes[0])


        if interrupted:
            ax2 = plot_param(fig, "%d%d%d" % (row, col, (idx * 2 + 2)), sharey=ax)
            # Make the spacing between the two axes a bit smaller
            wspace=0.05
            axes.append([ax,ax2])

        if interrupted:
            plots(ax, S, zlim=zlim, colormap=colormap, lw=lw, colorbar=False, step=step)
            plots(ax2, S,  lw=lw, zlim=zlim, colormap=colormap, step=step)

        else:
            axes.append(ax)
            showbar = barshow
            if row==3:
                showbar = False
            if not middle_ticks:
                if idx==1:
                    if S is  None:
                        ax.axis('off')
                        plots(ax, SL[idx-1], color='w', lw=lw,
                              zlim=zlim, colorbar=barshow, step=step)

                    else:
                        plots(ax, S, colormap=colormap,  zlim=zlim,
                              lw=lw, colorbar=barshow, step=step)
                    ax.set_yticklabels([])
                    ax.set_ylabel('')

                else:
                    plots(ax, S, colormap=colormap,  lw=lw,
                          zlim=zlim, colorbar=False, step=step)
            else:
                if idx==1 and S is  None:
                    ax.axis('off')
                    plots(ax, SL[idx-1], color='w',  lw=lw,
                          zlim=zlim, colorbar=barshow, step=step)
                else:
                    ax, lseg = plots(ax, S, colormap=colormap,  lw=lw,
                                     zlim=zlim, colorbar=showbar, step=step)

        if boldlines is not None:
            if isinstance(boldlines, int):
                plots(ax, S, color='k',  lw=lw, step=boldlines, zlim=zlim, )
                if interrupted:
                    plots(ax2, S, color='k',
                          lw=lw, step=boldlines, zlim=zlim, )

        if not interrupted and not plot_transposed:
            if title[idx]:
                # ax.set_title(title[idx])
                ax.text(0.97,0.89,
                        title[idx],
                        color=titlecolor[idx],
                        verticalalignment='top',
                        horizontalalignment='right',
                        transform=ax.transAxes
                        )
        else:
            if title[idx]:
                fig.suptitle(title[idx])

        ax.set_ylim(ylim_)
        ax.set_xlim(xlim_)
        if interrupted:
            # zoom-in / limit the view to different portions of the data
            ax2.set_xlim(xlim2_)

            # hide the spines between ax and ax2
            ax.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax.yaxis.tick_left()
            ax.tick_params(labeltop='off')  # don't put tick labels at the top
            ax2.yaxis.tick_right()
            ax2.tick_params(labelright='off')
            ax2.set_ylabel('')
            ax.set_xlabel('')
            ax2.set_xlabel('')
            fig.text(0.43, 0.03, 'wavenumbers (cm$^{-1}$)')

        if plot_transposed:
            axt = plot_param(fig, "%d%d%d" % (row, col, (idx * 2 + 2)))
            plots(axt, S,  lw=lw, transpose=True, zlim=zlim, )
            #axt.set_title(title[idx])

            if ylim_ is not None:
                axt.set_ylim(ylim_)
            else:
                axt.set_ylim((S.values.min(), S.values.max()))

        if not yticks:
            ax.set_yticks([])
        # case of three plots (denoised)
        if row==3:
            if idx==0:
                axes[0].set_ylabel('')
            if idx==2:
                axes[2].set_ylabel('')

        if idx<len(SL)-1:
            axes[idx].set_xticklabels([])


    if row==3 and barshow:

        axins = inset_axes(axes[1],
                   width="2.5%", # width = 10% of parent_bbox width
                   height="150%", # height : 50%
                   loc=3,
                   bbox_to_anchor=(1.05, -0.25, 1, 1),
                   bbox_transform=axes[1].transAxes,
                   borderpad=0,
                   )
        axcb = pl.colorbar(lseg, cax=axins)
        if zlim:
            vmin, vmax = zlim
            axcb.set_ticks(np.linspace(int(vmin), int(vmax), 5 ))
        axcb.set_label('TOS (h)')


    pl.subplots_adjust(
                    hspace=hspace, wspace=wspace,
                    bottom=bottom, top=top,
                    left=left, right=right)

    if plot_transposed:
        return fig, ax, axt
    else:
        return fig, axes
