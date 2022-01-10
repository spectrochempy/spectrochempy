# -*- coding: utf-8 -*-

#
# ======================================================================================================================
# Copyright (©) 2015-2022 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
Plotters.
"""
__all__ = [
    "plot_2D",
    "plot_map",
    "plot_stack",
    "plot_image",
]

__dataset_methods__ = __all__

from copy import copy as cpy

from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from spectrochempy.utils import make_label, add_docstring, plot_method
from spectrochempy.core.dataset.coord import LinearCoord

_PLOT2D_DOC = """
ax : |Axes| instance. Optional
    The axe where to plot. The default is the current axe or to create a new one if is None.
clear : bool, optional, default=`True`
    Should we plot on the ax previously used or create a new figure?.
figsize : tuple, optional
    The figure size expressed as a tuple (w,h) in inch.
fontsize : int, optional
    The font size in pixels, default is 10 (or read from preferences).
style : str
autolayout : `bool`, optional, default=True
    if True, layout will be set automatically.
output : str
    A string containing a path to a filename. The output format is deduced
    from the extension of the filename. If the filename has no extension,
    the value of the rc parameter savefig.format is used.
dpi : [ None | scalar > 0]
    The resolution in dots per inch. If None it will default to the
    value savefig.dpi in the matplotlibrc file.
colorbar :
transposed :
clear :
ax :
twinx :
use_plotly : bool, optional
    Should we use plotly instead of mpl for plotting. Default to `preferences.use_plotly`  (default=False)
data_only : `bool` [optional, default=`False`]
    Only the plot is done. No addition of axes or label specifications
    (current if any or automatic settings are kept.
method : str [optional among ``map``, ``stack``, ``image`` or ``3D``]
    The type of plot,
projections : `bool` [optional, default=False]
style : str, optional, default='notebook'
    Matplotlib stylesheet (use `available_style` to get a list of available
    styles for plotting
reverse : `bool` or None [optional, default=None
    In principle, coordinates run from left to right, except for wavenumbers
    (e.g., FTIR spectra) or ppm (e.g., NMR), that spectrochempy
    will try to guess. But if reverse is set, then this is the
    setting which will be taken into account.
x_reverse : `bool` or None [optional, default=None
"""

# ======================================================================================================================
# nddataset plot2D functions
# ======================================================================================================================


@plot_method("2D", _PLOT2D_DOC)
def plot_stack(dataset, **kwargs):
    """
    Plot a 2D dataset as a stack plot.

    Alias of plot_2D (with `method` argument set to ``stack``).
    """


@plot_method("2D", _PLOT2D_DOC)
def plot_map(dataset, **kwargs):
    """
    Plot a 2D dataset as a contoured map.

    Alias of plot_2D (with `method` argument set to ``map``.
    """


@plot_method("2D", _PLOT2D_DOC)
def plot_image(dataset, **kwargs):
    """
    Plot a 2D dataset as an image plot.

    Alias of plot_2D (with `method` argument set to ``image``).
    """


@add_docstring(_PLOT2D_DOC)
def plot_2D(dataset, method=None, **kwargs):
    """
    Plot of 2D array.

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to plot.
    method : ['stack', 'map', 'image'] , optional
        The method of plot of the dataset, which will determine the plotter to use.
        Default method is given 'stack' but this can be changed using
        ``dataset.preference.method_2D``.
    **kwargs : dic, optional
        Additional keywords parameters.
        See Other Parameters.

    Other Parameters
    ----------------
    {0}

    See Also
    --------
    plot_map
    plot_stack
    plot_image
    plot_surface
    plot_waterfall
    """

    # Get preferences
    # ------------------------------------------------------------------------

    prefs = dataset.preferences

    # before going further, check if the style is passed in the parameters
    style = kwargs.pop("style", None)
    if style is not None:
        prefs.style = style
    # else we assume this has been set before calling plot()

    prefs.set_latex_font(prefs.font.family)  # reset latex settings

    # Redirections ?
    # ------------------------------------------------------------------------

    # should we redirect the plotting to another method
    if dataset._squeeze_ndim < 2:
        return dataset.plot_1D(**kwargs)

    # if plotly execute plotly routine not this one
    if kwargs.get("use_plotly", prefs.use_plotly):
        return dataset.plotly(**kwargs)

    # do not display colorbar if it's not a surface plot
    # except if we have asked to d so

    # often we do need to plot only data when plotting on top of a previous plot
    data_only = kwargs.get("data_only", False)

    # Get the data to plot
    # ---------------------------------------------------------------

    # if we want to plot the transposed dataset
    transposed = kwargs.get("transposed", False)
    if transposed:
        new = dataset.copy().T  # transpose dataset
        nameadd = ".T"
    else:
        new = dataset  # .copy()
        nameadd = ""
    new = new.squeeze()

    if kwargs.get("y_reverse", False):
        new = new[::-1]

    # Figure setup
    # ------------------------------------------------------------------------
    method = new._figure_setup(ndim=2, method=method, **kwargs)

    ax = new.ndaxes["main"]
    ax.name = ax.name + nameadd

    # Other properties that can be passed as arguments
    # ------------------------------------------------------------------------

    lw = kwargs.get("linewidth", kwargs.get("lw", prefs.lines_linewidth))
    alpha = kwargs.get("calpha", prefs.contour_alpha)

    number_x_labels = prefs.number_of_x_labels
    number_y_labels = prefs.number_of_y_labels
    number_z_labels = prefs.number_of_z_labels

    if method in ["waterfall"]:
        nxl = number_x_labels * 2
        nyl = number_z_labels * 2
    elif method in ["stack"]:
        nxl = number_x_labels
        nyl = number_z_labels
    else:
        nxl = number_x_labels
        nyl = number_y_labels

    ax.xaxis.set_major_locator(MaxNLocator(nbins=nxl))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nyl))
    if method not in ["surface"]:
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

    # the next lines are to avoid multipliers in axis scale
    formatter = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # ------------------------------------------------------------------------
    # Set axis
    # ------------------------------------------------------------------------

    # set the abscissa axis
    # ------------------------------------------------------------------------
    # the actual dimension name is the last in the new.dims list
    dimx = new.dims[-1]
    x = getattr(new, dimx)
    if x is not None and x.implements("CoordSet"):
        # if several coords, take the default ones:
        x = x.default
    xsize = new.shape[-1]
    show_x_points = False
    if x is not None and hasattr(x, "show_datapoints"):
        show_x_points = x.show_datapoints
    if show_x_points:
        # remove data and units for display
        x = LinearCoord.arange(xsize)

    discrete_data = False

    if x is not None and (not x.is_empty or x.is_labeled):
        xdata = x.data
        if not np.any(xdata):
            if x.is_labeled:
                discrete_data = True
                # take into account the fact that sometimes axis have just labels
                xdata = range(1, len(x.labels) + 1)
    else:
        xdata = range(xsize)

    xl = [xdata[0], xdata[-1]]
    xl.sort()

    if xsize < number_x_labels + 1:
        # extend the axis so that the labels are not too close to the limits
        inc = abs(xdata[1] - xdata[0]) * 0.5
        xl = [xl[0] - inc, xl[1] + inc]

    if data_only:
        xl = ax.get_xlim()

    xlim = list(kwargs.get("xlim", xl))
    xlim.sort()
    xlim[-1] = min(xlim[-1], xl[-1])
    xlim[0] = max(xlim[0], xl[0])

    if kwargs.get("x_reverse", kwargs.get("reverse", x.reversed if x else False)):
        xlim.reverse()

    ax.set_xlim(xlim)

    xscale = kwargs.get("xscale", "linear")
    ax.set_xscale(xscale)  # , nonpositive='mask')

    # set the ordinates axis
    # ------------------------------------------------------------------------
    # the actual dimension name is the second in the new.dims list
    dimy = new.dims[-2]
    y = getattr(new, dimy)
    if y is not None and y.implements("CoordSet"):
        # if several coords, take the default ones:
        y = y.default
    ysize = new.shape[-2]

    show_y_points = False
    if y is not None and hasattr(y, "show_datapoints"):
        show_y_points = y.show_datapoints
    if show_y_points:
        # remove data and units for display
        y = LinearCoord.arange(ysize)

    if y is not None and (not y.is_empty or y.is_labeled):
        ydata = y.data

        if not np.any(ydata):
            if y.is_labeled:
                ydata = range(1, len(y.labels) + 1)
    else:
        ydata = range(ysize)

    yl = [ydata[0], ydata[-1]]
    yl.sort()

    if ysize < number_y_labels + 1:
        # extend the axis so that the labels are not too close to the limits
        inc = abs(ydata[1] - ydata[0]) * 0.5
        yl = [yl[0] - inc, yl[1] + inc]

    if data_only:
        yl = ax.get_ylim()

    ylim = list(kwargs.get("ylim", yl))
    ylim.sort()
    ylim[-1] = min(ylim[-1], yl[-1])
    ylim[0] = max(ylim[0], yl[0])

    yscale = kwargs.get("yscale", "linear")
    ax.set_yscale(yscale)

    # z intensity (by default we plot real component of the data)
    # ------------------------------------------------------------------------

    if not kwargs.get("imag", False):
        zdata = new.real.masked_data
    else:
        zdata = (
            new.RI.masked_data
        )  # new.imag.masked_data #TODO: quaternion case (3 imag.components)

    zlim = kwargs.get("zlim", (np.ma.min(zdata), np.ma.max(zdata)))

    if method in ["stack", "waterfall"]:

        # the z axis info
        # ---------------
        # zl = (np.min(np.ma.min(ys)), np.max(np.ma.max(ys)))
        amp = 0  # np.ma.ptp(zdata) / 50.
        zl = (np.min(np.ma.min(zdata) - amp), np.max(np.ma.max(zdata)) + amp)
        zlim = list(kwargs.get("zlim", zl))
        zlim.sort()
        z_reverse = kwargs.get("z_reverse", False)
        if z_reverse:
            zlim.reverse()

        # set the limits
        # ---------------

        if yscale == "log" and min(zlim) <= 0:
            # set the limits wrt smallest and largest strictly positive values
            ax.set_ylim(
                10 ** (int(np.log10(np.amin(np.abs(zdata)))) - 1),
                10 ** (int(np.log10(np.amax(np.abs(zdata)))) + 1),
            )
        else:
            ax.set_ylim(zlim)

    else:

        # the y axis info
        # ----------------
        if data_only:
            ylim = ax.get_ylim()

        ylim = list(kwargs.get("ylim", ylim))
        ylim.sort()
        y_reverse = kwargs.get("y_reverse", y.reversed if y else False)
        if y_reverse:
            ylim.reverse()

        # set the limits
        # ----------------
        ax.set_ylim(ylim)

    # ------------------------------------------------------------------------
    # plot the dataset
    # ------------------------------------------------------------------------
    ax.grid(prefs.axes_grid)

    normalize = kwargs.get("normalize", None)
    cmap = kwargs.get("colormap", kwargs.get("cmap", prefs.colormap))

    if method in ["map", "image", "surface"]:
        zmin, zmax = zlim
        zmin = min(zmin, -zmax)
        zmax = max(-zmin, zmax)
        norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

    if method in ["surface"]:
        X, Y = np.meshgrid(xdata, ydata)
        Z = zdata.copy()

        # masker data not taken into account in surface plot
        Z[dataset.mask] = np.nan

        # Plot the surface.  #TODO : improve this (or remove it)

        antialiased = kwargs.get("antialiased", prefs.antialiased)
        rcount = kwargs.get("rcount", prefs.rcount)
        ccount = kwargs.get("ccount", prefs.ccount)
        ax.set_facecolor("w")
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            linewidth=lw,
            antialiased=antialiased,
            rcount=rcount,
            ccount=ccount,
            edgecolor="k",
            norm=norm,
        )

    if method in ["waterfall"]:
        _plot_waterfall(ax, new, xdata, ydata, zdata, prefs, xlim, ylim, zlim, **kwargs)

    elif method in ["image"]:

        cmap = kwargs.get("cmap", kwargs.get("image_cmap", prefs.image_cmap))
        if discrete_data:
            method = "map"

        else:
            kwargs["nlevels"] = 500
            if not hasattr(new, "clevels") or new.clevels is None:
                new.clevels = _get_clevels(zdata, prefs, **kwargs)
            c = ax.contourf(xdata, ydata, zdata, new.clevels, alpha=alpha)
            c.set_cmap(cmap)
            c.set_norm(norm)

    elif method in ["map"]:
        if discrete_data:

            _colormap = plt.get_cmap(cmap)
            scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=_colormap)

            # marker = kwargs.get('marker', kwargs.get('m', None))
            markersize = kwargs.get("markersize", kwargs.get("ms", 5.0))
            # markevery = kwargs.get('markevery', kwargs.get('me', 1))

            for i in ydata:
                for j in xdata:
                    (li,) = ax.plot(j, i, lw=lw, marker="o", markersize=markersize)
                    li.set_color(scalarMap.to_rgba(zdata[i - 1, j - 1]))

        else:
            # contour plot
            # -------------
            if not hasattr(new, "clevels") or new.clevels is None:
                new.clevels = _get_clevels(zdata, prefs, **kwargs)

            c = ax.contour(xdata, ydata, zdata, new.clevels, linewidths=lw, alpha=alpha)
            c.set_cmap(cmap)
            c.set_norm(norm)

    elif method in ["stack"]:

        # stack plot
        # ----------

        # now plot the collection of lines
        # --------------------------------
        # map colors using the colormap

        vmin, vmax = ylim
        norm = mpl.colors.Normalize(
            vmin=vmin, vmax=vmax
        )  # we normalize to the max time
        if normalize is not None:
            norm.vmax = normalize

        _colormap = plt.get_cmap(cmap)
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=_colormap)

        # we display the line in the reverse order, so that the last
        # are behind the first.

        clear = kwargs.get("clear", True)
        lines = []
        if not clear and not transposed:
            lines.extend(ax.lines)  # keep the old lines

        line0 = mpl.lines.Line2D(xdata, zdata[0], lw=lw, picker=True)

        for i in range(zdata.shape[0]):
            li = cpy(line0)
            li.set_ydata(zdata[i])
            lines.append(li)
            li.set_color(scalarMap.to_rgba(ydata[i]))
            fmt = kwargs.get("label_fmt", "{:.5f}")
            li.set_label(fmt.format(ydata[i]))
            li.set_zorder(zdata.shape[0] + 1 - i)

        # store the full set of lines
        new._ax_lines = lines[:]

        # but display only a subset of them in order to accelerate the drawing
        maxlines = kwargs.get("maxlines", prefs.max_lines_in_stack)
        setpy = max(len(new._ax_lines) // maxlines, 1)

        for line in new._ax_lines[::setpy]:
            ax.add_line(line)

    if data_only or method in ["waterfall"]:
        # if data only (we will not set axes and labels
        # it was probably done already in a previous plot
        new._plot_resume(dataset, **kwargs)
        return ax

    # display a title
    # ------------------------------------------------------------------------
    title = kwargs.get("title", None)
    if title:
        ax.set_title(title)
    elif kwargs.get("plottitle", False):
        ax.set_title(new.name)

    # ------------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------------

    # x label
    # ------------------------------------------------------------------------
    xlabel = kwargs.get("xlabel", None)
    if show_x_points:
        xlabel = "data points"
    if not xlabel:
        xlabel = make_label(x, new.dims[-1])
    ax.set_xlabel(xlabel)

    uselabelx = kwargs.get("uselabel_x", False)
    if (
        x
        and x.is_labeled
        and (uselabelx or not np.any(x.data))
        and len(x.labels) < number_x_labels + 1
    ):
        # TODO refine this to use different orders of labels
        ax.set_xticks(xdata)
        ax.set_xticklabels(x.labels)

    # y label
    # ------------------------------------------------------------------------
    ylabel = kwargs.get("ylabel", None)
    if show_y_points:
        ylabel = "data points"
    if not ylabel:
        if method in ["stack"]:
            ylabel = make_label(new, "values")

        else:
            ylabel = make_label(y, new.dims[-2])
            # y tick labels
            uselabely = kwargs.get("uselabel_y", False)
            if (
                y
                and y.is_labeled
                and (uselabely or not np.any(y.data))
                and len(y.labels) < number_y_labels
            ):
                # TODO refine this to use different orders of labels
                ax.set_yticks(ydata)
                ax.set_yticklabels(y.labels)

    # z label
    # ------------------------------------------------------------------------
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        if method in ["stack"]:
            zlabel = make_label(y, new.dims[-2])
        elif method in ["surface"]:
            zlabel = make_label(new, "values")
            ax.set_zlabel(zlabel)
        else:
            zlabel = make_label(new, "z")

    # do we display the ordinate axis?
    if kwargs.get("show_y", True):
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticks([])

    if "colorbar" in new.ndaxes:
        if "surface" not in method and (not hasattr(new, "_axcb") or not new._axcb):
            axec = new.ndaxes["colorbar"]
            axec.name = axec.name + nameadd
            new._axcb = mpl.colorbar.ColorbarBase(
                axec, cmap=plt.get_cmap(cmap), norm=norm
            )
            new._axcb.set_label(zlabel)
    #        else:
    #            new._fig.colorbar(surf, shrink=0.5, aspect=10)

    # do we display the zero line
    if kwargs.get("show_zero", False):
        ax.haxlines()

    new._plot_resume(dataset, **kwargs)

    return ax


# ======================================================================================================================
# Waterfall
# ======================================================================================================================


def _plot_waterfall(ax, new, xdata, ydata, zdata, prefs, xlim, ylim, zlim, **kwargs):
    degazim = kwargs.get("azim", 10)
    degelev = kwargs.get("elev", 30)

    azim = np.deg2rad(degazim)
    elev = np.deg2rad(degelev)

    # transformation function Axes coordinates to Data coordinates
    def transA2D(x_, y_):
        return ax.transData.inverted().transform(ax.transAxes.transform((x_, y_)))

    # expansion in Axes coordinates
    xe, ze = np.sin(azim), np.sin(elev)

    incx, incz = transA2D(1 + xe, 1 + ze) - np.array((xlim[-1], zlim[-1]))

    def fx(y_):
        return (y_ - ydata[0]) * incx / (ydata[-1] - ydata[0])

    def fz(y_):
        return (y_ - ydata[0]) * incz / (ydata[-1] - ydata[0])

    zs = incz * 0.05
    base = zdata.min() - zs

    for i, row in enumerate(zdata):
        y = ydata[i]
        x = xdata + fx(y)
        z = row + fz(y)  # row.masked_data[0]
        ma = z.max()
        z2 = base + fz(y)
        line = mpl.lines.Line2D(x, z, color="k")
        line.set_label(f"{ydata[i]}")
        line.set_zorder(row.size + 1 - i)
        poly = plt.fill_between(
            x,
            z,
            z2,
            alpha=1,
            facecolors="w",
            edgecolors="0.85" if 0 < i < ydata.size - 1 else "k",
        )
        poly.set_zorder(row.size + 1 - i)
        try:
            ax.add_collection(poly)
        except ValueError:  # strange error with tests
            pass
        ax.add_line(line)

    (x0, y0), (x1, _) = transA2D(0, 0), transA2D(1 + xe + 0.15, 1 + ze)
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0 - zs - 0.05, ma * 1.1))

    ax.set_facecolor("w")
    ax.vlines(
        x=xdata[-1] + incx,
        ymin=zdata.min() - zs + incz,
        ymax=ax.get_ylim()[-1],
        color="k",
    )
    ax.vlines(
        x=xdata[0] + incx,
        ymin=zdata.min() - zs + incz,
        ymax=ax.get_ylim()[-1],
        color="k",
    )
    ax.vlines(
        x=xdata[0], ymin=y0 - zs, ymax=ax.get_ylim()[-1] - incz, color="k", zorder=5000
    )
    ax.vlines(
        x=xdata[0], ymin=y0 - zs, ymax=ax.get_ylim()[-1] - incz, color="k", zorder=5000
    )

    x = [xdata[0], xdata[0] + incx, xdata[-1] + incx]
    z = [ax.get_ylim()[-1] - incz, ax.get_ylim()[-1], ax.get_ylim()[-1]]
    x2 = [xdata[0], xdata[-1], xdata[-1] + incx]
    z2 = [y0 - zs, y0 - zs, y0 - zs + incz]
    poly = plt.fill_between(x, z, z2, alpha=1, facecolors=".95", edgecolors="w")
    try:
        ax.add_collection(poly)
    except ValueError:
        pass
    poly = plt.fill_between(x2, z, z2, alpha=1, facecolors=".95", edgecolors="w")
    try:
        ax.add_collection(poly)
    except ValueError:
        pass
    line = mpl.lines.Line2D(x, np.array(z), color="k", zorder=50000)
    ax.add_line(line)
    line = mpl.lines.Line2D(x2, np.array(z2), color="k", zorder=50000)
    ax.add_line(line)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # xticks (xaxis)
    ticks = ax.get_xticks()
    newticks = []
    xt = sorted(xlim)
    for tick in ticks:
        if xt[0] <= tick <= xt[1]:
            newticks.append(tick)
    ax.set_xticks(newticks)

    # yticks (zaxis)
    ticks = ax.get_yticks()
    newticks = []
    zt = [y0, ax.get_ylim()[-1] - incz]
    for tick in ticks:
        if zt[0] <= tick <= zt[1]:
            newticks.append(tick)
    _ = ax.set_yticks(newticks)

    # make yaxis
    def ctx(x_):
        return (
            ax.transData.inverted().transform((x_, 0))
            - ax.transData.inverted().transform((0, 0))
        )[0]

    yt = [y for y in np.linspace(ylim[0], ylim[-1], 5)]
    for y in yt:
        xmin = xdata[-1] + fx(y)
        xmax = xdata[-1] + fx(y) + ctx(3.5)
        pos = y0 - zs + fz(y)
        ax.hlines(pos, xmin, xmax, zorder=50000)
        lab = ax.text(xmax + ctx(8), pos, f"{y:.0f}", va="center")

    # display a title
    # ------------------------------------------------------------------------
    title = kwargs.get("title", None)
    if title:
        ax.set_title(title)

    # ------------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------------

    # x label
    # ------------------------------------------------------------------------
    xlabel = kwargs.get("xlabel", None)
    if not xlabel:
        xlabel = make_label(new.x, "x")
    ax.set_xlabel(xlabel, x=(ax.bbox._bbox.x0 + ax.bbox._bbox.x1) / 2 - xe)

    # y label
    # ------------------------------------------------------------------------
    ylabel = kwargs.get("ylabel", None)
    if not ylabel:
        ylabel = make_label(new.y, "y")
    ym = (ylim[0] + ylim[1]) / 2
    x = xdata[-1] + fx(ym)
    z = y0 - zs + fz(ym)
    offset = prefs.font.size * (len(lab._text)) + 30
    iz = ax.transData.transform((0, incz + z))[1] - ax.transData.transform((0, z))[1]
    ix = ax.transData.transform((incx + x, 0))[0] - ax.transData.transform((x, 0))[0]
    angle = np.rad2deg(np.arctan(iz / ix))
    ax.annotate(
        ylabel,
        (x, z),
        xytext=(offset, 0),
        xycoords="data",
        textcoords="offset pixels",
        ha="center",
        va="center",
        rotation=angle,
    )

    # z label
    # ------------------------------------------------------------------------
    zlabel = kwargs.get("zlabel", None)
    if not zlabel:
        zlabel = make_label(new, "value")

    # do we display the z axis?
    if kwargs.get("show_z", True):
        ax.set_ylabel(zlabel, y=(ax.bbox._bbox.y0 + 1 - ze) / 2)
    else:
        ax.set_yticks([])


# ======================================================================================================================
# get clevels
# ======================================================================================================================


def _get_clevels(data, prefs, **kwargs):
    # Utility function to determine contours levels

    # contours
    maximum = data.max()

    # minimum = -maximum

    nlevels = kwargs.get("nlevels", kwargs.get("nc", prefs.number_of_contours))
    start = kwargs.get("start", prefs.contour_start) * maximum
    negative = kwargs.get("negative", True)
    if negative < 0:
        negative = True

    c = np.arange(nlevels)
    cl = np.log(c + 1.0)
    clevel = cl * (maximum - start) / cl.max() + start
    clevelneg = -clevel
    clevelc = clevel
    if negative:
        clevelc = sorted(list(np.concatenate((clevel, clevelneg))))

    return clevelc


if __name__ == "__main__":
    pass
