# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
NDPlot Module.

This module implements plotting capabilities for NDDataset objects, providing:
- 1D, 2D and 3D plotting methods
- Figure and axes management
- Interactive visualization tools

The module defines the NDPlot base class which handles all plotting functionality
through a unified interface.

Classes
-------
NDPlot
    Main class providing plotting methods and figure management

Notes
-----
Currently supports:
- Matplotlib backend (primary)
- Plotly backend (experimental)

Handles units and coordinates from NDDatasets automatically.
"""

__all__ = ["plot"]

from typing import Any

import matplotlib as mpl
import traitlets as tr
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spectrochempy.application.application import error_
from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.utils.docutils import docprocess
from spectrochempy.utils.mplutils import _Axes
from spectrochempy.utils.mplutils import _Axes3D
from spectrochempy.utils.mplutils import get_figure
from spectrochempy.utils.optional import import_optional_dependency

go = import_optional_dependency("plotly.graph_objects", errors="ignore")
HAS_PLOTLY = go is not None


class NDPlot(tr.HasTraits):
    """
    Base class providing plotting capabilities for NDDataset objects.

    This class implements methods for creating and managing plots of NDDataset data
    in 1D, 2D and 3D. It handles figure creation, axes management, and plot styling.

    Attributes
    ----------
    _ax : _Axes
        Main plotting axes
    _fig : Union[plt.Figure, go.Figure]
        Figure object (matplotlib or plotly)
    _ndaxes : Dict[str, _Axes]
        Dictionary mapping axes names to axes objects

    Methods
    -------
    plot(method=None, **kwargs)
        Main plotting interface
    close_figure()
        Close the current figure
    """

    # Trait definitions
    _ax = tr.Instance(_Axes, allow_none=True)
    _fig = (
        tr.Union((tr.Instance(plt.Figure), tr.Instance(go.Figure)), allow_none=True)
        if HAS_PLOTLY
        else tr.Instance(plt.Figure, allow_none=True)
    )
    _ndaxes = tr.Dict(tr.Instance(_Axes))

    @docprocess.get_sections(
        base="plot",
        sections=["Parameters", "Other Parameters", "Returns"],
    )
    @docprocess.dedent
    def plot(self, method: str | None = None, **kwargs: Any) -> _Axes | None:
        """
        Plot the dataset using the specified method.

        Parameters
        ----------
        dataset : :class:`~spectrochempy.ddataset.nddataset.NDDataset`
            Source of data to plot.
        method : str, optional, default: `preference.method_1D` or `preference.method_2D`
            Name of plotting method to use. If None, method is chosen based on data
            dimensionality.

            1D plotting methods:

            - `pen` : Solid line plot
            - `bar` : Bar graph
            - `scatter` : Scatter plot
            - `scatter+pen` : Scatter plot with solid line

            2D plotting methods:

            - `stack` : Stacked plot
            - `map` : Contour plot
            - `image` : Image plot
            - `surface` : Surface plot
            - `waterfall` : Waterfall plot

        %(kwargs)s

        Other Parameters
        ----------------
        ax : Axe, optional
            Axe where to plot. If not specified, create a new one.
        clear : bool, optional, default: True
            If false, hold the current figure and ax until a new plot is performed.
        color or c : color, optional, default: auto
            color of the line.
        colorbar : bool, optional, default: True
            Show colorbar  (2D plots only).
        commands : str,
            matplotlib commands to be executed.
        data_only : bool, optional, default: False
            Only the plot is done. No addition of axes or label specifications.
        dpi : int, optional
            the number of pixel per inches.
        figsize : tuple, optional, default is (3.4, 1.7)
            figure size.
        fontsize : int, optional
            The font size in pixels, default is 10 (or read from preferences).
        imag : bool, optional, default: False
            Show imaginary component for complex data. By default the real component is
            displayed.
        linestyle or ls : str, optional, default: auto
            line style definition.
        linewidth or lw : float, optional, default: auto
            line width.
        marker, m: str, optional, default: auto
            marker type for scatter plot. If marker != "" then the scatter type of plot is chosen automatically.
        markeredgecolor or mec: color, optional
        markeredgewidth or mew: float, optional
        markerfacecolor or mfc: color, optional
        markersize or ms: float, optional
        markevery: None or int
        modellinestyle or modls : str
            line style of the model.
        offset : float
            offset of the model individual lines.
        output : str,
            name of the file to save the figure.
        plot_model : Bool,
            plot model data if available.
        plottitle: bool, optional, default: False
            Use the name of the dataset as title. Works only if title is not defined
        projections : bool, optional, default: False
            Show projections on the axes (2D plots only).
        reverse : bool or None [optional, default=None/False
            In principle, coordinates run from left to right,
            except for wavenumbers
            (*e.g.*, FTIR spectra) or ppm (*e.g.*, NMR), that spectrochempy
            will try to guess. But if reverse is set, then this is the
            setting which will be taken into account.
        show_complex : bool, optional, default: False
            Show both real and imaginary component for complex data.
            By default only the real component is displayed.
        show_mask: bool, optional
            Should we display the mask using colored area.
        show_z : bool, optional, default: True
            should we show the vertical axis.
        show_zero : bool, optional
            show the zero basis.
        style : str, optional, default: `scp.preferences.style` (scpy)
            Matplotlib stylesheet (use `available_style` to get a list of available
            styles for plotting.
        title : str
            Title of the plot (or subplot) axe.
        transposed : bool, optional, default: False
            Transpose the data before plotting (2D plots only).
        twinx : :class:`~matplotlib.axes.Axes` instance, optional, default: None
            If this is not None, then a twin axes will be created with a
            common x dimension.
        uselabel_x: bool, optional
            use x coordinate label as x tick labels
        vshift : float, optional
            vertically shift the line from its baseline.
        xlim : tuple, optional
            limit on the horizontal axis.
        xlabel : str, optional
            label on the horizontal axis.
        x_reverse : bool, optional, default: False
            reverse the x axis. Equivalent to `reverse`.
        ylabel or zlabel : str, optional
            label on the vertical axis.
        ylim or zlim : tuple, optional
            limit on the vertical axis.
        y_reverse : bool, optional, default: False
            reverse the y axis (2D plot only).

        Returns
        -------
        Matplolib Axes or None
            The matplotlib axes containing the plot if successful, None otherwise.

        """
        # Select appropriate plotting method
        if method:
            _plotter = getattr(self, f"plot_{method.replace('+', '_')}", None)
            if _plotter is None:
                error_(
                    NameError,
                    f"The specified plotter for method `{method}` was not found!",
                )
                raise OSError
        else:
            _plotter = self._plot_generic

        return _plotter(**kwargs)

    def _plot_generic(self, **kwargs: Any) -> _Axes | None:
        # Choose plotting method based on dataset dimensionality
        # Args:
        #    **kwargs: Plot options
        # Returns:
        #    Matplotlib axes if successful

        from spectrochempy.core.plotters.plot1d import plot_1D
        from spectrochempy.core.plotters.plot2d import plot_2D
        from spectrochempy.core.plotters.plot3d import plot_3D

        if self._squeeze_ndim == 1:
            ax = plot_1D(self, **kwargs)

        elif self._squeeze_ndim == 2:
            ax = plot_2D(self, **kwargs)

        elif self._squeeze_ndim == 3:
            ax = plot_3D(self, **kwargs)

        else:
            error_(Exception, "Cannot guess an adequate plotter, nothing done!")
            return False

        return ax

    def close_figure(self):
        """Close a Matplotlib figure associated to this dataset."""
        if self._fig is not None:
            plt.close(self._fig)

    def _figure_setup(
        self,
        ndim: int = 1,
        method: str | None = None,
        **kwargs: Any,
    ) -> str:
        # Set up figure and axes for plotting
        # Args:
        #    ndim: Number of dimensions to plot
        #    method: Plot method to use
        #    **kwargs: Additional options
        # Returns:
        #    Method name to use for plotting
        if not method:
            method = prefs.method_2D if ndim == 2 else prefs.method_1D

        ax3d = method in ["surface"]

        # Get current figure information
        # ------------------------------
        # should we use the previous figure?
        clear = kwargs.get("clear", True)

        # is ax in the keywords ?
        ax = kwargs.pop("ax", None)

        # is it a twin figure? In such case if ax and hold are also provided,
        # they will be ignored
        tax = kwargs.get("twinx")
        if tax is not None:
            if issubclass(type(tax), mpl.axes.Axes):
                clear = False
                ax = tax.twinx()
                # warning : this currently returns a normal Axes (so units-naive)
                # TODO: try to solve this
                ax.name = "main"
                tax.name = "twin"  # the previous main is renamed!
                self.ndaxes["main"] = ax
                self.ndaxes["twin"] = tax
            else:
                raise ValueError(f"{tax} is not recognized as a valid Axe")

        self._fig = get_figure(preferences=prefs, **kwargs)

        if clear:
            self._ndaxes = {}  # reset ndaxes
            self._divider = None

        if ax is not None:
            # ax given in the plot parameters,
            # in this case we will plot on this ax
            if issubclass(type(ax), mpl.axes.Axes):
                ax.name = "main"
                self.ndaxes["main"] = ax
            else:
                raise ValueError(f"{ax} is not recognized as a valid Axe")

        elif self._fig.get_axes():
            # no ax parameters in keywords, so we need to get those existing
            # We assume that the existing axes have a name
            self.ndaxes = self._fig.get_axes()
        else:
            # or create a new subplot
            # ax = self._fig.gca(projection=ax3d) :: passing parameters DEPRECATED in matplotlib 3.4
            # ---
            if not ax3d:
                ax = _Axes(self._fig, 1, 1, 1)
                ax = self._fig.add_subplot(ax)
            else:
                ax = _Axes3D(self._fig)
                ax = self._fig.add_axes(ax, projection="3d")

            ax.name = "main"
            self.ndaxes["main"] = ax

        # set the prop_cycle according to preference
        prop_cycle = eval(prefs.axes.prop_cycle)  # noqa: S307
        if isinstance(prop_cycle, str):
            # not yet evaluated
            prop_cycle = eval(prop_cycle)  # noqa: S307

        colors = prop_cycle.by_key()["color"]
        for i, c in enumerate(colors):
            try:
                c = to_rgba(c)
                colors[i] = c
            except ValueError:
                try:
                    c = to_rgba(f"#{c}")
                    colors[i] = c
                except ValueError as e:
                    raise e

        linestyles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^"]
        if ax is not None and "scatter" in method:
            ax.set_prop_cycle(
                cycler("color", colors * len(linestyles) * len(markers))
                + cycler("linestyle", linestyles * len(colors) * len(markers))
                + cycler("marker", markers * len(colors) * len(linestyles)),
            )
        elif ax is not None and "scatter" not in method:
            ax.set_prop_cycle(
                cycler("color", colors * len(linestyles))
                + cycler("linestyle", linestyles * len(colors)),
            )

        # Get the number of the present figure
        self._fignum = self._fig.number

        # for generic plot, we assume only a single axe
        # with possible projections
        # and an optional colobar.
        # other plot class may take care of other needs

        ax = self.ndaxes["main"]

        if ndim == 2:
            # TODO: also the case of 3D

            # show projections (only useful for map or image)
            # ------------------------------------------------
            self.colorbar = colorbar = kwargs.get("colorbar", prefs.colorbar)

            proj = kwargs.get("proj", prefs.show_projections)
            # TODO: tell the axis by title.

            xproj = kwargs.get("xproj", prefs.show_projection_x)

            yproj = kwargs.get("yproj", prefs.show_projection_y)

            SHOWXPROJ = (proj or xproj) and method in ["map", "image"]
            SHOWYPROJ = (proj or yproj) and method in ["map", "image"]

            # Create the various axes
            # -------------------------
            # create new axes on the right and on the top of the current axes
            # The first argument of the new_vertical(new_horizontal) method is
            # the height (width) of the axes to be created in inches.
            #
            # This is necessary for projections and colorbar

            self._divider = None
            if (SHOWXPROJ or SHOWYPROJ or colorbar) and self._divider is None:
                self._divider = make_axes_locatable(ax)

            divider = self._divider

            if SHOWXPROJ:
                axex = divider.append_axes(
                    "top",
                    1.01,
                    pad=0.01,
                    sharex=ax,
                    frameon=0,
                    yticks=[],
                )
                axex.tick_params(bottom="off", top="off")
                plt.setp(axex.get_xticklabels() + axex.get_yticklabels(), visible=False)
                axex.name = "xproj"
                self.ndaxes["xproj"] = axex

            if SHOWYPROJ:
                axey = divider.append_axes(
                    "right",
                    1.01,
                    pad=0.01,
                    sharey=ax,
                    frameon=0,
                    xticks=[],
                )
                axey.tick_params(right="off", left="off")
                plt.setp(axey.get_xticklabels() + axey.get_yticklabels(), visible=False)
                axey.name = "yproj"
                self.ndaxes["yproj"] = axey

            if colorbar and not ax3d:
                axec = divider.append_axes(
                    "right",
                    0.15,
                    pad=0.1,
                    frameon=0,
                    xticks=[],
                    yticks=[],
                )
                axec.tick_params(right="off", left="off")
                # plt.setp(axec.get_xticklabels(), visible=False)
                axec.name = "colorbar"
                self.ndaxes["colorbar"] = axec

        return method

    def _plot_resume(self, origin: Any, **kwargs: Any) -> None:
        # Clean up after plotting and handle plot output
        # Args:
        #    origin: Original dataset
        #    **kwargs: Plot options
        # put back the axes in the original dataset
        # (we have worked on a copy in plot)
        if not kwargs.get("data_transposed", False):
            origin.ndaxes = self.ndaxes
            if not hasattr(self, "_ax_lines"):
                self._ax_lines = None
            origin._ax_lines = self._ax_lines
            if not hasattr(self, "_axcb"):
                self._axcb = None
            origin._axcb = self._axcb
        else:
            nda = {}
            for k, v in self.ndaxes.items():
                nda[k + "T"] = v
            origin.ndaxes = nda
            origin._axT_lines = self._ax_lines
            if hasattr(self, "_axcb"):
                origin._axcbT = self._axcb

        origin._fig = self._fig

        loc = kwargs.get("legend")
        if isinstance(loc, str) or (
            isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], float)
        ):
            origin.ndaxes["main"].legend(loc=loc)
        elif loc is not None and not isinstance(loc, bool):
            origin.ndaxes["main"].legend(loc)

        # Additional matplotlib commands on the current plot
        # ---------------------------------------------------------------------
        commands = kwargs.get("commands", [])
        if commands:
            for command in commands:
                com, val = command.split("(")
                val = val.split(")")[0].split(",")
                ags = []
                kws = {}
                for item in val:
                    if "=" in item:
                        k, v = item.split("=")
                        kws[k.strip()] = eval(v)  # noqa: S307
                    else:
                        ags.append(eval(item))  # noqa: S307
                getattr(self.ndaxes["main"], com)(*ags, **kws)  # TODO: improve this

        # output command should be after all plot commands

        savename = kwargs.get("output")
        if savename is not None:
            # we save the figure with options found in kwargs
            # starting with `save`
            kw = {}
            for key, value in kwargs.items():
                if key.startswith("save"):
                    key = key[4:]
                    kw[key] = value
            self._fig.savefig(savename, **kw)

    # ------------------------------------------------------------------------
    # Special attributes
    # ------------------------------------------------------------------------
    def _attributes_(self):
        return ["fignum", "ndaxes", "divider"]

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------
    @property
    def fig(self):
        """Matplotlib figure associated to this dataset."""
        return self._fig

    @property
    def fignum(self):
        """Matplotlib figure associated to this dataset."""
        return self._fignum

    @property
    def ndaxes(self):
        """A dictionary containing all the axes of the current figures."""
        return self._ndaxes

    @ndaxes.setter
    def ndaxes(self, axes):
        # we assume that the axes have a name
        if isinstance(axes, list):
            # a list a axes have been passed
            for ax in axes:
                self._ndaxes[ax.name] = ax
        elif isinstance(axes, dict):
            self._ndaxes.update(axes)
        elif isinstance(axes, _Axes):
            # it's an axe! add it to our list
            self._ndaxes[axes.name] = axes

    @property
    def ax(self):
        """The main matplotlib axe associated to this dataset."""
        return self._ndaxes["main"]

    @property
    def axT(self):
        """The matplotlib axe associated to the transposed dataset."""
        return self._ndaxes["mainT"]

    @property
    def axec(self):
        """Matplotlib colorbar axe associated to this dataset."""
        return self._ndaxes["colorbar"]

    @property
    def axecT(self):
        """Matplotlib colorbar axe associated to the transposed dataset."""
        return self._ndaxes["colorbarT"]

    @property
    def axex(self):
        """Matplotlib projection x axe associated to this dataset."""
        return self._ndaxes["xproj"]

    @property
    def axey(self):
        """Matplotlib projection y axe associated to this dataset."""
        return self._ndaxes["yproj"]

    @property
    def divider(self):
        """Matplotlib plot divider."""
        return self._divider

    # prepare docstring for 1D plot and plot_<method> function
    docprocess.delete_params(
        "plot.other_parameters", "colorbar", "projections", "transposed", "y_reverse"
    )
    docprocess.delete_params("plot.parameters", "method")


# make plot accessible directly from the scp API
plot = NDPlot.plot
