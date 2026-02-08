# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
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

import traitlets as tr

from spectrochempy.application.application import debug_
from spectrochempy.application.application import error_
from spectrochempy.application.preferences import preferences as prefs

from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.docutils import docprocess
from spectrochempy.utils.mplutils import _Axes
from spectrochempy.utils.mplutils import _Axes3D
from spectrochempy.utils.mplutils import get_figure
from spectrochempy.utils.mplutils import show as mpl_show
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
    _fig = tr.Any(allow_none=True)
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
        show : bool, optional, default: True
            call `matplotlib.pyplot.show()` at the end of the plot.
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

        # 🚀 LAZY TRIGGER: This is the ONLY place that initializes matplotlib
        # ALL matplotlib setup happens here on the first plot() call
        from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

        lazy_ensure_mpl_config()

        show = kwargs.pop("show", True)

        # --- Default plotting method ---
        if method is None:
            if self._squeeze_ndim == 1:
                method = "pen"
            elif self._squeeze_ndim == 2:
                method = "stack"
            elif self._squeeze_ndim == 3:
                method = "surface"

        _plotter = getattr(self, f"plot_{method.replace('+', '_')}", None)
        if _plotter is None:
            error_(
                NameError,
                f"The specified plotter for method `{method}` was not found!",
            )
            raise OSError

        ax = _plotter(**kwargs)

        if show:
            mpl_show()

        return ax

    @deprecated(
        removed="0.8",
    )
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
        if self._fig is None:
            return
        try:
            import matplotlib.figure

            if isinstance(self._fig, matplotlib.figure.Figure):
                self._fig.clf()
                self._fig.canvas.manager = None  # optional; often unnecessary
        except Exception:
            debug_("Could not import the figure before closing.")

    def _figure_setup(self, ndim=1, method=None, **kwargs):

        # Always ensure full lazy initialization to avoid state corruption
        from spectrochempy.core.plotters.plot_setup import lazy_ensure_mpl_config

        lazy_ensure_mpl_config()

        from matplotlib.axes import Axes

        clear = kwargs.get("clear", True)
        ax = kwargs.pop("ax", None)

        self._fig = get_figure(
            preferences=prefs,
            style=kwargs.get("style"),
            figsize=kwargs.get("figsize"),
            dpi=kwargs.get("dpi"),
        )

        if clear:
            self._ndaxes = {}
            self._divider = None

        if ax is not None:
            if isinstance(ax, Axes):
                ax.name = "main"
                self.ndaxes["main"] = ax
            else:
                raise ValueError(f"{ax} is not a valid Matplotlib Axes")

        elif self._fig.get_axes():
            self.ndaxes = self._fig.get_axes()

        else:
            if ndim < 3:
                ax = self._fig.add_subplot(1, 1, 1)
            else:
                ax = self._fig.add_subplot(111, projection="3d")

            ax.name = "main"
            self.ndaxes["main"] = ax
            self._fignum = None
        return method or ""

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
