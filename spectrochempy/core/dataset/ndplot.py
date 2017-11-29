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

"""
This module define the class :class:`NDPlot` in which generic plot
methods for a :class:`~spectrochempy.core.dataset.nddataset.NDDataset`
are defined.

"""

# Python and third parties imports
# ----------------------------------

import copy
import datetime
import time
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler

import numpy as np
from numpy.compat import asbytes, asstr
from numpy.lib.format import write_array, MAGIC_PREFIX
from numpy.lib.npyio import zipfile_factory, NpzFile
from traitlets import Dict, List, Bool, Float, HasTraits, \
    Instance, observe, All, Int

# local import
# ------------

from spectrochempy.core.dataset.ndarray import masked
from spectrochempy.core.dataset.ndcoords import CoordSet, Coord
from spectrochempy.core.dataset.ndmeta import Meta
from spectrochempy.core.units import Unit
from spectrochempy.gui import gui
from spectrochempy.utils import SpectroChemPyWarning
from spectrochempy.utils import is_sequence
from spectrochempy.core.plotters.utils import cmyk2rgb
from spectrochempy.application import app
plotoptions = app.plotoptions
log = app.log
options = app
from spectrochempy.utils import docstrings

# Constants
# ---------

__all__ = ['NDPlot',

           'figure',
           'show',
           'plot',

           # 'interactive_masks',
           '_set_figure_style',

           # styles and colors
           'available_styles',
           'NBlack', 'NRed', 'NBlue', 'NGreen',

           ]
_classes = ['NDPlot']

# For color blind people, it is safe to use only 4 colors in graphs:
# see http://jfly.iam.u-tokyo.ac.jp/color/ichihara_etal_2008.pdf
#   Black CMYK=0,0,0,0
#   Red CMYK= 0, 77, 100, 0 %
#   Blue CMYK= 100, 30, 0, 0 %
#   Green CMYK= 85, 0, 60, 10 %
NBlack = (0, 0, 0)
NRed = cmyk2rgb(0, 77, 100, 0)
NBlue = cmyk2rgb(100, 30, 0, 0)
NGreen = cmyk2rgb(85, 0, 60, 10)


# =============================================================================
# Class NDPlot to handle plotting of datasets
# =============================================================================

class NDPlot(HasTraits):
    """
    Plotting interface
    for :class:`~spectrochempy.core.dataset.nddataset.NDDataset`

    This class is used as basic plotting interface of the
    :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.

    """

    # The figure on which this dataset can be plotted
    _fig = Instance(plt.Figure, allow_none=True)

    # The axes on which this dataset and other elements such as projections 
    # and colorbar can be plotted
    _ndaxes = Dict(Instance(plt.Axes))

    # -------------------------------------------------------------------------
    # generic plotter and plot related methods or properties
    # -------------------------------------------------------------------------

    # .........................................................................
    @docstrings.get_sectionsf('plot')
    @docstrings.dedent
    def plot(self, **kwargs):

        """
        Generic plot function for
        a :class:`~spectrochempy.core.dataset.nddataset.NDDataset` which
        actually delegate the work to a plotter defined by the parameter ``method``.

        Parameters
        ----------

        method : `str`, optional

            The method of plot of the dataset,
            which will determine the plotter to use.
            For instance, for 2D data, it can be `map`, `stack` or `image`
            among other method.

        ax : :class:`matplotlib.Axes` instance. Optional, default = current or new one

            The axe where to plot

        figsize : `tuple`, optional, default is mpl.rcParams['figure.figsize']

            The figure size

        fontsize : `int`, optional

            The font size in pixels, default is 10 (or read from preferences)

        hold : `bool`, optional, default = `False`.

            Should we plot on the ax previously used
            or create a new figure?

        style : `str`

        autolayout : `bool`, optional, default=``True``

            if True, layout will be set automatically

        output: `str`

            A string containing a path to a filename. The output format is deduced
            from the extension of the filename. If the filename has no extension,
            the value of the rc parameter savefig.format is used.

        dpi : [ None | scalar > 0]

            The resolution in dots per inch. If None it will default to the
            value savefig.dpi in the matplotlibrc file.

        """

        # color cycle
        # prop_cycle = options.prop_cycle
        # mpl.rcParams['axes.prop_cycle']= r" cycler('color', %s) " % prop_cycle

        # -------------------------------------------------------------------------
        # select plotter depending on the dimension of the data
        # -------------------------------------------------------------------------
        method = kwargs.pop('method', 'generic')
        log.debug('Call to plot_{}'.format(method))

        # Find or guess the adequate plotter
        # -----------------------------------

        _plotter = getattr(self, 'plot_{}'.format(method), None)
        if _plotter is None:
            # no plotter found
            log.error('The specified plotter for method '
                      '`{}` was not found!'.format(method))
            raise IOError

        # Execute the plotter
        # --------------------

        return _plotter(**kwargs)

    # -------------------------------------------------------------------------
    # plotter: plot_generic
    # -------------------------------------------------------------------------

    # .........................................................................
    def plot_generic(self, **kwargs):
        """
        The generic plotter. It try to guess an adequate basic plot for the data.
        Other method of plotters are defined explicitely in the `viewer` package.

        Parameters
        ----------

        ax : :class:`matplotlib.axe`

            the viewplot where to plot.

        kwargs : optional additional arguments

        Returns
        -------

        ax : return the handler to ax where the main plot was done

        """

        if self.ndim == 1:

            ax = self.plot_1D(**kwargs)

        elif self.ndim == 2:

            ax = self.plot_2D(**kwargs)

        elif self.ndim == 3:

            ax = self.plot_3D(**kwargs)

        else:
            log.error('Cannot guess an adequate plotter, nothing done!')
            return False

        return ax

    # -------------------------------------------------------------------------
    # setup figure properties
    # -------------------------------------------------------------------------

    # .........................................................................
    def _figure_setup(self, ndim=1, **kwargs):

        _set_figure_style(**kwargs)

        self._figsize = mpl.rcParams['figure.figsize'] = \
            kwargs.get('figsize', mpl.rcParams['figure.figsize'])

        mpl.rcParams[
            'figure.autolayout'] = kwargs.pop('autolayout', True)

        # Get current figure information
        # ------------------------------
        log.debug('update plot')

        # get the current figure
        hold = kwargs.get('hold', False)
        self._fig = _curfig(hold)
        self._fig.rcParams = plt.rcParams.copy()

        # is ax in the keywords ?
        ax = kwargs.pop('ax', None)
        if not hold:
            self._ndaxes = {}  # reset ndaxes
            self._divider = None

        if ax is not None:
            # ax given in the plot parameters,
            # in this case we will plot on this ax
            if isinstance(ax, plt.Axes):
                ax.name = 'main'
                self.ndaxes['main'] = ax
            else:
                raise ValueError('{} is not recognized'.format(ax))

        elif self._fig.get_axes():
            # no ax parameters in keywords, so we need to get those existing
            # We assume that the existing axes have a name
            self.ndaxes = self._fig.get_axes()
        else:
            # or create a new subplot
            ax = self._fig.gca()
            ax.name = 'main'
            self.ndaxes['main'] = ax

        if ax is not None and kwargs.get('method') in ['scatter']:
            ax.set_prop_cycle(
                    cycler('color',
                           [NBlack, NBlue, NRed, NGreen] * 3) +
                    cycler('linestyle',
                           ['-', '--', ':', '-.'] * 3) +
                    cycler('marker',
                           ['o', 's', '^'] * 4))
        elif ax is not None and kwargs.get('method') in ['lines']:
            ax.set_prop_cycle(
                    cycler('color',
                           [NBlack, NBlue, NRed, NGreen]) +
                    cycler('linestyle',
                           ['-', '--', ':', '-.']))

        # Get the number of the present figure
        self._fignum = self._fig.number

        # for generic plot, we assume only a single axe
        # with possible projections
        # and an optional colobar.
        # other plot class may take care of other needs

        ax = self.ndaxes['main']

        if ndim == 2:
            # TODO: also the case of 3D

            method = kwargs.get('method', plotoptions.method_2D)

            # show projections (only useful for map or image)
            # ------------------------------------------------

            colorbar = kwargs.get('colorbar', True)

            proj = kwargs.get('proj', plotoptions.show_projections)
            # TODO: tell the axis by title.

            xproj = kwargs.get('xproj', plotoptions.show_projection_x)

            yproj = kwargs.get('yproj', plotoptions.show_projection_y)

            SHOWXPROJ = (proj or xproj) and method in ['map', 'image']
            SHOWYPROJ = (proj or yproj) and method in ['map', 'image']

            # Create the various axes
            # -------------------------
            # create new axes on the right and on the top of the current axes
            # The first argument of the new_vertical(new_horizontal) method is
            # the height (width) of the axes to be created in inches.
            #
            # This is necessary for projections and colorbar

            if (SHOWXPROJ or SHOWYPROJ or colorbar) \
                    and self._divider is None:
                self._divider = make_axes_locatable(ax)

            divider = self._divider

            if SHOWXPROJ:
                axex = divider.append_axes("top", 1.01, pad=0.01, sharex=ax,
                                           frameon=0, yticks=[])
                axex.tick_params(bottom='off', top='off')
                plt.setp(axex.get_xticklabels() + axex.get_yticklabels(),
                         visible=False)
                axex.name = 'xproj'
                self.ndaxes['xproj'] = axex

            if SHOWYPROJ:
                axey = divider.append_axes("right", 1.01, pad=0.01, sharey=ax,
                                           frameon=0, xticks=[])
                axey.tick_params(right='off', left='off')
                plt.setp(axey.get_xticklabels() + axey.get_yticklabels(),
                         visible=False)
                axey.name = 'yproj'
                self.ndaxes['yproj'] = axey

            if colorbar:
                axec = divider.append_axes("right", .15, pad=0.1, frameon=0,
                                           xticks=[], yticks=[])
                axec.tick_params(right='off', left='off')
                # plt.setp(axec.get_xticklabels(), visible=False)
                axec.name = 'colorbar'
                self.ndaxes['colorbar'] = axec

    # -------------------------------------------------------------------------
    # resume a figure plot
    # -------------------------------------------------------------------------

    # .........................................................................
    def _plot_resume(self, origin, **kwargs):

        log.debug('resume plot')

        # put back the axes in the original source
        # (we have worked on a copy in plot)
        if not kwargs.get('data_transposed', False):
            origin.ndaxes = self.ndaxes
            origin._ax_lines = self._ax_lines
            if hasattr(self, "_axcb"):
                origin._axcb = origin._axcb
        else:
            nda = {}
            for k, v in self.ndaxes.items():
                nda[k + 'T'] = v
            origin.ndaxes = nda
            origin._axT_lines = self._ax_lines
            if hasattr(self, "_axcb"):
                origin._axcbT = origin._axcb

        origin._fig = self._fig

        # Additional matplotlib commands on the current plot
        # ---------------------------------------------------------------------

        commands = kwargs.get('commands', [])
        if commands:
            for command in commands:
                com, val = command.split('(')
                val = val.split(')')[0].split(',')
                ags = []
                kws = {}
                for item in val:
                    if '=' in item:
                        k, v = item.split('=')
                        kws[k.strip()] = eval(v)
                    else:
                        ags.append(eval(item))
                getattr(self.ndaxes['main'], com)(*ags,
                                                  **kws)  # TODO:improve this

        # output command should be after all plot commands

        savename = kwargs.get('output', None)
        if savename is not None:
            # we save the figure with options found in kwargs
            # starting with `save`
            log.debug('save plot to {}'.format(savename))
            kw = {}
            for key, value in kwargs.items():
                if key.startswith('save'):
                    key = key[4:]
                    kw[key] = value
            self._fig.savefig(savename, **kw)

        #mpl.interactive(True)
        plt.draw()

    # -------------------------------------------------------------------------
    # Special attributes
    # -------------------------------------------------------------------------

    # .........................................................................
    def __dir__(self):
        return ['fignum', 'ndaxes', 'divider']

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    # .........................................................................
    @property
    def fig(self):
        """
        Matplotlib figure associated to this dataset

        """
        return self._fig

    # .........................................................................
    @property
    def fignum(self):
        """
        Matplotlib figure associated to this dataset

        """
        return self._fignum

    # .........................................................................
    @property
    def ndaxes(self):
        """
        A dictionary containing all the axes of the current figures
        """
        return self._ndaxes

    # .........................................................................
    @ndaxes.setter
    def ndaxes(self, axes):
        # we assume that the axes have a name
        if isinstance(axes, list):
            # a list a axes have been passed
            for ax in axes:
                log.debug('add axe: {}'.format(ax.name))
                self._ndaxes[ax.name] = ax
        elif isinstance(axes, dict):
            self._ndaxes.update(axes)
        elif isinstance(axes, Axes):
            # it's an axe! add it to our list
            self._ndaxes[axes.name] = axes

    # .........................................................................
    @property
    def ax(self):
        """
        the main matplotlib axe associated to this dataset

        """
        return self._ndaxes['main']

    # .........................................................................
    @property
    def axT(self):
        """
        the matplotlib axe associated to the transposed dataset

        """
        return self._ndaxes['mainT']

    # .........................................................................
    @property
    def axec(self):
        """
        Matplotlib colorbar axe associated to this dataset

        """
        return self._ndaxes['colorbar']

    # .........................................................................
    @property
    def axecT(self):
        """
        Matplotlib colorbar axe associated to the transposed dataset

        """
        return self._ndaxes['colorbarT']

    # .........................................................................
    @property
    def axex(self):
        """
        Matplotlib projection x axe associated to this dataset

        """
        return self._ndaxes['xproj']

    # .........................................................................
    @property
    def axey(self):
        """
        Matplotlib projection y axe associated to this dataset

        """
        return self._ndaxes['yproj']

    # .........................................................................
    @property
    def divider(self):
        """
        Matplotlib plot divider

        """
        return self._divider

    # -------------------------------------------------------------------------
    # events and interactive functions
    # -------------------------------------------------------------------------

    # a flag to say if we act on zoom
    # (useful for 'with_transposed' method of plot).
    _zoom_detection = Bool

    _all_masks = List()  # to store temporary the mask positions

    _xlim = List()  # used to detect zoom in ax
    _ylim = List()  # used to detect zoom in transposed axT

    # .........................................................................
    def interactive_masks(self, **kwargs):
        """
        Set mask interactively

        Parameters
        ----------
        colorbar

        kwargs


        """
        # TODO: make it for 1D too!

        # delayed import to avoir circular import
        from spectrochempy.core.plotters.multiplot import plot_with_transposed

        colorbar = kwargs.get('colorbar', True)

        # simplify plot for better interactivity
        plt.rcParams['path.simplify_threshold'] = 0

        # reduce the number of lines (max = max_lines_in_stack per default)
        self._maxlines = kwargs.get('maxlines', plotoptions.max_lines_in_stack)

        plot_with_transposed(
                source=self,
                colorbar=colorbar,
                suptitle='INTERACTIVE MASK SELECTION '
                         '(press `a` for help)',
                suptitle_color=NBlue,
                **kwargs)


        ax = self.ax
        axT = self.axT
        axec = self.axec
        axecT = self.axecT

        fig = self.fig

        # set the working set of lines (a copy to preserve original list)
        # ax_lines and axT_lines are the displayed set that can be modified
        # while _ax_lines and _axT_lines are the original

        self.ax_lines = ax_lines = self._ax_lines[:]
        self.axT_lines = axT_lines = self._axT_lines[:]
        self.ax_zorder = len(self._ax_lines) + 2
        self.axT_zorder = len(self._axT_lines) + 2

        help_message = \
            """ 
             ================================================================
             HELP
             ================================================================

             --------- KEYS -------------------------------------------------
             * Press and hold 'a' for this help.
             * Press 'esc' to force redraw and hide any displayed text.
             * Press 'ctrl+z' to undo the last set or last selected mask.
             * Press 'ctrl+x' to apply all mask selections and exit. 

             --------- MOUSE ------------------------------------------------
             * click the right button to pick a row and mask it.
             * click the left button on a mask to select it.
             * double-click the left button to pick and mask a single column.
             * Press the left button, move and release for a range selection. 

             ================================================================
            """

        self._helptext = axT.text(0.02, 0.02, help_message, fontsize=10,
                                  fontweight='bold',
                                  transform=fig.transFigure, color='blue',
                                  bbox={'facecolor': 'white',
                                        'edgecolor': 'blue'})
        self._tpos = axT.text(0.5, 0.5, '', fontsize=12,
                              ha='center', va='center',
                              fontweight='bold',
                              transform=fig.transFigure, color='green',
                              bbox={'facecolor': 'white',
                                    'edgecolor': 'green'})

        def show_help():
            self._helptext.set_visible(True)
            fig.canvas.draw()

        def show_action(message):
            self._tpos.set_text(message)
            self._tpos.set_visible(True)
            fig.canvas.draw()
            log.debug("show action : " + message)

        def hide_help():
            try:
                self._helptext.set_visible(False)
            except:
                pass

        def hide_action():
            try:
                self._tpos.set_visible(False)
            except:
                pass

        def get_limits():
            # get limits (if they change, they will triger a change observed
            # below in the self._limits_changed function
            self._xlim = ax.get_xlim()
            self._ylim = axT.get_xlim()

        def exact_coord_x(c):
            # get the closest nddataset x coordinate from c
            idx = self._loc2index(c, -1)
            return (idx, self.x.data[idx])

        def exact_coord_y(c):
            # get the closest nddataset y coordinate from c
            idx = self._loc2index(c, 0)
            return (idx, self.y.data[idx])

        self._hover_T_line = None

        def get_hover_T_line(c):
            idx, _ = exact_coord_x(c)
            self._hover_T_line = copy.copy(self._axT_lines[idx])
            self._hover_T_line.set_linewidth(2)
            self._hover_T_line.set_color(NGreen)
            self._hover_T_line.set_zorder(self.axT_zorder)
            self._update_axes()

        self._hover_line = None

        def get_hover_line(c):
            idx, _ = exact_coord_y(c)
            self._hover_line = copy.copy(self._ax_lines[idx])
            self._hover_line.set_linewidth(2)
            self._hover_line.set_color(NRed)
            self._hover_line.set_zorder(self.ax_zorder)
            self._update_axes()

        # hide messages
        # -------------

        hide_help()
        hide_action()

        # utility transform for drawinfg on the colorbars
        # ------------------------------------------------

        ax_axis_to_data = ax.transAxes + ax.transData.inverted()
        ax_data_to_axis = ax_axis_to_data.inverted()
        axdata2axis = lambda x: ax_data_to_axis.transform((x, 0))[0]
        axT_axis_to_data = axT.transAxes + axT.transData.inverted()
        axT_data_to_axis = axT_axis_to_data.inverted()
        axTdata2axis = lambda x: axT_data_to_axis.transform((x, 0))[0]

        # cursors
        # ---------

        # init them out of the axes (not visible)
        self.x0 = min(ax.get_xlim()) - 10.
        self.y0 = min(axT.get_xlim()) - 10.

        self.ax_cursor = ax.axvline(self.x0, alpha=.8, color=NRed, lw=1.5,
                                    label='cursor',
                                    zorder=self.ax_zorder)
        self.axecT_cursor = axecT.axhline(axdata2axis(self.x0), alpha=.8,
                                          color=NGreen,
                                          lw=1.5,
                                          label='cursor')

        self.axT_cursor = axT.axvline(self.y0, alpha=.8, color=NGreen, lw=1.5,
                                      label='cursor',
                                      zorder=self.axT_zorder)
        self.axec_cursor = axec.axhline(axTdata2axis(self.y0), alpha=.8,
                                        color=NRed,
                                        lw=1.5,
                                        label='cursor')

        # row or cols to be masked initialisation
        # ---------------------------------------

        # self._all_masks will contain full informations about the selected masks
        self._all_masks = []

        # update axes display data with a reduce number of lines
        # -------------------------------------------------------

        self._update_axes()

        # now we can start to watch for changes (zoom, etc...)
        # ----------------------------------------------------

        self._zoom_detection = True

        # mouse events
        # ------------

        def _onleaveaxes(event):
            # fired on a mouse motion leading to leave an ax
            # here we delete displayed cursor on all axes

            self.axT_cursor.set_xdata(self.y0)
            self.axec_cursor.set_ydata(axTdata2axis(self.y0))

            self.ax_cursor.set_xdata(self.x0)
            self.axecT_cursor.set_ydata(axdata2axis(self.y0))

            self._hover_line = None
            self._hover_T_line = None

            self._update_axes()

        def _onmove(event):
            # fired on a mouse motion
            # we use this event to display cursors
            if not event.inaxes:
                return

            hide_help()
            hide_action()
            # and to get the new limts in case for example
            # of an interative zoom
            get_limits()

            if event.inaxes is ax:
                x = event.xdata
                self.ax_cursor.set_xdata(x)
                self.axecT_cursor.set_ydata(axdata2axis(x))
                get_hover_T_line(x)

            elif event.inaxes is axT:
                x = event.xdata
                self.axT_cursor.set_xdata(x)
                self.axec_cursor.set_ydata(axTdata2axis(x))
                get_hover_line(x)

        def _onclick(event):
            # fired on a mouse click.

            # if it is not fired in ax or the transposed axT, return
            # immediately and do nothing, except ot hide the 'help' text.
            hide_help()

            if event.inaxes not in [self.ax, self.axT]:
                return

            # check which button was pressed

            if event.button == 1 and event.dblclick:  # double-click left button
                inax = event.inaxes
                x = event.xdata

                if inax is self.axT:
                    # set x to the closest original nddataset y coordinate
                    idx, x = exact_coord_y(x)
                    axvT = inax.axvline(x,
                                        lw=2, color='white', alpha=.9,
                                        picker=True,
                                        zorder=self.axT_zorder + 10,
                                        label='mask_row_%d' % idx)
                    self._all_masks.append(('row', axvT, x, idx))

                    # corresponding value in the original display
                    # it is a complete row that we remove
                    line = self._ax_lines[idx]
                    show_action('mask row at y={:.2f}'.format(x))
                    self._update_axes()

                elif inax is self.ax:
                    idx, x = exact_coord_x(x)
                    axv = inax.axvline(x,
                                       lw=2, color='white', alpha=.9,
                                       picker=True,
                                       zorder=self.ax_zorder + 10,
                                       label='mask_col_%d' % idx)
                    self._all_masks.append(('col', axv, x, idx))

                    # corresponding value in the transposed display
                    # it is a complete row of axT

                    # corresponding value in the original display
                    # it is a complete row that we remove
                    line = self._axT_lines[idx]
                    show_action('mask column at x={:.2f}'.format(x))
                    self._update_axes()

            pass

        self._pressevent = self.fig.canvas.mpl_connect(
                'button_press_event', _onclick)
        self._motionevent = self.fig.canvas.mpl_connect(
                'motion_notify_event', _onmove)
        self._leaveevent = self.fig.canvas.mpl_connect(
                'axes_leave_event', _onleaveaxes)

        # key events
        # ----------

        def _on_key(event):

            if event.key in ['h', 'a']:
                # we show the help.
                show_help()

            if event.key in ['esc']:
                # we show the help.
                hide_help()
                hide_action()
                fig.canvas.draw()

        def _on_key_release(event):

            if event.key in ['a', 'h']:
                hide_help()

            if event.key in ['ctrl+z']:
                if self._all_masks:
                    last = list(self._all_masks.pop(-1))
                    #if last[0] in ['rowspan', 'colspan', '''col']:
                    last[1].remove()
                    #else:
                    #    last[1].set_color(last[3])
                    #    last[1].set_linewidth(last[4])
                    show_action('removed {} selection at {:.2f}'.format(last[0],
                                                                        last[
                                                                            2]))
                    fig.canvas.draw()
                else:
                    show_action('all masks were already removed')

            if event.key in ['ctrl+x']:
                log.info("apply all defined mask to the dataset")

                for item in self._all_masks:
                    _item = list(item)
                    if _item[0] in ['colspan']:
                        xmin, xmax = _item[2:]
                        self[:, xmin:xmax] = masked
                        log.debug("span {}:{}".format(xmin, xmax))

                    elif _item[0] in ['rowspan']:
                        ymin, ymax = _item[2:]
                        self[ymin:ymax] = masked
                        log.debug("rowspan {}:{}".format(ymin, ymax))

                    elif _item[0] in ['col']:
                        x = _item[2]
                        self[:, x] = masked
                        log.debug("col {}".format(x))

                    elif _item[0] in ['row']:
                        y = eval(_item[2])
                        self[y] = masked
                        log.debug("row {}".format(y))

                show_action('Masks applied to the dataset')

                plt.close(self._fig)

        self.fig.canvas.mpl_connect('key_press_event', _on_key)
        self.fig.canvas.mpl_connect('key_release_event', _on_key_release)

        # pick event
        # ----------

        def _onpick(event):

            inax = event.mouseevent.inaxes

            if isinstance(event.artist, Line2D):

                button = event.mouseevent.button
                sel = event.artist

                if sel.get_label().startswith('mask'):
                    # TODO: offer the possibility of deleting this mask selection
                    return

                if inax is ax:
                    y = eval(sel.get_label())
                    x = event.mouseevent.xdata
                    cond_row = (button == 3)
                    cond_col = (button == 1 and event.mouseevent.dblclick)

                elif inax is axT:
                    x = eval(sel.get_label())
                    y = event.mouseevent.xdata
                    cond_col = (button == 3)
                    cond_row = (button == 1 and event.mouseevent.dblclick)

                if cond_row:
                    # right button -> row selection
                    idx, y = exact_coord_y(y)
                    axvT = axT.axvline(y,
                                       lw=2, color='white', alpha=.9,
                                       picker=True,
                                       zorder=self.axT_zorder + 10,
                                       label='mask_row_%d' % idx)
                    self._all_masks.append(('row', axvT, y, idx))
                    show_action("mask row picked at y={:.2f}".format(y))

                elif cond_col:

                    # left button -> column selection
                    idx, x = exact_coord_x(x)
                    axv = ax.axvline(x, lw=2, color='white', alpha=.9,
                                     picker=True,
                                     zorder=self.ax_zorder + 10,
                                     label='mask_col_%d' % idx)
                    self._all_masks.append(('col', axv, x, idx))
                    show_action("mask column picked at x={:.2f}".format(x))

                self._update_axes()

        self.fig.canvas.mpl_connect('pick_event', _onpick)

        def _onspanx(xmin, xmax):
            xmin, xmax = sorted((xmin, xmax))
            sp = ax.axvspan(xmin, xmax, facecolor='white',
                            edgecolor='white',
                            zorder=self.ax_zorder, picker=True)
            self._all_masks.append(('colspan', sp, xmin, xmax))
            show_action("col span between {:.2f} and {:.2f}".format(xmin, xmax))
            self._donotupdate = False
            self._pressevent = self.fig.canvas.mpl_connect(
                    'button_press_event', _onclick)
            self._motionevent = self.fig.canvas.mpl_connect(
                    'motion_notify_event', _onmove)
            self._leaveevent = self.fig.canvas.mpl_connect(
                    'axes_leave_event', _onleaveaxes)

            self._update_axes()

        def _onmovesp(xmin, xmax):
            try:

                self.fig.canvas.mpl_disconnect(self._pressevent)
                self.fig.canvas.mpl_discconnect(self._motionevent)
                self.fig.canvas.mpl_disconnect(self._leaveevent)
            except:
                pass
            self._donotupdate = True

        min_x_span = np.diff(self.x.data)[0]*.1
        self._spanx = SpanSelector(ax, _onspanx, 'horizontal',
                                   minspan=min_x_span,
                                   button=[1], onmove_callback=_onmovesp,
                                   useblit=False,
                                   rectprops=dict(alpha=0.5,
                                                  zorder=self.axT_zorder,
                                                  facecolor=NRed,
                                                  edgecolor='w'))

        def _onspany(ymin, ymax):
            ymin, ymax = sorted((ymin, ymax))
            sp = axT.axvspan(ymin, ymax, facecolor='white',
                             edgecolor='white',
                             zorder=self.axT_zorder, picker=True)
            self._all_masks.append(('rowspan', sp, ymin, ymax))
            show_action("row span between {:.2f} and {:.2f}".format(ymin, ymax))
            self._donotupdate = False
            self._pressevent = self.fig.canvas.mpl_connect(
                    'button_press_event', _onclick)
            self._motionevent = self.fig.canvas.mpl_connect(
                    'motion_notify_event', _onmove)
            self._leaveevent = self.fig.canvas.mpl_connect(
                    'axes_leave_event', _onleaveaxes)
            self._update_axes()

        min_y_span = np.diff(self.y.data)[0]*.1
        self._spany = SpanSelector(axT, _onspany, 'horizontal',
                                   minspan=min_y_span,
                                   button=[1], onmove_callback=_onmovesp,
                                   useblit=False,
                                   rectprops=dict(alpha=0.5,
                                                  zorder=self.axT_zorder,
                                                  facecolor=NGreen,
                                                  edgecolor='w'))
        return ax

    # .........................................................................
    def _get_masked_lines(self):

        # get the masks related lines and markers
        masked_lines = {}
        masked_T_lines = {}
        masked_markers = []
        masked_T_markers = []

        for item in self._all_masks:

            _item = list(item)

            if _item[0].endswith('span'):

                direction = _item[0][:3]
                val1, val2 = sorted(_item[2:])
                if direction=='row':
                    lines = self.ax_lines[:]
                    axlines = self.ax_lines
                elif direction=='col':
                    lines = self.axT_lines[:]
                    axlines = self.axT_lines
                for line in lines:
                    select = (eval(line.get_label()) >= val1
                               and eval(line.get_label()) <= val2)
                    if select and line in axlines:
                            axlines.remove(line)

            elif _item[0]=='col':

                idx = _item[3]
                line = self._axT_lines[idx]
                masked_T_lines[line.get_label()] = line

                axv = _item[1]
                masked_markers.append(axv)

            elif _item[0]=='row':

                idx = _item[3]
                line = self._ax_lines[idx]
                masked_lines[line.get_label()] = line

                axv = _item[1]
                masked_T_markers.append(axv)

        return (masked_lines, masked_T_lines, masked_markers, masked_T_markers)

    # .........................................................................
    def _update_axes(self):

        if self._donotupdate:
            return

        (masked_lines, masked_T_lines,
         masked_markers, masked_T_markers) = self._get_masked_lines()

        self.ax.lines = []
        setpy = max(len(self.ax_lines) // self._maxlines, 1)
        self.ax.lines = self.ax_lines[::setpy]  # displayed ax lines

        for line in masked_lines.values():  # remove masked line
            if line in self.ax.lines:
                self.ax.lines.remove(line)

        self.ax.lines.append(self.ax_cursor)  # restore cursor line

        if self._hover_line is not None: # \
                #and self._hover_line.get_label() not in masked_lines.keys():  # show hover line
            log.debug(self._hover_line.get_label())
            self.ax.lines.append(self._hover_line)

        for line in masked_markers:
            self.ax.lines.append(line)

        self.axT.lines = []
        setpx = max(len(self.axT_lines) // self._maxlines, 1)
        self.axT.lines = self.axT_lines[::setpx]  # displayed axTlines

        for line in masked_T_lines.values():
            if line in self.axT.lines:
                self.axT.lines.remove(line)

        self.axT.lines.append(self.axT_cursor)  # restore cursor line

        if self._hover_T_line is not None:# \
                #and self._hover_T_line.get_label() not in masked_T_lines.keys():
            self.axT.lines.append(self._hover_T_line)

        for line in masked_T_markers:
            self.axT.lines.append(line)

        self.fig.canvas.draw()

    # .........................................................................
    @observe('_xlim', '_ylim')
    def _limits_changed(self, change):
        if not self._zoom_detection:
            return
        self.ax_lines = self._ax_lines[:]
        self.axT_lines = self._axT_lines[:]

        if change['name'] == '_xlim':
            x1, x2 = sorted(self._xlim)
            lines = self.axT_lines[:]
            for line in lines:
                if eval(line.get_label()) < x1 or eval(line.get_label()) > x2:
                    self.axT_lines.remove(line)

        if change['name'] == '_ylim':
            y1, y2 = sorted(self._ylim)
            lines = self.ax_lines[:]
            for line in lines:
                if eval(line.get_label()) < y1 or eval(line.get_label()) > y2:
                    self.ax_lines.remove(line)
        self._update_axes()


# .............................................................................
def figure(**kwargs):
    """
    Method to open a new figure
    """
    return _curfig(hold=False, **kwargs)


# .............................................................................
def show():
    """
    Method to force the `matplotlib` figure display

    """
    if not plotoptions.do_not_block or plt.isinteractive():
        if _curfig(True):  # True to avoid opening a new one
            plt.show()


# .............................................................................
def _curfig(hold=False, **kwargs):
    # Get the figure where to plot.

    n = plt.get_fignums()

    if not n or not hold:
        # create a figure
        return plt.figure(**kwargs)

    # a figure already exists - if several we take the last
    return plt.figure(n[-1])


# .............................................................................
def _set_figure_style(**kwargs):
    # set temporarily a new style if any

    log.debug('set style')

    style = kwargs.get('style', None)

    if style:
        if not is_sequence(style):
            style = [style]
        if isinstance(style, dict):
            style = [style]
        style = ['classic', plotoptions.style] + list(style)
        plt.style.use(style)
    else:
        style = ['classic', plotoptions.style]
        plt.style.use(style)
        plt.style.use(plotoptions.style)

        fontsize = mpl.rcParams['font.size'] = \
            kwargs.get('fontsize', mpl.rcParams['font.size'])
        mpl.rcParams['legend.fontsize'] = int(fontsize * .8)
        mpl.rcParams['xtick.labelsize'] = int(fontsize)
        mpl.rcParams['ytick.labelsize'] = int(fontsize)
        mpl.rcParams['axes.prop_cycle'] = (
            cycler('color', [NBlack, NBlue, NRed, NGreen]))

        return mpl.rcParams

# .............................................................................
def available_styles():
    """
    Styles availables in SpectroChemPy

    Todo
    -----
    Make this list extensible programmatically

    Returns
    -------
    l : a list of style

    """
    return ['notebook', 'paper', 'poster', 'talk', 'sans']


# .............................................................................
plot = NDPlot.plot  # make plot accessible directly from the API

# =============================================================================
if __name__ == '__main__':
    # test interactive masks

    from spectrochempy.api import *

    options.log_level = DEBUG

    A = NDDataset.read_omnic(
            os.path.join(scpdata, 'irdata', 'NH4Y-activation.SPG'))
    A.y -= A.y[0]
    A.y.to('hour', inplace=True)
    A.y.title = u'Aquisition time'


    # ax = A[:, 1600.:4000.].plot()
    # plt.show()

    # ax = A[:, 1600.:4000.].plot_stack()
    # plt.show()


    def _interactive_masks():
        A[:, :].interactive_masks(
                method='stack', figsize=(9, 6),
                right=.905,  # to lease space for the labels of the colorbar
        )
        pass


    _interactive_masks()
    plt.show()
