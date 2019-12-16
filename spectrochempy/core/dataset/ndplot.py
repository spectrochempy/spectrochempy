# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module defines the class |NDPlot| in which generic plot
methods for a |NDDataset| are defined.

"""

__all__ = ['NDPlot',
           'plot',

           # styles and colors
           '_set_figure_style'
           ]

# Python and third parties imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import warnings

from cycler import cycler
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from traitlets import Dict, HasTraits, Instance, default

# local import
# ----------------------------------------------------------------------------------------------------------------------
from spectrochempy.utils import (is_sequence, SpectroChemPyDeprecationWarning,
                                 docstrings, NBlack, NBlue, NGreen, NRed, get_figure)
from spectrochempy.core import general_preferences, project_preferences
from ...core import info_, debug_, error_, warning_

prefs = general_preferences


from spectrochempy.core.plotters.plot1d import plot_1D
from spectrochempy.core.plotters.plot3d import plot_3D
from spectrochempy.core.plotters.plot2d import plot_2D
from spectrochempy.utils.meta import Meta
from spectrochempy.utils import deprecated


# ======================================================================================================================
# Class NDPlot to handle plotting of datasets
# ======================================================================================================================

class NDPlot(HasTraits):
    """
    Plotting interface for |NDDataset|

    This class is used as basic plotting interface of the |NDDataset|.

    """

    # The figure on which this dataset can be plotted
    _fig = Instance(plt.Figure, allow_none=True)

    # The axes on which this dataset and other elements such as projections 
    # and colorbar can be plotted
    _ndaxes = Dict(Instance(plt.Axes))

    # add metadata to store plot parameters
    _plotmeta = Instance(Meta, allow_none=True)

    # ------------------------------------------------------------------------------------------------------------------
    # generic plotter and plot related methods or properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @docstrings.get_sectionsf('plot')
    @docstrings.dedent
    def plot(self, **kwargs):
        """
        Generic plot function for
        a |NDDataset| which
        actually delegate the work to a plotter defined by the parameter ``method``.

        Parameters
        ----------
        method : str, optional
            The method of plot of the dataset, which will determine the
            plotter to use. For instance, for 2D data, it can be `map`,
            `stack` or `image` among other method.
        ax : |Axes| instance. Optional, default:current or new one
            The axe where to plot
        figsize : tuple, optional, default is mpl.rcParams['figure.figsize']
            The figure size
        fontsize : int, optional
            The font size in pixels, default is 10 (or read from preferences)
        clear : `bool`, optional, default=`True`.
            Should we plot on the ax previously used or create a new figure?
        style : str
        autolayout : `bool`, optional, default=True
            if True, layout will be set automatically
        output : str
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

        # --------------------------------------------------------------------
        # select plotter depending on the dimension of the data
        # --------------------------------------------------------------------

        method = 'generic'

        # check the deprecated use of `kind`
        kind = kwargs.pop('kind', None)
        if kind is not None:
            method = kind
            warnings.warn('`kind`is deprecated, use `method` instead',
                          SpectroChemPyDeprecationWarning)

        method = kwargs.pop('method', method)
        # debug_('Call to plot_{}'.format(method))

        # Find or guess the adequate plotter
        # -----------------------------------

        _plotter = getattr(self, 'plot_{}'.format(method), None)
        if _plotter is None:
            # no plotter found
            error_('The specified plotter for method '
                      '`{}` was not found!'.format(method))
            raise IOError

        # Execute the plotter
        # --------------------

        return _plotter(**kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # plotter: plot_generic
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
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

        if self._squeeze_ndim == 1:

            ax = plot_1D(self, **kwargs)

        elif self._squeeze_ndim == 2:

            ax = plot_2D(self, **kwargs)

        elif self.squeze_ndim == 3:

            ax = plot_3D(self, **kwargs)

        else:
            error_('Cannot guess an adequate plotter, nothing done!')
            return False

        return ax

    # ------------------------------------------------------------------------------------------------------------------
    # setup figure properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _figure_setup(self, ndim=1, **kwargs):

        # debug_('figure setup')

        # by default we use the matplotlib librairy especially for plotting
        # in the jupyter notebook
        # but there is also possibility that the plots are made using
        # pyqtgraph in the GUI - usempl is the flag for that option

        usempl = kwargs.get('usempl', True)
        method = kwargs.get('method', project_preferences.method_2D)
        ax3d = '3d' if method in ['surface'] else None

        # make matplolib specific setup

        if usempl:
            _set_figure_style(**kwargs)

            self._figsize = mpl.rcParams['figure.figsize'] = \
                kwargs.get('figsize', mpl.rcParams['figure.figsize'])

            mpl.rcParams[
                'figure.autolayout'] = kwargs.pop('autolayout', True)

        # Get current figure information
        # ------------------------------

        # should we use the previous figure?
        clear = kwargs.get('clear', True)

        # is ax in the keywords ?
        ax = kwargs.pop('ax', None)

        # is it a twin figure? In such case if ax and hold are also provided,
        # they will be ignored
        tax = kwargs.get('twinx', None)
        if tax is not None:
            if isinstance(tax, plt.Axes):
                clear = False
                ax = tax.twinx()
                ax.name = 'main'
                tax.name = 'twin'  # the previous main is renamed!
                self.ndaxes['main'] = ax
                self.ndaxes['twin'] = tax
            else:
                raise ValueError(
                    '{} is not recognized as a valid Axe'.format(tax))

        if usempl:
            # get the current figure (or the last used)
            self._fig = get_figure(clear)
            self._fig.rcParams = plt.rcParams.copy()

        if clear:
            self._ndaxes = {}  # reset ndaxes
            self._divider = None

        if ax is not None:
            # ax given in the plot parameters,
            # in this case we will plot on this ax
            if isinstance(ax, (plt.Axes)) or (hasattr(ax, 'implements') and
                                              ax.implements('PyQtGraphWidget')):
                ax.name = 'main'
                self.ndaxes['main'] = ax
            else:
                raise ValueError(
                    '{} is not recognized as a valid Axe'.format(ax))

        elif self._fig.get_axes():
            # no ax parameters in keywords, so we need to get those existing
            # We assume that the existing axes have a name
            self.ndaxes = self._fig.get_axes()
        else:
            # or create a new subplot
            ax = self._fig.gca(projection=ax3d)
            ax.name = 'main'
            self.ndaxes['main'] = ax

        if ax is not None and kwargs.pop('scatter', False):
            ax.set_prop_cycle(
                cycler('color',
                       [NBlack, NBlue, NRed, NGreen, 'magenta', 'cyan'] *
                       2) +
                cycler('linestyle',
                       ['-', '--', ':', '-.'] * 3) +
                cycler('marker',
                       ['o', 's', '^'] * 4))
        elif ax is not None and kwargs.pop('pen', False):
            ax.set_prop_cycle(
                cycler('color',
                       [NBlack, NBlue, NRed, NGreen]) +
                cycler('linestyle',
                       ['-', '--', ':', '-.']))

        if usempl:
            # Get the number of the present figure
            self._fignum = self._fig.number

        # for generic plot, we assume only a single axe
        # with possible projections
        # and an optional colobar.
        # other plot class may take care of other needs

        ax = self.ndaxes['main']

        if ndim == 2:
            # TODO: also the case of 3D

            method = kwargs.get('method', self.plotmeta.method_2D)

            # show projections (only useful for map or image)
            # ------------------------------------------------

            colorbar = kwargs.get('colorbar', self.plotmeta.colorbar)

            proj = kwargs.get('proj', self.plotmeta.show_projections)
            # TODO: tell the axis by title.

            xproj = kwargs.get('xproj', self.plotmeta.show_projection_x)

            yproj = kwargs.get('yproj', self.plotmeta.show_projection_y)

            SHOWXPROJ = (proj or xproj) and method in ['map', 'image']
            SHOWYPROJ = (proj or yproj) and method in ['map', 'image']

            # Create the various axes
            # -------------------------
            # create new axes on the right and on the top of the current axes
            # The first argument of the new_vertical(new_horizontal) method is
            # the height (width) of the axes to be created in inches.
            #
            # This is necessary for projections and colorbar

            self._divider = None
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

            if colorbar and not ax3d:
                axec = divider.append_axes("right", .15, pad=0.1, frameon=0,
                                           xticks=[], yticks=[])
                axec.tick_params(right='off', left='off')
                # plt.setp(axec.get_xticklabels(), visible=False)
                axec.name = 'colorbar'
                self.ndaxes['colorbar'] = axec

    # ------------------------------------------------------------------------------------------------------------------
    # resume a figure plot
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _plot_resume(self, origin, **kwargs):

        # debug_('resume plot')

        # put back the axes in the original dataset
        # (we have worked on a copy in plot)
        if not kwargs.get('data_transposed', False):
            origin.ndaxes = self.ndaxes
            if not hasattr(self, '_ax_lines'):
                self._ax_lines = None
            origin._ax_lines = self._ax_lines
            if not hasattr(self, "_axcb"):
                self._axcb = None
            origin._axcb = self._axcb
        else:
            nda = {}
            for k, v in self.ndaxes.items():
                nda[k + 'T'] = v
            origin.ndaxes = nda
            origin._axT_lines = self._ax_lines
            if hasattr(self, "_axcb"):
                origin._axcbT = self._axcb

        origin._fig = self._fig

        loc = kwargs.get("legend", None)
        if loc:
            origin.ndaxes['main'].legend(loc=loc)
        
        
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
            # debug_('save plot to {}'.format(savename))
            kw = {}
            for key, value in kwargs.items():
                if key.startswith('save'):
                    key = key[4:]
                    kw[key] = value
            self._fig.savefig(savename, **kw)

    # ------------------------------------------------------------------------------------------------------------------
    # Special attributes
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def __dir__(self):
        return ['fignum', 'ndaxes', 'divider']

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @default('_plotmeta')
    def _plotmeta_default(self):
        return Meta()

    # ..................................................................................................................
    @property
    def plotmeta(self):
        """
        |Meta| instance object - Additional metadata.

        """
        return self._plotmeta

    # ..................................................................................................................
    @plotmeta.setter
    def plotmeta(self, plotmeta):
        # property.setter for plotmeta
        if plotmeta is not None:
            self._plotmeta.update(plotmeta)

    # ..................................................................................................................
    @property
    def fig(self):
        """
        Matplotlib figure associated to this dataset

        """
        return self._fig

    # ..................................................................................................................
    @property
    def fignum(self):
        """
        Matplotlib figure associated to this dataset

        """
        return self._fignum

    # ..................................................................................................................
    @property
    def ndaxes(self):
        """
        A dictionary containing all the axes of the current figures
        """
        return self._ndaxes

    # ..................................................................................................................
    @ndaxes.setter
    def ndaxes(self, axes):
        # we assume that the axes have a name
        if isinstance(axes, list):
            # a list a axes have been passed
            for ax in axes:
                # debug_('add axe: {}'.format(ax.name))
                self._ndaxes[ax.name] = ax
        elif isinstance(axes, dict):
            self._ndaxes.update(axes)
        elif isinstance(axes, plt.Axes):
            # it's an axe! add it to our list
            self._ndaxes[axes.name] = axes

    # ..................................................................................................................
    @property
    def ax(self):
        """
        the main matplotlib axe associated to this dataset

        """
        return self._ndaxes['main']

    # ..................................................................................................................
    @property
    def axT(self):
        """
        the matplotlib axe associated to the transposed dataset

        """
        return self._ndaxes['mainT']

    # ..................................................................................................................
    @property
    def axec(self):
        """
        Matplotlib colorbar axe associated to this dataset

        """
        return self._ndaxes['colorbar']

    # ..................................................................................................................
    @property
    def axecT(self):
        """
        Matplotlib colorbar axe associated to the transposed dataset

        """
        return self._ndaxes['colorbarT']

    # ..................................................................................................................
    @property
    def axex(self):
        """
        Matplotlib projection x axe associated to this dataset

        """
        return self._ndaxes['xproj']

    # ..................................................................................................................
    @property
    def axey(self):
        """
        Matplotlib projection y axe associated to this dataset

        """
        return self._ndaxes['yproj']

    # ..................................................................................................................
    @property
    def divider(self):
        """
        Matplotlib plot divider

        """
        return self._divider


# .............................................................................
def _set_figure_style(**kwargs):
    # set temporarily a new style if any

    # debug_('set style')

    # first, reset to default
    plt.style.use('classic')
    try:  # try because if the installation is not correct, this 'scpy' style
        # may not be found
        plt.style.use('scpy')
    except OSError:
        # scpy not found! may be due to a failing installation
        # make a basic style here
        # get the local version:
        plt.style.use(os.path.join(prefs.stylesheets, 'scpy.mplstyle'))

    # now get the required style form args
    style = kwargs.get('style', None)

    if style:
        # if a style was passed, then we use it
        if not is_sequence(style):
            style = [style]
        if isinstance(style, dict):
            style = [style]
        try:
            plt.style.use(style)
        except OSError:
            # try a local version
            plt.style.use(os.path.join(prefs.stylesheets, style[0] + '.mplstyle'))

    else:
        # else, we try to use the preferences
        if project_preferences.style == 'scpy':
            # already used
            return

        plt.style.use(project_preferences.style)
        # and set some of the parameters here
        fontsize = mpl.rcParams['font.size'] = \
            kwargs.get('fontsize', mpl.rcParams['font.size'])
        mpl.rcParams['legend.fontsize'] = int(fontsize * .8)
        mpl.rcParams['xtick.labelsize'] = int(fontsize)
        mpl.rcParams['ytick.labelsize'] = int(fontsize)
        mpl.rcParams['axes.prop_cycle'] = (
            cycler('color', [NBlack, NBlue, NRed, NGreen]))


# .............................................................................
plot = NDPlot.plot  # make plot accessible directly from the scp API

# ======================================================================================================================
if __name__ == '__main__':
    pass
