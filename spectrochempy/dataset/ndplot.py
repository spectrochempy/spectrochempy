# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B)
# See full LICENSE agreement in the root directory
# =============================================================================



"""
This module define the class :class:`NDPlot` in which generic plot
methods for a :class:`~spectrochempy.dataset.nddataset.NDDataset`
are defined.

"""

# Python and third parties imports
# ----------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
from traitlets import Dict, HasTraits, Instance

from spectrochempy.utils import is_sequence
from spectrochempy.plotters.utils import cmyk2rgb
from spectrochempy.application import app

# local import
# ------------
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
    for :class:`~spectrochempy.dataset.nddataset.NDDataset`

    This class is used as basic plotting interface of the
    :class:`~spectrochempy.dataset.nddataset.NDDataset`.

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
        a :class:`~spectrochempy.dataset.nddataset.NDDataset` which
        actually delegate the work to a plotter defined by the parameter ``method``.

        Parameters
        ----------

        method : str, optional

            The method of plot of the dataset,
            which will determine the plotter to use.
            For instance, for 2D data, it can be `map`, `stack` or `image`
            among other method.

        ax : :class:`matplotlib.Axes` instance. Optional, default = current or new one

            The axe where to plot

        figsize : tuple, optional, default is mpl.rcParams['figure.figsize']

            The figure size

        fontsize : int, optional

            The font size in pixels, default is 10 (or read from preferences)

        hold : `bool`, optional, default = `False`.

            Should we plot on the ax previously used
            or create a new figure?

        style : str

        autolayout : `bool`, optional, default=``True``

            if True, layout will be set automatically

        output: str

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
    pass


