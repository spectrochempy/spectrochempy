# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
This module defines the class |NDPlot| in which generic plot methods for a |NDDataset| are defined.
"""

__all__ = ['NDPlot', 'plot']

import re
import textwrap

from cycler import cycler
import matplotlib as mpl
from matplotlib.colors import to_rgba
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
from traitlets import Dict, HasTraits, Instance, Union, default, TraitError

from spectrochempy.utils import get_figure, pathclean
from spectrochempy.core.dataset.meta import Meta
from spectrochempy.core import preferences, dataset_preferences, project_preferences, matplotlib_preferences
from spectrochempy.core import error_
from spectrochempy.core.plotters.plot1d import plot_1D
from spectrochempy.core.plotters.plot3d import plot_3D
from spectrochempy.core.plotters.plot2d import plot_2D


# from spectrochempy.utils import deprecated

# ======================================================================================================================
# Management of the preferences for datasets
# ======================================================================================================================

class Preferences(Meta):
    """
    Preferences management
    """

    def __init__(self, **data):
        super().__init__(**data)

    def __getitem__(self, key):

        # search on the preferences
        if self.parent is not None:
            res = getattr(self.parent, f'{self.name}_{key}')
        elif hasattr(matplotlib_preferences, key):
            res = getattr(matplotlib_preferences, key)
        elif hasattr(dataset_preferences, key):
            res = getattr(dataset_preferences, key)
        elif hasattr(project_preferences, key):
            res = getattr(project_preferences, key)
        elif hasattr(preferences, key):
            res = getattr(preferences, key)
        else:
            alias = self._get_alias(key)
            if alias:
                if isinstance(alias, list):
                    res = Preferences(parent=self, name=key, **dict([(n, getattr(self, f'{key}_{n}')) for n in alias]))
                else:
                    res = getattr(self, alias)
            else:
                res = super().__getitem__(key)
                if res is None:
                    error_(
                            f'not found {key}')  # key = key.replace('_','.').replace('...', '_').replace('..',
                    # '-')  #  # res = mpl.rcParams[key]

        return res

    def __setitem__(self, key, value):

        # also change the corresponding preferences
        if hasattr(matplotlib_preferences, key):
            try:
                setattr(matplotlib_preferences, key, value)
            except TraitError:
                value = type(matplotlib_preferences.traits()[key].default_value)(value)
                setattr(matplotlib_preferences, key, value)
        elif hasattr(dataset_preferences, key):
            setattr(dataset_preferences, key, value)
        elif hasattr(project_preferences, key):
            setattr(project_preferences, key, value)
        elif hasattr(preferences, key):
            setattr(preferences, key, value)
        elif key in self.keys():
            newkey = f'{self.name}_{key}'
            setattr(matplotlib_preferences, newkey, value)
            self.parent[newkey] = value
            return
        else:
            # try to find an alias for matplotlib values
            alias = self._get_alias(key)
            if alias:
                newkey = f'{alias}_{key}'
                setattr(matplotlib_preferences, newkey, value)
                self.parent[newkey] = value
            else:
                error_(f'not found {key}')
            return

        super().__setitem__(key, value)

        # # ........................ TO WORK ON  ## from spectrochempy.core import preferences,  # config_manager  #
        # read json files in the pscp file (obj[f])  # # then write it in the main config  #   #  # directory  # f =
        # 'ProjectPreferences.json'  # if f in obj.files:  # TODO: work on this  #     prefjsonfile  #  =  #
        #  os.path.join(config_dir, f)  #     with open(prefjsonfile, 'w') as fd:  #         json.dump(obj[f], fd,
        #  indent=4)  #     # we must also reinit preferences  #     app.init_all_preferences()  #  #    #  #  #  #
        #  app.load_config_file(prefjsonfile)  #     app.project_preferences = ProjectPreferences(config=app.config,
        #  parent=app)

    # ------------------------------------------------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_alias(self, key):

        alias = []
        lkeyp = (len(key) + 1)

        regex = r"[a-zA-Z0-9_]*(?:\b|_)" + key + "(?:\b|_)[a-zA-Z0-9_]*"
        for item in matplotlib_preferences.trait_names():
            matches = re.match(regex, item)
            if matches is not None:
                alias.append(item)

        if alias:

            starts = any([par.startswith(key) for par in alias])
            # ends = any([par.endswith(key) for par in alias])

            if len(alias) > 1:
                if alias[0].endswith(key) and (not starts and self.parent is not None):
                    # it is a member of a group but we don't know which one:
                    raise KeyError(
                            f'Found several keys for {key}: {alias}, so it is ambigous. Please choose on one of them')
                else:
                    if any([par.startswith(key) for par in alias]):
                        # we return the group of parameters
                        pars = []
                        for par in alias:
                            if par.startswith(key):
                                pars.append(par[lkeyp:])
                        return pars
            else:
                return alias[0][:-lkeyp]

        raise KeyError(f'{key} not found in matplolib preferences')

    # ------------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self):

        # remove the matplotlib_user json file to reset to defaults
        config_dir = pathclean(preferences.cfg.config_dir)
        f = config_dir / 'MatplotlibPreferences.json'
        if f.exists():
            f.unlink()

        matplotlib_preferences._apply_style('scpy')
        self.style = 'scpy'

        # reset also non-matplolib preferences
        nonmplpars = ['method_1D', 'method_2D', 'method_3D', 'colorbar', 'show_projections', 'show_projection_x',
                      'show_projection_y', 'colormap', 'max_lines_in_stack', 'simplify', 'number_of_x_labels',
                      'number_of_y_labels', 'number_of_z_labels', 'number_of_contours', 'contour_alpha',
                      'contour_start', 'antialiased', 'rcount', 'ccount']
        for par in nonmplpars:
            setattr(self, par, matplotlib_preferences.traits()[par].default_value)

        self._data = {}

    def all(self):
        """
        List all parameters with their current and default value
        """
        for key in matplotlib_preferences.trait_names(config=True):
            self.help(key)

    def help(self, key):
        """
        Display information on a given parameter

        Parameters
        ----------
        key: str
            name of the parameter for which we want information
        """
        from spectrochempy.utils import colored, TBold
        value = self[key]
        trait = matplotlib_preferences.traits()[key]
        default = trait.default_value
        thelp = trait.help.replace('\n', ' ').replace('\t', ' ')
        sav = ''
        while thelp != sav:
            sav = thelp
            thelp = thelp.replace('  ', ' ')
        help = '\n'.join(textwrap.wrap(thelp, 100, initial_indent=' ' * 20, subsequent_indent=' ' * 20))

        value = colored(value, 'GREEN')
        default = colored(default, 'BLUE')

        print(TBold(f"{key} = {value} \t[default: {default}]"))
        print(f"{help}\n")

    def makestyle(self, filename='mydefault', to_mpl=False):

        if filename.startswith('scpy'):
            error_('`scpy` is READ-ONLY. Please use an another style name.')
            return

        txt = ""
        sline = ""

        for key in mpl.rcParams.keys():
            if key in ['animation.avconv_args', 'animation.avconv_path', 'animation.html_args', 'keymap.all_axes',
                       'mathtext.fallback_to_cm', 'validate_bool_maybe_none', 'savefig.jpeg_quality',
                       'text.latex.preview', 'backend', 'backend_fallback', 'date.epoch', 'docstring.hardcopy',
                       'figure.max_open_warning', 'figure.raise_window', 'interactive', 'savefig.directory',
                       'timezone', 'tk.window_focus', 'toolbar', 'webagg.address', 'webagg.open_in_browser',
                       'webagg.port', 'webagg.port_retries']:
                continue
            val = str(mpl.rcParams[key])
            sav = ''
            while val != sav:
                sav = val
                val = val.replace('  ', ' ')
            line = f'{key:40s} : {val}\n'
            if line[0] != sline:
                txt += '\n'
                sline = line[0]
            if key not in ['axes.prop_cycle']:
                line = line.replace('[', '').replace(']', "").replace('\'', '').replace('"', '')
            if key == 'savefig.bbox':
                line = f'{key:40s} : standard\n'
            txt += line.replace("#", '')

        # Non matplotlib parameters,
        # some parameters are not saved in matplotlib style sheets so we willa dd them here
        nonmplpars = ['method_1D', 'method_2D', 'method_3D', 'colorbar', 'show_projections', 'show_projection_x',
                      'show_projection_y', 'colormap', 'max_lines_in_stack', 'simplify', 'number_of_x_labels',
                      'number_of_y_labels', 'number_of_z_labels', 'number_of_contours', 'contour_alpha',
                      'contour_start', 'antialiased', 'rcount', 'ccount']
        txt += '\n\n##\n## ADDITIONAL PARAMETERS FOR SPECTROCHEMPY\n##\n'
        for par in nonmplpars:
            txt += f"##@{par:37s} : {getattr(self, par)}\n"

        stylesheet = (pathclean(self.stylesheets) / filename).with_suffix('.mplstyle')
        stylesheet.write_text(txt)

        if to_mpl:
            # make it also accessible to pyplot
            stylelib = (pathclean(mpl.get_configdir()) / 'stylelib' / filename).with_suffix('.mplstyle')
            stylelib.write_text(txt)

        # matplotlib_preferences.traits()['style'].trait_types = matplotlib_preferences.traits()['style'].trait_types +\
        #                                                       (Unicode(filename),)
        self.style = filename
        return self.style


# ======================================================================================================================
# Class NDPlot to handle plotting of datasets
# ======================================================================================================================

class NDPlot(HasTraits):
    """
    Plotting interface for |NDDataset|

    This class is used as basic plotting interface of the |NDDataset|.
    """

    # variable containing the matplotlib axis defined for a NDArray object
    _ax = Instance(plt.Axes, allow_none=True)

    # The figure on which this NDArray can be plotted
    _fig = Union((Instance(plt.Figure), Instance(go.Figure)), allow_none=True)

    # The axes on which this dataset and other elements such as projections
    # and colorbar can be plotted
    _ndaxes = Dict(Instance(plt.Axes))

    # add metadata to store plot parameters
    _preferences = Instance(Preferences, allow_none=True)

    # ------------------------------------------------------------------------------------------------------------------
    # generic plotter and plot related methods or properties
    # ------------------------------------------------------------------------------------------------------------------
    def plot(self, **kwargs):
        """
        Generic plot function.

        This apply to a |NDDataset| but actually delegate the work to a plotter defined by the parameter ``method``.
        """

        # --------------------------------------------------------------------
        # select plotter depending on the dimension of the data
        # --------------------------------------------------------------------

        method = 'generic'

        method = kwargs.pop('method', method)

        # Find or guess the adequate plotter
        # -----------------------------------

        _plotter = getattr(self, f"plot_{method.replace('+', '_')}", None)
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
        The generic plotter.

        It try to guess an adequate basic plot for the data. Other method of plotters are defined explicitely in the
        ``plotters`` package.

        Parameters
        ----------
        ax : :class:`matplotlib.axe`
            the viewplot where to plot.
        kwargs : optional additional arguments

        Returns
        -------
        ax
            Return the handler to ax where the main plot was done
        """

        if self._squeeze_ndim == 1:

            ax = plot_1D(self, **kwargs)

        elif self._squeeze_ndim == 2:

            ax = plot_2D(self, **kwargs)

        elif self._squeeze_ndim == 3:

            ax = plot_3D(self, **kwargs)

        else:
            error_('Cannot guess an adequate plotter, nothing done!')
            return False

        return ax

    def close_figure(self):
        """Close a Matplotlib figure associated to this dataset"""
        if self._fig is not None:
            plt.close(self._fig)

    # ------------------------------------------------------------------------------------------------------------------
    # setup figure properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _figure_setup(self, ndim=1, **kwargs):

        prefs = self.preferences

        method = prefs.method_2D if ndim == 2 else prefs.method_1D
        method = kwargs.get('method', method)
        ax3d = '3d' if method in ['surface'] else None

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
                raise ValueError(f'{tax} is not recognized as a valid Axe')

        self._fig = get_figure(preferences=prefs, **kwargs)

        if clear:
            self._ndaxes = {}  # reset ndaxes
            self._divider = None

        if ax is not None:
            # ax given in the plot parameters,
            # in this case we will plot on this ax
            if isinstance(ax, (plt.Axes)):
                ax.name = 'main'
                self.ndaxes['main'] = ax
            else:
                raise ValueError('{} is not recognized as a valid Axe'.format(ax))

        elif self._fig.get_axes():
            # no ax parameters in keywords, so we need to get those existing
            # We assume that the existing axes have a name
            self.ndaxes = self._fig.get_axes()
        else:
            # or create a new subplot
            ax = self._fig.gca(projection=ax3d)
            ax.name = 'main'
            self.ndaxes['main'] = ax

        # set the prop_cycle according to preference
        prop_cycle = eval(prefs.axes.prop_cycle)
        if isinstance(prop_cycle, str):
            # not yet evaluated
            prop_cycle = eval(prop_cycle)

        colors = prop_cycle.by_key()['color']
        for i, c in enumerate(colors):
            try:
                c = to_rgba(c)
                colors[i] = c
            except ValueError:
                try:
                    c = to_rgba(f'#{c}')
                    colors[i] = c
                except ValueError as e:
                    raise e

        linestyles = ['-', '--', ':', '-.']
        markers = ['o', 's', '^']
        if ax is not None and (kwargs.pop('scatter', False) or kwargs.pop('scatterpen', False)):
            ax.set_prop_cycle(cycler('color', colors * len(linestyles) * len(markers)) + cycler('linestyle',
                                                                                                linestyles * len(
                                                                                                        colors) * len(
                                                                                                        markers)) +
                              cycler(
                    'marker', markers * len(colors) * len(linestyles)))
        elif ax is not None and kwargs.pop('pen', False):
            ax.set_prop_cycle(cycler('color', colors * len(linestyles)) + cycler('linestyle', linestyles * len(colors)))

        # Get the number of the present figure
        self._fignum = self._fig.number

        # for generic plot, we assume only a single axe
        # with possible projections
        # and an optional colobar.
        # other plot class may take care of other needs

        ax = self.ndaxes['main']

        if ndim == 2:
            # TODO: also the case of 3D

            # show projections (only useful for map or image)
            # ------------------------------------------------

            self.colorbar = colorbar = kwargs.get('colorbar', prefs.colorbar)

            proj = kwargs.get('proj', prefs.show_projections)
            # TODO: tell the axis by title.

            xproj = kwargs.get('xproj', prefs.show_projection_x)

            yproj = kwargs.get('yproj', prefs.show_projection_y)

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
            if (SHOWXPROJ or SHOWYPROJ or colorbar) and self._divider is None:
                self._divider = make_axes_locatable(ax)

            divider = self._divider

            if SHOWXPROJ:
                axex = divider.append_axes("top", 1.01, pad=0.01, sharex=ax, frameon=0, yticks=[])
                axex.tick_params(bottom='off', top='off')
                plt.setp(axex.get_xticklabels() + axex.get_yticklabels(), visible=False)
                axex.name = 'xproj'
                self.ndaxes['xproj'] = axex

            if SHOWYPROJ:
                axey = divider.append_axes("right", 1.01, pad=0.01, sharey=ax, frameon=0, xticks=[])
                axey.tick_params(right='off', left='off')
                plt.setp(axey.get_xticklabels() + axey.get_yticklabels(), visible=False)
                axey.name = 'yproj'
                self.ndaxes['yproj'] = axey

            if colorbar and not ax3d:
                axec = divider.append_axes("right", .15, pad=0.1, frameon=0, xticks=[], yticks=[])
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
                getattr(self.ndaxes['main'], com)(*ags, **kws)  # TODO: improve this

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
    @default('_preferences')
    def _preferences_default(self):
        return Preferences()

    # ..................................................................................................................
    @property
    def preferences(self):
        """
        |Meta| instance object - Additional metadata.
        """
        return self._preferences

    # ..................................................................................................................
    @preferences.setter
    def preferences(self, preferences):
        # property.setter for preferences
        if preferences is not None:
            self._preferences.update(preferences)

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
plot = NDPlot.plot  # make plot accessible directly from the scp API

# ======================================================================================================================
if __name__ == '__main__':
    pass
