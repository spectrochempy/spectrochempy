# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module define the `application` on which the API rely. It also define
the default application preferences and IPython magic functions."""

__all__ = []

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import glob
import sys
import logging
import subprocess
import datetime
import warnings
import pprint
import json
import textwrap

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

from pkg_resources import get_distribution, DistributionNotFound
from setuptools_scm import get_version

from traitlets.config.configurable import Config, Configurable
from traitlets.config.application import Application, catch_config_error
from traitlets import (Bool, Unicode, List, Dict, Integer, Float, Enum,
                       HasTraits, Instance, default, observe, import_item, )
from traitlets.config.manager import BaseJSONConfigManager

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core import magic_arguments
from IPython.core.magic import (Magics, magics_class, line_cell_magic)
from IPython.core.magics.code import extract_symbols
from IPython.core.error import UsageError
from IPython.utils.text import get_text_list

from spectrochempy.utils import MetaConfigurable, display_info_string

# Log levels
# ----------------------------------------------------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# ----------------------------------------------------------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------------------------------------------------------

def version_scheme(version):
    branch = version.branch
    dist = version.distance
    tag = next = version.tag.public
    dirty = version.dirty
    next = next.split('.')
    next[-1]=str(int(next[-1])+1)
    next = '.'.join(next)
    dev = ''
    if branch != 'master':
        dev = f'-dev.{dist}'
        tag = next
    dev = dev
    dirty = '+dirty' if dirty else ''
    version_scheme = f'{tag}{dev}{dirty}'
    if not dev and not dirty:
        # write it only for stable TAG
        # This means, always tab the latest master version
        with open(os.path.join('..','..',os.path.dirname(os.path.relpath(__file__)), 'version.py'), 'w') as f:
            f.write(f"version = '{version_scheme}' # Do not delete. Automatically set when needed\n\n")
    return version_scheme

try:
    __release__ = get_distribution('spectrochempy').version
    "Release version string of this package"
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    __release__ = '0.1a'
    
try:
    __version__ = get_version(root='..',
                              relative_to=__file__,
                              version_scheme = version_scheme)
    "Version string of this package"
except: # pragma: no cover
    __version__ = __release__

# ............................................................................
def _get_copyright():
    current_year = datetime.date.today().year
    copyright = '2014-{}'.format(current_year)
    copyright += ' - A.Travert & C.Fernandez @ LCS'
    return copyright


__copyright__ = _get_copyright()
"Copyright string of this package"


# .............................................................................
def _get_release_date():
    try:
        return subprocess.getoutput(
            "git log -1 --tags --date='short' --format='%ad'")
    except: # pragma: no cover
        pass


__release_date__ = _get_release_date()
"Last release date of this package"

# other info
# ............................................................................

__url__ = "http://www.spectrochempy.fr"
"URL for the documentation of this package"

__author__ = "C. Fernandez & A. Travert @LCS"
"First authors(s) of this package"

__contributor__ = ""
"contributor(s) to this package"

__license__ = "CeCILL-B license"
"Licence of this package"

# colorsmaps and plot styles
# ----------------------------------------------------------------------------------------------------------------------

cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'Greys', 'Purples', 'Blues',
         'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
         'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary',
         'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer',
         'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat',
         'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
         'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'Pastel1', 'Pastel2',
         'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',
         'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
         'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
         'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

markers = list((Line2D.markers.keys()))
markers.remove('')
markers.remove(' ')
linestyles = list((Line2D.lineStyles.keys()))
linestyles.remove('')
linestyles.remove(' ')


# ............................................................................
def available_styles():
    """
    All matplotlib `styles <https://matplotlib.org/users/style_sheets.html>`_
    which are available in |scpy|

    Returns
    -------
    A list of matplotlib styles

    """
    # Todo: Make this list extensible programmatically (adding files to stylelib)

    cfgdir = mpl.get_configdir()

    stylelib = os.path.join(cfgdir, 'stylelib')

    # AT: checks if stylelib exists and adds matplotlib pre-defined styles
    styles = plt.style.available
    if os.path.isdir(stylelib):
        listdir = os.listdir(stylelib)
        for style in listdir:
            if style.endswith('.mplstyle'):
                styles.append(style)
    styles = list(set(styles))  # in order to remove possible duplicates

    if 'scpy' not in styles:  # pragma: no cover
        styles.append('scpy')  # the standard style of spectrochempy

    return styles


# ======================================================================================================================
# Magic ipython function
# ======================================================================================================================
@magics_class
class SpectroChemPyMagics(Magics):
    """
    This class implements the addscript ipython magic function.

    """

    @line_cell_magic
    def addscript(self, pars='', cell=None):
        """
        This works both as **%addscript** and as **%%addscript**

        This magic command can either take a local filename, element in the
        namespace or history range (see %history),
        or the current cell content


        Usage:

            %addscript  -p project  n1-n2 n3-n4 ... n5 .. n6 ...

            or

            %%addscript -p project
            ...code lines ...


        Options:

            -p <string>         Name of the project where the script will be stored.
                                If not provided, a project with a standard
                                name : `proj` is searched.
            -o <string>         script name
            -s <symbols>        Specify function or classes to load from python
                                source.
            -a                  append to the current script instead of
                                overwriting it.
            -n                  search symbol in the current namespace


        Examples
        --------

        .. sourcecode:: ipython

           In[1]: %addscript myscript.py

           In[2]: %addscript 7-27

           In[3]: %addscript -s MyClass,myfunction myscript.py
           In[4]: %addscript MyClass

           In[5]: %addscript mymodule.myfunction


        """
        opts, args = self.parse_options(pars, 'p:o:s:n:a')

        append = 'a' in opts
        mode = 'a' if append else 'w'
        search_ns = 'n' in opts

        if not args and not cell and not search_ns: # pragma: no cover
            raise UsageError('Missing filename, input history range, '
                             'or element in the user namespace.\n '
                             'If no argument are given then the cell content '
                             'should '
                             'not be empty')
        name = 'script'
        if 'o' in opts:
            name = opts['o']

        proj = 'proj'
        if 'p' in opts:
            proj = opts['p']
        if proj not in self.shell.user_ns: # pragma: no cover
            raise ValueError('Cannot find any project with name `{}` in the '
                             'namespace.'.format(proj))
        # get the proj object
        projobj = self.shell.user_ns[proj]

        contents = ""
        if search_ns:
            contents += "\n" + self.shell.find_user_code(opts['n'],
                                                         search_ns=search_ns) + "\n"

        args = " ".join(args)
        if args.strip():
            contents += "\n" + self.shell.find_user_code(args,
                                                         search_ns=search_ns) + "\n"

        if 's' in opts: # pragma: no cover
            try:
                blocks, not_found = extract_symbols(contents, opts['s'])
            except SyntaxError:
                # non python code
                logging.error("Unable to parse the input as valid Python code")
                return

            if len(not_found) == 1:
                warnings.warn('The symbol `%s` was not found' % not_found[0])
            elif len(not_found) > 1:
                warnings.warn(
                    'The symbols %s were not found' % get_text_list(not_found,
                                                                    wrap_item_with='`'))

            contents = '\n'.join(blocks)

        if cell:
            contents += "\n" + cell

        # import delayed to avoid circular import error
        from spectrochempy.core.scripts.script import Script
        script = Script(name, content=contents)
        projobj[name] = script

        return "Script {} created.".format(name)

        # @line_magic  # def runscript(self, pars=''):  #     """  #  #
        # """  #     opts, args = self.parse_options(pars, '')  #  #     if
        # not args:  #         raise UsageError('Missing script name')  #  #
        # return args


# ======================================================================================================================
# DataDir class
# ======================================================================================================================

def _get_pkg_datadir_path(data_name, package=None):
    data_name = os.path.normpath(data_name)

    path = os.path.dirname(import_item(package).__file__)
    path = os.path.join(path, data_name)

    if not os.path.isdir(path): # pragma: no cover
        return os.path.dirname(path)

    return path


class DataDir(HasTraits):
    """ A class used to determine the path to the testdata directory. """

    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------

    path = Unicode()

    def listing(self):
        """
        Create a str representing a listing of the testdata folder.

        Returns
        -------
        listing : str

        """
        strg = os.path.basename(self.path) + "\n"

        def _listdir(s, initial, ns):
            ns += 1
            for f in glob.glob(os.path.join(initial, '*')):
                fb = os.path.basename(f)
                if not fb.startswith('acqu') and not fb.startswith(
                        'pulse') and fb not in ['ser', 'fid']:
                    s += "   " * ns + "|__" + "%s\n" % fb
                if os.path.isdir(f):
                    s = _listdir(s, f, ns)
            return s

        return _listdir(strg, self.path, -1)

    @classmethod
    def class_print_help(cls):
        # to work with --help-all
        """"""  # TODO: make some useful help

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------

    def __str__(self):
        return self.listing()

    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------

    @default('path')
    def _get_path_default(self):
        # the spectra path in package data
        return _get_pkg_datadir_path('testdata', 'scp_data')

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self):
        # _repr_html is needed to output in notebooks
        return self.listing().replace('\n', '<br/>').replace(" ", "&nbsp;")


# ======================================================================================================================
# General Preferences
# ======================================================================================================================


# ======================================================================================================================
class GeneralPreferences(MetaConfigurable):
    """
    Preferences that apply to the |scpy| application in general

    They should be accessible from the main API

    Examples
    --------

    >>> import spectrochempy as scp # doctest: +ELLIPSIS

    >>> delimiter = scp.general_preferences.csv_delimiter


    """

    updated = Bool(False)

    def __init__(self, **kwargs):
        super(GeneralPreferences, self).__init__(**kwargs)

    # various settings
    # ------------------------------------------------------------------------------------------------------------------

    show_info_on_loading = Bool(True, help='Display info on loading').tag(
        config=True)

    show_close_dialog = Bool(True, help='Display the close project dialog'
                                        ' project changing or on '
                                        'apllication exit').tag(config=True)

    csv_delimiter = Unicode(',', help='CSV data delimiter').tag(config=True)

    project_directory = Unicode(help='Directory where projects are '
                                     'stored by default').tag(config=True,
                                                              type='folder')

    @default('project_directory')
    def _get_default_project_directory(self):
        """
        Determines the SpectroChemPy project directory name and
        creates the directory if it doesn't exist.

        This directory is typically ``$HOME/spectrochempy/projects``,
        but if the
        SCP_PROJECTS_HOME environment variable is set and the
        ``$SCP_PROJECTS_HOME`` directory exists, it will be that
        directory.

        If neither exists, the former will be created.

        Returns
        -------
        dir : str
            The absolute path to the projects directory.

        """

        # first look for SCP_PROJECTS_HOME
        scp = os.environ.get('SCP_PROJECTS_HOME')

        if scp is not None and os.path.exists(scp):
            return os.path.abspath(scp)

        scp = os.path.join(os.path.expanduser('~'), 'spectrochempy',
                           'projects')

        if not os.path.exists(scp):
            os.makedirs(scp, exist_ok=True)

        elif not os.path.isdir(scp):
            raise IOError('Intended Projects directory is actually a file.')

        return os.path.abspath(scp)

    autosave_projects = Bool(False, help='Automatic saving of the current '
                                         'project').tag(config=True)

    autoload_project = Bool(False, help='Automatic loading of the last '
                                        'project at startup').tag(config=True)

    datadir = Unicode(help='Directory where to look for data by '
                           'default').tag(config=True, type="folder")

    stylesheets = Unicode(help='Directory where to look for local defined '
                               ' matplotlib styles when they are not in the '
                               ' standard location').tag(config=True,
                                                         type="folder")

    @default('stylesheets')
    def _get_stylesheets_default(self):
        # the spectra path in package data
        return _get_pkg_datadir_path('stylesheets', 'scp_data')

    @default('datadir')
    def _get_default_datadir(self):
        return self.parent.datadir.path

    @observe('datadir')
    def _datadir_changed(self, change):
        self.parent.datadir.path = change['new']

    @property
    def log_level(self):
        """
        int - logging level
        """
        return self.parent.log_level

    @log_level.setter
    def log_level(self, value):
        if isinstance(value, str):
            value = getattr(logging, value, None)
            if value is None:
                warnings.warn('Log level not changed: invalid value given\n'
                              'string values must be DEBUG, INFO, WARNING, '
                              'or ERROR')
        self.parent.log_level = value

    # ------------------------------------------------------------------------------------------------------------------
    # General configuration for plotting
    # ------------------------------------------------------------------------------------------------------------------

    leftbuttonpan = Bool(True, help="LeftButtonPan: If false, left button "
                                    "drags a rubber band for "
                                    "zooming in viewbox").tag(config=True, )

    background_color = Unicode('#EFEFEF',
                               help='Background color for plots').tag(
        config=True, type='color')

    foreground_color = Unicode('#000000',
                               help='Foreground color for plots elements').tag(
        config=True, type='color')

    antialias = Bool(True, help='Antialiasing')

    # matplotlib specific  group
    # ------------------------------------------------------------------------------------------------------------------

    max_lines_in_stack = Integer(1000, min=1, help='Maximum number of lines to'
                                                   ' plot in stack plots').tag(
        config=True)

    usempl = Bool(help='Use MatPlotLib for plotting (slow but mode suitable '
                       'for printing)').tag(config=True, group='mpl')

    simplify = Bool(help='Matplotlib path simplification for improving '
                         'performance').tag(config=True, group='mpl')

    @observe('simplify')
    def _simplify_changed(self, change):
        plt.rcParams['path.simplify'] = change.new
        plt.rcParams['path.simplify_threshold'] = 1.


# ======================================================================================================================
class ProjectPreferences(MetaConfigurable):
    """
    Per project preferences

    include plotting and views preference for the included datasets

    """

    updated = Bool(False)

    def __init__(self, **kwargs):
        super(ProjectPreferences, self).__init__(jsonfile='ProjectPreferences',
                                                 **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # attributes
    # ------------------------------------------------------------------------------------------------------------------

    name = Unicode('ProjectPreferences')

    description = Unicode('Options for datasets, e.g., for plotting')

    # ------------------------------------------------------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    style = Enum(available_styles(), default_value='scpy',
                 help='Basic matplotlib style to use').tag(config=True,
                                                           type='list')

    @observe('style')
    def _style_changed(self, change):
        plt.style.use(change.new)

    # ..................................................................................................................
    use_latex = Bool(True, help='Use latex for plotting labels and texts').tag(
        config=True)

    @observe('use_latex')
    def _use_latex_changed(self, change):
        mpl.rc('text', usetex=change.new)

    # ..................................................................................................................
    latex_preamble = Unicode(r"""\usepackage{siunitx}
                            \sisetup{detect-all}
                            \usepackage{times} # set the normal font here
                            \usepackage{sansmath}
                            # load up the sansmath so that math -> helvet
                            \sansmath
                            """,
                             help='Latex preamble for matplotlib outputs'
                             ).tag(config=True, type='text')

    @observe('latex_preamble')
    def _set_latex_preamble(self, change):
        mpl.rcParams['text.latex.preamble'] = change.new.split('\n')

    # ------------------------------------------------------------------------------------------------------------------

    # 1D options
    # ------------------------------------------------------------------------------------------------------------------

    method_1D = Enum(['pen', 'scatter', 'scatter+pen', 'bar'],
                     default_value='pen',
                     help='Default plot methods for 1D datasets').tag(
        config=True, type='list')

    # 2D/3D options
    # ----------

    method_2D = Enum(['map', 'image', 'stack', 'surface', '3D'], default_value='stack',
                     help='Default plot methods for 2D datasets').tag(
        config=True, type='list')

    # method_3D = Enum(['surface'], default_value='surface',
    #                 help='Default plot methods for 3D datasets').tag(
    #    config=True, type='list')

    colorbar = Bool(True, help='Show color bar for 2D plots').tag(config=True)

    show_projections = Bool(False, help='Show all projections').tag(
        config=True)

    show_projection_x = Bool(False, help='Show projection along x').tag(
        config=True)

    show_projection_y = Bool(False, help='Show projection along y').tag(
        config=True)

    number_of_x_labels = Integer(5, min=3, help='Number of X labels').tag(
        config=True)

    number_of_y_labels = Integer(5, min=3, help='Number of Y labels').tag(
        config=True)

    number_of_z_labels = Integer(5, min=3, help='Number of Z labels').tag(
        config=True)

    number_of_contours = Integer(50, min=10, help='Number of contours').tag(
        config=True)

    contour_alpha = Float(1.00, min=0., max=1.0, help='Transparency of the ' \
                                                      'contours').tag(
        config=True)

    contour_start = Float(0.05, min=0.001, help='Fraction of the maximum '
                                                'for starting contour '
                                                'levels').tag(
        config=True)

    antialiased = Bool(True, help='antialiased option for surface plot').tag(
        config=True)

    rcount = Integer(50, help='rcount (steps in the row mode) for surface plot').tag(
        config=True)

    ccount = Integer(50, help='ccount (steps in the column mode) for surface plot').tag(
        config=True)

    # colors / style
    # ------------------------------------------------------------------------------------------------------------------

    pen_linewidth = Float(.70, min=0, help='Default pen width').tag(
        config=True)

    pen_color = Unicode('#000000', help='Default pen color for 1D plots').tag(
        config=True, type='color')

    pen_linestyle = Enum(linestyles, default_value=linestyles[0],
                         help='Default pen style for 1D plots').tag(
        config=True, type='list')

    marker = Enum(markers, default_value=markers[2],
                  help='Default symbols for 1D scatter plots').tag(config=True,
                                                                   type='list')

    markersize = Float(5.0, min=.5, help='Default symbol size for scatter '
                                         'plots').tag(config=True)

    colormap = Enum(cmaps, default_value='jet',
                    help='Default colormap for contour plots').tag(config=True,
                                                                   type='list')

    colormap_stack = Enum(cmaps, default_value='viridis',
                          help='Default colormap for stack plots').tag(
        config=True, type='list')

    colormap_surface = Enum(cmaps, default_value='jet',
                            help='Default colormap for surface plots').tag(
        config=True, type='list')

    colormap_transposed = Enum(cmaps, default_value='magma',
                               help='Default colormap for transposed stack '
                                    'plots').tag(config=True, type='list')


# ======================================================================================================================
# Application
# ======================================================================================================================


# ======================================================================================================================
class SpectroChemPy(Application):
    """
    This class SpectroChemPy is the main class, containing most of the setup,
    configuration and more.

    """

    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.initialize()

    # ------------------------------------------------------------------------------------------------------------------
    # applications attributes
    # ------------------------------------------------------------------------------------------------------------------

    icon = Unicode('scpy.png')
    "Icon for the application"

    running = Bool(False)
    "Running status of the |scpy| application"

    name = Unicode('SpectroChemPy')
    "Running name of the application"

    description = Unicode('SpectroChemPy is a '
                          'framework for processing, '
                          'analysing and modelling Spectroscopic data for '
                          'Chemistry with Python.')
    "Short description of the |scpy| application"

    long_description = Unicode
    "Long description of the |scpy| application"

    @default('long_description')
    def _get_long_description(self):
        desc = """
        Welcome to <strong>SpectroChemPy</strong> Application<br><br>
        <p><strong>SpectroChemPy</strong> is a framework for processing,
        analysing and
        modelling <strong>Spectro</>scopic data for <strong>Chem</strong>istry
        with
        <strong>Py</strong>thon. It is a cross platform software, running on
        Linux,
        Windows or OS X.</p><br><br>
        <strong>version:</strong> {version}<br>
        <strong>Authors:</strong> {authors}<br>
        <strong>License:</strong> {license}<br>
        <div class='warning'> SpectroChemPy is still experimental and under active
        development. Its current design and functionalities are subject to major
        changes, reorganizations, bugs and crashes!!!. Please report any issues
        to the
        <a url='https://bitbucket.org/spectrocat/spectrochempy'>Issue Tracker<a>
        </div><br><br>
        When using <strong>SpectroChemPy</strong> for your own work, you are
        kindly requested to cite it this way:
        <pre>
        Arnaud Travert & Christian Fernandez,
        SpectroChemPy, a framework for processing, analysing and modelling of
        Spectroscopic data for Chemistry with Python https://www.spectrochempy.fr, (version {version})
        Laboratoire Catalyse and Spectrochemistry, ENSICAEN/University of
        Caen/CNRS, 2019
        </pre>
        </p>
        """.format(version=__release__, authors=__author__, license=__license__)

        return desc

    # ------------------------------------------------------------------------------------------------------------------
    # Configuration parameters
    # ------------------------------------------------------------------------------------------------------------------

    # Config file setting
    # ------------------------------------------------------------------------------------------------------------------
    _loaded_config_files = List()

    reset_config = Bool(False, help='Should we restore a default '
                                    'configuration ?').tag(config=True)
    """Flag: True if one wants to reset settings to the original config
    defaults"""

    config_file_name = Unicode(None, help="Configuration file name").tag(
        config=True)
    """Configuration file name"""

    @default('config_file_name')
    def _get_config_file_name_default(self):
        return str(self.name).lower() + '_cfg'

    config_dir = Unicode(None,
                         help="Set the configuration directory location").tag(
        config=True)
    """Configuration directory"""

    @default('config_dir')
    def _get_config_dir_default(self):
        return self._get_config_dir()

    config_manager = Instance(BaseJSONConfigManager)

    @default('config_manager')
    def _get_default_config_manager(self):
        return BaseJSONConfigManager(config_dir=self.config_dir)

    # Logger at startup
    # ------------------------------------------------------------------------------------------------------------------

    debug = Bool(True, help='Set DEBUG mode, with full outputs').tag(
        config=True)
    """Flag to set debugging mode"""
    info = Bool(False, help='Set INFO mode, with msg outputs').tag(
        config=True)
    """Flag to set info mode"""
    quiet = Bool(False, help='Set Quiet mode, with minimal outputs').tag(
        config=True)
    """Flag to set in fully quite mode (even no warnings)"""
    nodisplay = Bool(False, help='Set NO DISPLAY mode, i.e., no graphics '
                                 'outputs').tag(config=True)
    """Flag to set in NO DISPLAY mode """

    # last project
    # ------------------------------------------------------------------------------------------------------------------

    last_project = Unicode('', help='Last used project').tag(config=True,
                                                             type='project')

    @observe('last_project')
    def _last_project_changed(self, change):
        # update configuration

        if change.name in self.traits(config=True):
            self.config_manager.update(self.config_file_name, {
                self.__class__.__name__: {change.name: change.new, }
            })

    # filename to load at startup
    # ------------------------------------------------------------------------------------------------------------------
    startup_filename = Unicode(os.path.join('irdata', 'nh4y-activation.spg'),
                               help='File name to load at startup').tag(
        config=True, type='file')

    # TESTING
    # ------------------------------------------------------------------------------------------------------------------

    show_config = Bool(help="Dump configuration to stdout at startup").tag(
        config=True)

    show_config_json = Bool(help="Dump configuration to stdout (as JSON)").tag(
        config=True)

    @observe('show_config_json')
    def _show_config_json_changed(self, change):
        self.show_config = change.new

    @observe('show_config')
    def _show_config_changed(self, change):
        if change.new:
            self._save_start = self.start
            self.start = self.start_show_config

    test = Bool(False, help='test flag').tag(config=True)
    """Flag to set the application in testing mode"""

    # Command line interface
    # ------------------------------------------------------------------------------------------------------------------

    aliases = Dict(
        dict(test='SpectroChemPy.test',
             p='SpectroChemPy.last_project',
             f='SpectroChemPy.startup_filename',
             ))

    flags = Dict(
        dict(
            debug=({'SpectroChemPy': {'log_level': DEBUG}},
                   "Set log_level to DEBUG - most verbose mode"),
            info=({'SpectroChemPy': {'log_level': INFO}},
                  "Set log_level to INFO - verbose mode"),
            quiet=({'SpectroChemPy': {'log_level': ERROR}},
                   "Set log_level to ERROR - no verbosity at all"),
            nodisplay=({'SpectroChemPy': {'nodisplay': True}},
                       "Set NO DISPLAY mode to true - no graphics at all"),
            reset_config=(
                {'SpectroChemPy': {'reset_config': True}}, "Reset config to default"),
            show_config=({'SpectroChemPy': {'show_config': True, }},
                         "Show the application's configuration (human-readable "
                         "format)"),
            show_config_json=({'SpectroChemPy': {'show_config_json': True, }},
                              "Show the application's configuration (json "
                              "format)"),
        ))

    classes = List([GeneralPreferences, ProjectPreferences, DataDir, ])

    # ------------------------------------------------------------------------------------------------------------------
    # Initialisation of the application
    # ------------------------------------------------------------------------------------------------------------------

    # @catch_config_error
    def initialize(self, argv=None):
        """
        Initialisation function for the API applications

        Parameters
        ----------
        argv :  List, [optional].
            List of configuration parameters.

        """

        # parse the argv
        # --------------------------------------------------------------------

        # if we are running this under ipython and jupyter notebooks
        # deactivate potential command line arguments
        # (such that those from jupyter which cause problems here)

        self.log.debug('initialization of SpectroChemPy')

        IN_IPYTHON = False
        if InteractiveShell.initialized():
            IN_IPYTHON = True

        self.log.debug("scpy command line arguments are: %s" % " ".join(sys.argv))

        # workaround the problem with argument not in our aliases or flags
        # e.g., when using pytest options or setup.py options

        options = []
        for item in sys.argv[:]:
            for k in list(self.flags.keys()):
                if item.startswith("--" + k) or k in ['--help', '--help-all']:
                    options.append(item)
                continue
            for k in list(self.aliases.keys()):
                if item.startswith("-" + k) or k in ['h', ]:
                    options.append(item)

        if not IN_IPYTHON:
            self.parse_command_line(options)

        # Get preferences from the config file and init everything
        # ---------------------------------------------------------------------

        self.init_all_preferences()

        # we catch warnings and error for a ligther display to the end-user.
        # except if we are in debugging mode

        # warning handler
        # --------------------------------------------------------------------
        def send_warnings_to_log(message, category, filename, lineno, *args):
            self.log.warning('%s:  %s' % (category.__name__, message))
            return

        warnings.showwarning = send_warnings_to_log

        # exception handler
        # --------------------------------------------------------------------

        if IN_IPYTHON:

            ip = get_ipython()

            def _custom_exc(shell, etype, evalue, tb, tb_offset=None):
                
                if self.log_level == logging.DEBUG:
                    shell.showtraceback((etype, evalue, tb),
                                        tb_offset=tb_offset)
                else:
                    self.log.error("%s: %s" % (etype.__name__, evalue))

            ip.set_custom_exc((Exception,), _custom_exc)

            # load our custom magic extensions
            # --------------------------------------------------------------------
            if ip is not None:
                ip.register_magics(SpectroChemPyMagics)

    def init_all_preferences(self):

        # Get preferences from the config file
        # ---------------------------------------------------------------------

        if self.config_file_name:
            config_file = os.path.join(self.config_dir, self.config_file_name)

            if self.reset_config:
                # remove the user json file to reset to defaults
                l = os.listdir(self.config_dir)
                for f in l:
                    if f.endswith('.json'):
                        jsonname = os.path.join(self.config_dir, f)
                        os.remove(jsonname)

            self.config = Config()

            for cfgname in [config_file, ]:
                self.load_config_file(cfgname)
                if cfgname not in self._loaded_config_files:
                    self._loaded_config_files.append(cfgname)

        # add other preferences
        # ---------------------------------------------------------------------

        self.datadir = DataDir(config=self.config)
        self.general_preferences = GeneralPreferences(config=self.config,
                                                      parent=self)
        self.project_preferences = ProjectPreferences(config=self.config,
                                                      parent=self)

        # Eventually write the default config file
        # --------------------------------------
        self._make_default_config_file()

    def start_show_config(self, **kwargs):
        """start function used when show_config is True"""
        config = self.config.copy()
        # exclude show_config flags from displayed config
        for cls in self.__class__.mro():
            if cls.__name__ in config:
                cls_config = config[cls.__name__]
                cls_config.pop('show_config', None)
                cls_config.pop('show_config_json', None)

        if self.show_config_json:
            json.dump(config, sys.stdout, indent=1, sort_keys=True,
                      default=repr)
            # add trailing newlines
            sys.stdout.write('\n')
            print()
            return self._start()

        if self._loaded_config_files:
            print("Loaded config files:")
            for f in self._loaded_config_files:
                print('  ' + f)
            print()

        for classname in sorted(config):
            class_config = config[classname]
            if not class_config:
                continue
            print(classname)
            pformat_kwargs = dict(indent=4)
            if sys.version_info >= (3, 4):
                # use compact pretty-print on Pythons that support it
                pformat_kwargs['compact'] = True
            for traitname in sorted(class_config):
                value = class_config[traitname]
                print('  .{} = {}'.format(traitname,
                                          pprint.pformat(value, **pformat_kwargs), ))
        print()

        # now run the actual start function
        return self._start()

    # ------------------------------------------------------------------------------------------------------------------
    # start the application
    # ------------------------------------------------------------------------------------------------------------------

    def start(self):
        """
        Start the |scpy| API

        All configuration must have been done before calling this function
        """
        return self._start()

    def _start(self):

        debug = self.log.debug

        if self.running:
            debug('API already started. Nothing done!')
            return

        self.log.debug("show info on loading %s" % self.general_preferences.show_info_on_loading)
        if self.general_preferences.show_info_on_loading:
            info_string = "SpectroChemPy's API - v.{}\n" \
                          "© Copyright {}".format(__version__, __copyright__)
            ip = get_ipython()
            if ip is not None and "TerminalInteractiveShell" not in str(ip):
                display_info_string(message=info_string.strip())

            else:
                print(info_string.strip())

        self.running = True

        debug('MPL backend: {}'.format(mpl.get_backend()))

        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = os.path.join(self.config_dir, self.config_file_name + '.py')

        if not os.path.exists(fname):
            s = self.generate_config_file()
            self.log.info("Generating default config file: %r" % fname)
            with open(fname, 'w') as f:
                f.write(s)

    # ..................................................................................................................
    @staticmethod
    def _find_or_create_spectrochempy_dir(directory):
        directory = os.path.join(os.path.expanduser('~'), '.spectrochempy',
                                 directory)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        elif not os.path.isdir(directory):
            msg = 'Intended SpectroChemPy directory `{0}` is ' \
                  'actually a file.'
            raise IOError(msg.format(directory))

        return os.path.abspath(directory)

    # ..................................................................................................................
    def _get_config_dir(self):
        """
        Determines the SpectroChemPy configuration directory name and
        creates the directory if it doesn't exist.

        This directory is typically ``$HOME/.spectrochempy/config``,
        but if the
        SCP_CONFIG_HOME environment variable is set and the
        ``$SCP_CONFIG_HOME`` directory exists, it will be that
        directory.

        If neither exists, the former will be created.

        Returns
        -------
        config_dir : str
            The absolute path to the configuration directory.

        """

        # first look for SCP_CONFIG_HOME
        scp = os.environ.get('SCP_CONFIG_HOME')

        if scp is not None and os.path.exists(scp):
            return os.path.abspath(scp)

        return os.path.abspath(
            self._find_or_create_spectrochempy_dir('config'))

    # ------------------------------------------------------------------------------------------------------------------
    # Events from Application
    # ------------------------------------------------------------------------------------------------------------------

    @observe('log_level')
    def _log_level_changed(self, change):

        self.log_format = '%(message)s'
        if change.new == DEBUG:
            self.log_format = '[%(filename)s-%(funcName)s %(levelname)s] %(' \
                              'message)s'
        self.log._cache = {}
        self.log.level = self.log_level
        for handler in self.log.handlers:
            handler.level = self.log_level
        self.log.info("changed default log_level to {}".format(
            logging.getLevelName(change.new)))


# ======================================================================================================================

if __name__ == "__main__":
    pass
