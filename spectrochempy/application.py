# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module define the `application` on which the API rely.


"""

__all__ = []

# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

import os
import glob
import sys
import logging
import subprocess
import datetime
import warnings
import pprint
import json

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

from pkg_resources import get_distribution, DistributionNotFound
from setuptools_scm import get_version
from traitlets.config.configurable import Configurable
from traitlets.config.application import Application, catch_config_error
from traitlets import (Bool, Unicode, List, Dict, default, observe,
                       import_item)
import matplotlib as mpl
from matplotlib import pyplot as plt
from IPython import get_ipython
from IPython.core.magic import (Magics, magics_class, line_cell_magic)
from IPython.core.magics.code import extract_symbols
from IPython.core.error import UsageError
from IPython.utils.text import get_text_list

from spectrochempy.utils import docstrings
from spectrochempy.core.projects.projectpreferences import ProjectPreferences
from spectrochempy.core.plotters.plotterpreferences import PlotterPreferences
from spectrochempy.core.processors.processorpreferences import \
    ProcessorPreferences
from spectrochempy.core.writers.writerpreferences import WriterPreferences
from spectrochempy.core.readers.readerpreferences import ReaderPreferences

# Log levels
# -----------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# ----------------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------------

try:
    __release__ = get_distribution('spectrochempy').version
    "Release version string of this package"
except DistributionNotFound:
    # package is not installed
    __release__ = '0.1.alpha'

try:
    __version__ = get_version(root='..', relative_to=__file__)
    "Version string of this package"
except:
    __version__ = __release__


# ............................................................................
def _get_copyright():
    current_year = datetime.date.today().year
    copyright = '2014-{}'.format(current_year)
    copyright += ' - A.Travert and C.Fernandez @ LCS'
    return copyright


__copyright__ = _get_copyright()
"Copyright string of this package"


# .............................................................................
def _get_release_date():
    try:
        return subprocess.getoutput(
            "git log -1 --tags --date='short' --format='%ad'")
    except:
        pass


__release_date__ = _get_release_date()
"Last release date of this package"

# other info
# ............................................................................

__url__ = "http://www-lcs.ensicaen.fr/spectrochempy"
"URL for the documentation of this package"

__author__ = "C. Fernandez & A. Travert @LCS"
"First authors(s) of this package"

__contributor__ = ""
"contributor(s) to this package"

__license__ = "CeCILL-B license"
"Licence of this package"


# ============================================================================
# Magic ipython function
# ============================================================================
@magics_class
class SpectroChemPyMagics(Magics):
    """
    This class implements the addscript ipython magic function.

    """

    @line_cell_magic
    def addscript(self, pars='', cell=None):
        """This works both as **%addscript** and as **%%addscript**

        This magic command can either take a local filename, element in the
        namespace or history range (see %history),
        or the current cell content


        Usage
            %addscript  -p project  n1-n2 n3-n4 ... n5 .. n6 ...

             or

            %%addscript -p project
            ...code lines ...


        Options
            -p <string>         Name of the project where the script will be
            stored.
                                If not provided, a project with a standard
                                name:
                                `proj` is searched.
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
        # print(opts)
        # print(args)
        # print(cell)

        append = 'a' in opts
        mode = 'a' if append else 'w'
        search_ns = 'n' in opts

        if not args and not cell and not search_ns:
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
        if proj not in self.shell.user_ns:
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

        if 's' in opts:
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

        from spectrochempy.core.scripts.script import Script  # import

        # delayed to avoid circular import error
        script = Script(name, content=contents)
        projobj[name] = script

        return "Script {} created.".format(name)

        # @line_magic
        # def runscript(self, pars=''):
        #     """
        #
        #     """
        #     opts, args = self.parse_options(pars, '')
        #
        #     if not args:
        #         raise UsageError('Missing script name')
        #
        #     return args


# ============================================================================
# DataDir class
# ============================================================================

def _get_pkg_datadir_path(data_name, package=None):
    data_name = os.path.normpath(data_name)

    path = os.path.dirname(import_item(package).__file__)
    path = os.path.join(path, data_name)

    if not os.path.isdir(path):
        return os.path.dirname(path)

    return path


class DataDir(Configurable):
    """ A configurable class used to determine the path to the testdata
    directory. """

    path = Unicode().tag(config=True)
    "Directory where to look for data"

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __str__(self):
        return self.listing()

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------

    @default('path')
    def _get_path_default(self):
        # the spectra path in package data
        return _get_pkg_datadir_path('testdata', 'scp_data')

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    def _repr_html_(self):
        # _repr_html is needed to output in notebooks
        return self.listing().replace('\n', '<br/>').replace(" ", "&nbsp;")


# ============================================================================
# Main application and configurators
# ============================================================================
class Preferences(Configurable):
    """
    Preferences that apply to the |scpy| application in general

    They should be accessible from the main API

    Examples
    --------

    >>> import spectrochempy as scp # doctest: +ELLIPSIS
    SpectroChemPy's API...
    >>> delimiter = scp.preferences.csv_delimiter


    """

    def __init__(self, **kwargs):
        super(Preferences, self).__init__(**kwargs)

    # various settings
    # ----------------

    show_info_on_loading = Bool(True, help='Display info on loading?').tag(
        config=True)

    csv_delimiter = Unicode(';', help='CSV data delimiter').tag(config=True)

    startup_filename = Unicode('',
      help='Name of the file to load at startup').tag(config=True, type='file')

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


class SpectroChemPy(Application):
    """
    This class SpectroChemPy is the main class, containing most of the setup,
    configuration and more.

    """

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        super(SpectroChemPy, self).__init__(*args, **kwargs)

        if kwargs.get('debug', False):
            self.log_level = DEBUG

        self.initialize()

    # ------------------------------------------------------------------------
    # applications attributes
    # ------------------------------------------------------------------------

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
        desc = """\
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
    kindly 
    requested to cite it this way:
    <pre>
     Arnaud Travert & Christian Fernandez,
     SpectroChemPy, a framework for processing, analysing and modelling of 
     Spectroscopic data for Chemistry with Python
     https://bitbucket.org/spectrocat/spectrochempy, (version {version})
     Laboratoire Catalyse and Spectrochemistry, ENSICAEN/University of
     Caen/CNRS, 2018
    </pre>
    </p>

    """.format(version=__release__, authors=__author__, license=__license__)

        return desc

    # ------------------------------------------------------------------------
    # Configuration parameters
    # ------------------------------------------------------------------------

    # Config file setting
    # -------------------
    _loaded_config_files = List()

    reset_config = Bool(False, help='Should we restore a default '
                                    'configuration?').tag(config=True)
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

    # Logger at startup
    # -----------------

    debug = Bool(False, help='Set DEBUG mode, with full outputs').tag(
        config=True)
    """Flag to set debugging mode"""
    quiet = Bool(False, help='Set Quiet mode, with minimal outputs').tag(
        config=True)
    """Flag to set in fully quite mode (even no warnings)"""


    # TESTING
    # --------

    show_config = Bool(
        help="Instead of starting the Application, dump configuration to stdout"
    ).tag(config=True)

    show_config_json = Bool(
        help="Instead of starting the Application, dump configuration to stdout (as JSON)"
    ).tag(config=True)

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

    do_not_block = Bool(False)
    "Flag to make the plots BUT do not stop for TESTS or DOCS building"

    # Command line interface
    # ----------------------

    aliases = Dict(
        dict(test='SpectroChemPy.test',
             p='ProjectPreferences.startup_project',
             f='Preferences.startup_filename'))

    flags = Dict(
        dict(
            debug=({'SpectroChemPy': {'log_level': DEBUG}},
                   "Set log_level to DEBUG - most verbose mode"),
            quiet=({'SpectroChemPy': {'log_level': ERROR}},
                   "Set log_level to ERROR - no verbosity at all"),
            show_config=({'SpectroChemPy': {'show_config': True,}},
    "Show the application's configuration (human-readable format)"),
            show_config_json=({'SpectroChemPy': {'show_config_json': True,}},
    "Show the application's configuration (json format)"), ))

    classes = List(
        [Preferences, ProjectPreferences, PlotterPreferences, DataDir, ])

    # ------------------------------------------------------------------------
    # Initialisation of the application
    # ------------------------------------------------------------------------

    @catch_config_error
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

        _do_parse = True
        for arg in ['egg_info', '--egg-base', 'pip-egg-info', 'develop', '-f',
                    '-x', '-c']:
            if arg in sys.argv:
                _do_parse = False

        if _do_parse:
            self.parse_command_line(sys.argv)

        # Get preferences from the config file and init everything
        # ---------------------------------------------------------------------

        self.init_all_preferences()

        # Test, Sphinx,  ...  detection
        # ---------------------------------------------------------------------

        for caller in ['builddocs.py', '-c']:
            # `-c` happen if the pytest is executed in parallel mode
            # using the plugin pytest-xdist

            if caller in sys.argv[0]:
                # this is necessary to build doc
                # with sphinx-gallery and doctests
                plt.ioff()
                self.do_not_block = True
                break

        for caller in ['pytest', 'py.test']:

            if caller in sys.argv[0]:
                # let's set do_not_block flag to true only if we are running
                #  the whole suite of tests
                if len(sys.argv) > 1 and sys.argv[1].endswith("tests"):
                    plt.ioff()
                    self.do_not_block = True

        # case we have passed -test arguments to a script
        if len(sys.argv) > 1 and "-test" in sys.argv[1]:
            plt.ioff()
            self.do_not_block = True

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
        ip = get_ipython()
        if ip is not None:

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
            config_file = os.path.join(self.config_dir,
                                       self.config_file_name)
            self.load_config_file(config_file)
            if config_file not in self._loaded_config_files:
                self._loaded_config_files.append(config_file)

        # add other preferences
        # ---------------------------------------------------------------------

        self._init_preferences()
        self._init_datadir()
        self._init_plotter_preferences()
        self._init_project_preferences()
        self._init_processor_preferences()
        self._init_reader_preferences()
        self._init_writer_preferences()

        # Possibly write the default config file
        # --------------------------------------------------------------------
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
            json.dump(config, sys.stdout,
                      indent=1, sort_keys=True, default=repr)
            # add trailing newline
            sys.stdout.write('\n')
            return

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
            if sys.version_info >= (3,4):
                # use compact pretty-print on Pythons that support it
                pformat_kwargs['compact'] = True
            for traitname in sorted(class_config):
                value = class_config[traitname]
                print('  .{} = {}'.format(
                    traitname,
                    pprint.pformat(value, **pformat_kwargs),
                ))
    # ------------------------------------------------------------------------
    # start the application
    # ------------------------------------------------------------------------

    @docstrings.get_sectionsf('SpectroChemPy.start')
    @docstrings.dedent
    def start(self, **kwargs):
        """
        Start the |scpy| API

        Parameters
        ----------
        debug : bool
            Set application in debugging mode (log debug message
            are displayed in the standard output console).
        quiet : bool
            Set the application in minimal messaging mode. Only errors are
            displayed (bu no warnings). If both debug and quiet are set
            (which is contradictory) debug has the priority.
        reset_config : bool
            Reset the configuration file to default values.

        """

        try:

            if self.running:
                self.log.debug('API already started. Nothing done!')
                return

            for key in list(kwargs.keys()):
                if hasattr(self, key):
                    setattr(self, key, kwargs[key])

            self.log_format = '%(highlevel)s %(message)s'

            if self.quiet:
                self.log_level = ERROR

            if self.debug:
                self.log_level = DEBUG
                self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(' \
                                  'message)s'

            info_string = "SpectroChemPy's API - v.{}\n" \
                          "© Copyright {}".format(__version__, __copyright__)

            # print(self.preferences.show_info_on_loading)
            if self.preferences.show_info_on_loading:
                print(info_string)

            self.log.debug(
                "The application was launched with ARGV : %s" % str(sys.argv))

            self.running = True

            self.log.debug('MPL backend: {}'.format(mpl.get_backend()))

            return True

        except:

            return False

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    # ........................................................................
    def _init_preferences(self):

        self.preferences = Preferences(config=self.config, parent=self)

    # ........................................................................
    def _init_datadir(self):

        self.datadir = DataDir(config=self.config)

    # ........................................................................
    def _init_project_preferences(self):

        self.project_preferences = ProjectPreferences(config=self.config)

    # ........................................................................
    def _init_plotter_preferences(self):

        self.plotter_preferences = PlotterPreferences(config=self.config)

    def _init_reader_preferences(self):

        self.reader_preferences = ReaderPreferences(config=self.config)

    def _init_writer_preferences(self):

        self.writer_preferences = WriterPreferences(config=self.config)

    def _init_processor_preferences(self):

        self.processor_preferences = ProcessorPreferences(config=self.config)

    # ........................................................................
    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = os.path.join(self.config_dir,
                                           self.config_file_name+'.py')

        if not os.path.exists(fname) or self.reset_config:
            s = self.generate_config_file()
            self.log.warning("Generating default config file: %r" % fname)
            with open(fname, 'w') as f:
                f.write(s)




    # ........................................................................

    # ........................................................................
    @staticmethod
    def _get_config_dir(create=True):
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

        return os.path.abspath(_find_or_create_spectrochempy_dir('config'))


    # ------------------------------------------------------------------------
    # Events from Application
    # ------------------------------------------------------------------------

    @observe('log_level')
    def _log_level_changed(self, change):

        self.log_format = '%(highlevel)s %(message)s'
        if change.new == DEBUG:
            self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'
        self.log.level = self.log_level
        for handler in self.log.handlers:
            handler.level = self.log_level
        self.log.debug("changed default log_level to {}".format(
            logging.getLevelName(change.new)))


# Main application object that should not be called directly by a end user.
# It is advisable to use the main `scp` import to access all public methods of
# this object.
app = SpectroChemPy()

log = app.log
preferences = app.preferences
plotter_preferences = app.plotter_preferences
project_preferences = app.project_preferences
processor_preferences = app.processor_preferences
reader_preferences = app.reader_preferences
writer_preferences = app.writer_preferences
do_not_block = app.do_not_block
datadir = app.datadir
description = app.description
long_description = app.long_description

"""The main logger of the |scpy| application"""

# TODO: look at the subcommands capabilities of traitlets

if __name__ == "__main__":

    pass
