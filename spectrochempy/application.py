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

This module has no public members and so is not intended to be
accessed directly by the end user.

"""

# ============================================================================
# standard library import
# ============================================================================

import os
import glob
import sys
import logging
import warnings

# ============================================================================
# third party imports
# ============================================================================

from traitlets.config.configurable import Configurable
from traitlets.config.application import Application, catch_config_error
from traitlets import (Instance, Bool, Unicode, List, Dict, default, observe,
                       import_item)
import matplotlib as mpl
from setuptools_scm import get_version

from IPython import get_ipython
from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic,
                                line_cell_magic)
from IPython.core.magics.code import extract_symbols
from IPython.core.error import UsageError
from IPython.utils.text import get_text_list

# ============================================================================
# constants
# ============================================================================

__all__ = ['app']  # no public methods

# ============================================================================
# Magic ipython function
# ============================================================================
@magics_class
class SpectroChemPyMagics(Magics):

    @line_cell_magic
    def addscript(self, pars='', cell=None):
        """This works both as %addscript and as %%addscript

        This magic command can either take a local filename, element in the
        namespace or history range (see %history),
        or the current cell content

        %addscript myscript.py
        %addscript 7-27
        %addscript -s MyClass,myfunction myscript.py
        %addscript MyClass
        %addscript mymodule.myfunction

        Usage:\\
          %addscript  -p project  n1-n2 n3-n4 ... n5 .. n6 ...

        or

        Usage:\\
          %%addscript -p project
          ...code lines ...

        Options:

          -p <string>: Name of the project where the script will be stored.
                       If not provided, a project with a standard name:
                       `proj` is searched.
          -o <string>: script name

          -s <symbols>: Specify function or classes to load from python source.

          -a : append to the current script instead of overwriting it.

          -n : search symbol in the current namespace

        """
        opts, args = self.parse_options(pars, 'p:o:s:n:a')
        #print(opts)
        #print(args)
        #print(cell)

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
        if not proj in self.shell.user_ns:
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
                warnings.warn('The symbols %s were not found' % get_text_list(
                    not_found, wrap_item_with='`'))

            contents = '\n'.join(blocks)

        if cell:
            contents += "\n" + cell

        from spectrochempy.scripts.script import Script
        script = Script(name, content=contents)
        projobj[name]=script

        return "Script {} created.".format(name)

    @line_magic
    def runscript(self, pars=''):
        """

        """
        opts, args = self.parse_options(pars, '')

        if not args:
            raise UsageError('Missing script name')




        return args

# ==============================================================================
# SCPData class
# ==============================================================================

class SCPData(Configurable):
    """
    This class is used to determine the path to the scp_data directory.

    """

    data = Unicode(help="Directory where to look for data").tag(config=True)

    _data = Unicode()

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

    def listing(self):
        """
        Create a str representing a listing of the data repertory.

        Returns
        -------
        listing : str

        """
        s = os.path.basename(self.data) + "\n"

        def _listdir(s, initial, ns):
            ns += 1
            for f in glob.glob(os.path.join(initial, '*')):
                fb = os.path.basename(f)
                if not fb.startswith('acqu') and \
                        not fb.startswith('pulse') and fb not in ['ser', 'fid']:
                    s += "   " * ns + "|__" + "%s\n" % fb
                if os.path.isdir(f):
                    s = _listdir(s, f, ns)
            return s

        return _listdir(s, self.data, -1)

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __str__(self):
        return self.listing()

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------

    @default('data')
    def _get_data_default(self):
        # return the spectra dir by default
        return self._data

    @default('_data')
    def _get__data_default(self):
        # the spectra path in package data
        return self._get_pkg_data_dir('testdata', 'scp_data')


    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    def _repr_html_(self):
        # _repr_html is needed to output in notebooks
        return self.listing().replace('\n', '<br/>').replace(" ", "&nbsp;")

    def _get_pkg_data_dir(self, data_name, package=None) :

        data_name = os.path.normpath(data_name)

        datadir = os.path.dirname(import_item(package).__file__)
        datadir = os.path.join(datadir, data_name)

        if not os.path.isdir(datadir) :
            return os.path.dirname(datadir)

        return datadir


# ============================================================================
# Main application and configurators
# ============================================================================

class SpectroChemPy(Application):
    """
    _SpectroChemPy is the main class, containing most of the setup,
    configuration and more.

    """
    from spectrochempy.utils import docstrings

    from spectrochempy.projects.projectsoptions import ProjectsOptions
    from spectrochempy.plotters.plottersoptions import PlotOptions
    #from spectrochempy.readers.readersoptions import ReadOptions
    #from spectrochempy.writers.writersoptions import WriteOptions
    #from spectrochempy.processors.processorsoptions import ProcessOptions


    name = Unicode('SpectroChemPyApp')
    description = Unicode('This is the main SpectroChemPy Application ')

    version = Unicode
    dev_version = Unicode
    release = Unicode
    copyright = Unicode

    # configuration parameters  ______________________________________________

    reset_config = Bool(False,
                        help='should we restaure a default configuration?'
                        ).tag(config=True)

    config_file_name = Unicode(None,
                               help="Load this config file"
                               ).tag(config=True)

    config_dir = Unicode(None,
                         help="Set the configuration dir location"
                         ).tag(config=True)

    backend = Unicode('spectrochempy',
                      help='backend to be used in the application'
                      ).tag(config=True)

    info_on_loading = Bool(True,
                           help='display info on loading'
                           ).tag(config=True)

    running = Bool(False,
                   help="Is SpectrochemPy running?"
                   ).tag(config=True)

    test = Bool(False,
                help='set application in testing mode'
                ).tag(config=True)

    debug = Bool(False,
                 help='set DEBUG mode, with full outputs'
                 ).tag(config=True)

    quiet = Bool(False,
                 help='set Quiet mode, with minimal outputs'
                 ).tag(config=True)

    _scpdata = Instance(SCPData,
                        help="Set a data directory where to look for data")

    csv_delimiter = Unicode(';',
                            help='set csv delimiter').tag(config=True)

    project = Unicode('', help='project to load at startup').tag(config=True)

    aliases = Dict(
        dict(test='SpectroChemPy.test',
             p='SpectroChemPy.project',
             log_level='SpectroChemPy.log_level'))

    flags = Dict(dict(
        debug=(
            {'SpectroChemPy': {'log_level': 10}},
            "Set loglevel to DEBUG")
    ))

    classes = List([PlotOptions,
                    ProjectsOptions])  # TODO: check if this still usefull

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------

    @default('version')
    def _get_version(self):
        from spectrochempy import __version__
        return __version__

    @default('release')
    def _get_release(self):
        from spectrochempy import __release__
        return __release__

    @default('copyright')
    def _get_copyright(self):
        from spectrochempy import __copyright__
        return __copyright__

    @default('config_file_name')
    def _get_config_file_name_default(self):
        return self.name + '_config.py'


    @default('config_dir')
    def _get_config_dir_default(self):
        return self._get_config_dir()

    @default('_scpdata')
    def _get__data_default(self):
        return SCPData()

    @property
    def scpdata(self):
        return self._scpdata.data

    @property
    def list_scpdata(self):
        return self._scpdata


    def __init__(self, *args, **kwargs):
        super(SpectroChemPy, self).__init__(*args, **kwargs)
        if kwargs.get('debug', False):
            self.log_level = logging.DEBUG

        self.initialize()

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
        for arg in ['egg_info', '--egg-base',
                    'pip-egg-info', 'develop', '-f', '-x', '-c']:
            if arg in sys.argv:
                _do_parse = False

        if _do_parse:
            self.parse_command_line(sys.argv)

        # Get options from the config file
        # ---------------------------------------------------------------------

        if self.config_file_name:
            config_file = os.path.join(self.config_dir, self.config_file_name)
            self.load_config_file(config_file)

        # add other options
        # ---------------------------------------------------------------------

        self._init_plotoptions()
        self._init_projectsoptions()

        # Test, Sphinx,  ...  detection
        # ---------------------------------------------------------------------

        _do_not_block = self.plotoptions.do_not_block

        for caller in ['builddocs.py', 'pytest', 'py.test', '-c']:
            # `-c` happen if the pytest is executed in parallel mode
            # using the plugin pytest-xdist

            if caller in sys.argv[0]:
                # this is necessary to build doc
                # with sphinx-gallery and doctests

                _do_not_block = self.plotoptions.do_not_block = True
                break

        # case we have passed -test arguments to a script
        if len(sys.argv) > 1 and "-test" in sys.argv[1]:
            _do_not_block = self.plotoptions.do_not_block = True
            caller = sys.argv[0]

        if _do_not_block:
            self.log.warning(
                'Running {} - set do_not_block: {}'.format(
                    caller, _do_not_block))

        # we catch warnings and error for a ligther display to the end-user.
        # except if we are in debugging mode

        # warning handler
        # --------------------------------------------------------------------
        def send_warnings_to_log(message, category, filename,
                                 lineno,
                                 *args):
            self.log.warning(
                '%s:  %s' %
                (category.__name__, message))
            return

        warnings.showwarning = send_warnings_to_log

        # exception handler
        # --------------------------------------------------------------------
        ip = get_ipython()
        if ip is not None:

            def _custom_exc(shell, etype, evalue, tb,
                            tb_offset=None):
                if self.log_level == logging.DEBUG:
                    shell.showtraceback((etype, evalue, tb),
                                        tb_offset=tb_offset)
                else:
                    self.log.error(
                        "%s: %s" % (etype.__name__, evalue))

            ip.set_custom_exc((Exception,), _custom_exc)

        # load our custom magic extensions
        # --------------------------------------------------------------------
        if ip is not None:
            ip.register_magics(SpectroChemPyMagics)

        # Possibly write the default config file
        # --------------------------------------------------------------------
        self._make_default_config_file()

    # ------------------------------------------------------------------------
    # start the application
    # ------------------------------------------------------------------------

    @docstrings.get_sectionsf('SpectroChemPy.start')
    @docstrings.dedent
    def start(self, **kwargs):
        """
        Start the |scp| API or only make a plot if an `output` filename is
        given.

        Parameters
        ----------
        debug : `bool`
            Set application in debugging mode (log debug message
            are displayed in the standart output console).
        quiet : `bool`
            Set the application in minimal messaging mode. Only errors are
            displayed (bu no warnings). If both bebug and quiet are set
            (which is contradictory) debug has the priority.
        reset_config : `bool`
            Reset the configuration file to default values.

        Examples
        --------
        >>> app = SpectroChemPy()
        >>> app.initialize()
        >>> app.start(
        ...    reset_config=True,   # for restoring default configuration
        ...    debug=True,          # debugging logs
        ...    )
        True

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
                self.log_level = logging.ERROR

            if self.debug:
                self.log_level = logging.DEBUG
                self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'

            info_string = "SpectroChemPy's API - v.{}\n" \
                          "© Copyright {}".format(self.version, self.copyright)

            if self.info_on_loading and \
                    not self.plotoptions.do_not_block:
                print(info_string)

            self.log.debug(
                "The application was launched with ARGV : %s" % str(
                    sys.argv))

            self.running = True

            self.log.debug('MPL backend: {}'.format(mpl.get_backend()))

            return True

        except:

            return False

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    # ........................................................................
    def _init_plotoptions(self):
        from spectrochempy.plotters.plottersoptions import PlotOptions
        from spectrochempy.utils import install_styles

        # Pass config to other classes for them to inherit the config.
        self.plotoptions = PlotOptions(config=self.config)

        if not self.plotoptions.latex_preamble:
            self.plotoptions.latex_preamble = [
                r'\usepackage{siunitx}',
                r'\sisetup{detect-all}',
                r'\usepackage{times}',  # set the normal font here
                r'\usepackage{sansmath}',
                # load up the sansmath so that math -> helvet
                r'\sansmath'
            ]

        # also install style to be sure everything is set
        install_styles()

    # ........................................................................
    def _init_projectsoptions(self):

        from spectrochempy.projects.projectsoptions import ProjectsOptions

        self.projectsoptions = ProjectsOptions(config=self.config)

    # ........................................................................
    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = config_file = os.path.join(self.config_dir,
                                           self.config_file_name)

        if not os.path.exists(fname) or self.reset_config:
            s = self.generate_config_file()
            self.log.warning("Generating default config file: %r" % fname)
            with open(fname, 'w') as f:
                f.write(s)

    # ........................................................................
    def _find_or_create_spectrochempy_dir(self, directory) :

        directory = os.path.join(os.path.expanduser('~'),
                                 '.spectrochempy', directory)

        if not os.path.exists(directory) :
            os.makedirs(directory, exist_ok=True)
        elif not os.path.isdir(directory) :
            msg = 'Intended SpectroChemPy directory `{0}` is ' \
                  'actually a file.'
            raise IOError(msg.format(directory))

        return os.path.abspath(directory)

    # ........................................................................
    def _get_config_dir(self, create=True) :
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

        if scp is not None and os.path.exists(scp) :
            return os.path.abspath(scp)

        return os.path.abspath(
            self._find_or_create_spectrochempy_dir('config'))

    # ------------------------------------------------------------------------
    # Events from Application
    # ------------------------------------------------------------------------

    @observe('log_level')
    def _log_level_changed(self, change):

        self.log_format = '%(highlevel)s %(message)s'
        if change.new == logging.DEBUG:
            self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'
        self.log.level = self.log_level
        for handler in self.log.handlers:
            handler.level = self.log_level
        self.log.debug("changed default log_level to {}".format(
                                             logging.getLevelName(change.new)))

#: Main application object that should not be called directly by a end user.
#: It is advisable to use the main `api` import to access all public methods of
#: this object.
app = SpectroChemPy()

# Log levels
# -----------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


# TODO: look at the subcommands capabilities of traitlets
if __name__ == "__main__":
    pass

