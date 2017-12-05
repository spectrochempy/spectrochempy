# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================



"""
This module define the `application` on which the API rely.

This module has no public members and so is not intended to be
accessed directly by the end user.



"""

# ==============================================================================
# standard library import
# ==============================================================================

import os
import glob
import sys
import logging
import warnings

from traitlets.config.configurable import Configurable
from traitlets.config.application import Application, catch_config_error
from traitlets import (Instance, Bool, Unicode, List, Dict, default, observe)
from IPython import get_ipython
import matplotlib as mpl

# ==============================================================================
# third party imports
# ==============================================================================

# ==============================================================================
# local imports
# ==============================================================================

__all__ = ['app']  # no public methods
#__slots__ = ['app']
# add iparallel client

# pcl = ipp.Client()[:]  #TODO: parallelization



# ==============================================================================
# SCP class
# ==============================================================================

class _SCPData(Configurable):
    """
    This class is used to determine the path to the scp_data directory.

    """

    data = Unicode(help="Directory where to look for data").tag(config=True)

    _data = Unicode()

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

    def __str__(self):
        return self.listing()

    def _repr_html_(self):
        # _repr_html is needed to output in notebooks
        return self.listing().replace('\n', '<br/>').replace(" ", "&nbsp;")

    @default('data')
    def _get_data_default(self):
        # return the spectra dir by default
        return self._data

    @default('_data')
    def _get__data_default(self):
        # the spectra path in package data
        from spectrochempy.utils import get_pkg_data_dir
        return get_pkg_data_dir('testdata', 'scp_data')


# ==============================================================================
# Main application and configurators
# ==============================================================================

class _SpectroChemPy(Application):
    """
    _SpectroChemPy is the main class, containing most of the setup, configuration,
    and more.



    """
    from spectrochempy.utils import docstrings

    from spectrochempy.plotters.plottersoptions import PlotOptions
    #from spectrochempy.readers.readersoptions import ReadOptions
    #from spectrochempy.writers.writersoptions import WriteOptions
    #from spectrochempy.processors.processorsoptions import ProcessOptions

    # info ____________________________________________________________________

    name = Unicode('SpectroChemPyApp')
    description = Unicode('This is the main SpectroChemPy Application ')

    version = Unicode('0.1').tag(config=True)
    dev_version = Unicode('0.1').tag(config=True)
    release = Unicode('0.1').tag(config=True)

    @default('version')
    def _get_version(self):
        from spectrochempy.utils import get_version
        return get_version()[0]

    @default('release')
    def _get_release(self):
        from spectrochempy.utils import get_version
        return get_version()[1]

    @default('dev_version')
    def _get_dev_version(self) :
        from spectrochempy.utils import get_version
        if len(get_version())>2:
            return get_version()[2]
        else:
            return get_version()[0]

    copyright = Unicode('')

    @default('copyright')
    def _get_copyright(self):
        from spectrochempy.utils import get_copyright
        return get_copyright()[3]

    classes = List([PlotOptions, ])  # TODO: check if this still usefull

    # configuration parameters  ________________________________________________

    reset_config = Bool(False,
                        help='should we restaure a default configuration?').tag(
        config=True)

    config_file_name = Unicode(None,
                               help="Load this config file").tag(config=True)

    @default('config_file_name')
    def _get_config_file_name_default(self):
        return self.name + '_config.py'

    config_dir = Unicode(None,
                         help="Set the configuration dir location").tag(
        config=True)

    @default('config_dir')
    def _get_config_dir_default(self):
        from spectrochempy.utils import get_config_dir
        return get_config_dir()

    info_on_loading = Bool(True,
                           help='display info on loading').tag(config=True)

    running = Bool(False,
                   help="Is SpectrochemPy running?").tag(config=True)

    test = Bool(False,
                help='set application in testing mode').tag(config=True)

    debug = Bool(False,
                 help='set DEBUG mode, with full outputs').tag(config=True)

    quiet = Bool(False,
                 help='set Quiet mode, with minimal outputs').tag(config=True)

    workspace = Unicode(os.path.expanduser(os.path.join('~', 'scpworkspace'))
                        ).tag(config=True)

    _scpdata = Instance(_SCPData,
                        help="Set a data directory where to look for data")

    csv_delimiter = Unicode(';',
                            help='set csv delimiter').tag(config=True)

    @default('_scpdata')
    def _get__data_default(self):
        return _SCPData()

    @property
    def scpdata(self):
        return self._scpdata.data

    @property
    def list_scpdata(self):
        return self._scpdata

    aliases = Dict(
        dict(test='SpectroChemPy.test',
             log_level='SpectroChemPy.log_level'))

    flags = Dict(dict(
        debug=(
            {'SpectroChemPy': {'log_level': 10}},
            "Set loglevel to DEBUG")
    ))

    backend = Unicode('spectrochempy', help='backend to be used in the '
                                            'application').tag(config=True)

    # --------------------------------------------------------------------------
    # Initialisation of the plot options
    # --------------------------------------------------------------------------

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

    def __init__(self, *args, **kwargs):
        super(_SpectroChemPy, self).__init__(*args, **kwargs)
        if kwargs.get('debug', False):
            self.log_level = logging.DEBUG

        self.initialize()

    # --------------------------------------------------------------------------
    # Initialisation of the application
    # --------------------------------------------------------------------------

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
        # ---------------------------------------------------------------------

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
        # ---------------------------------------------------------------------
        def send_warnings_to_log(message, category, filename,
                                 lineno,
                                 *args):
            self.log.warning(
                '%s:  %s' %
                (category.__name__, message))
            return

        warnings.showwarning = send_warnings_to_log

        # exception handler
        # ---------------------------------------------------------------------
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

        # Possibly write the default config file
        # ---------------------------------------------------------------------
        self._make_default_config_file()

    # --------------------------------------------------------------------------
    # start the application
    # --------------------------------------------------------------------------

    @docstrings.get_sectionsf('SpectroChemPy.start')
    @docstrings.dedent
    def start(self, **kwargs):
        """
        Start the |scp| API or only make a plot if an `output` filename is given

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
        >>> app = _SpectroChemPy()
        >>> app.initialize()
        >>> app.start(
        ...    reset_config=True,   # option for restoring default configuration
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

            info_string = "SpectroChemPy's API - v.{}, " \
                          "Copyright {}".format(self.version, self.copyright)

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

    # --------------------------------------------------------------------------
    # Store default configuration file
    # --------------------------------------------------------------------------

    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = config_file = os.path.join(self.config_dir,
                                           self.config_file_name)

        if not os.path.exists(fname) or self.reset_config:
            s = self.generate_config_file()
            self.log.warning("Generating default config file: %r" % fname)
            with open(fname, 'w') as f:
                f.write(s)

    # --------------------------------------------------------------------------
    # Events from Application
    # --------------------------------------------------------------------------

    @observe('log_level')
    def _log_level_changed(self, change):
        self.log_format = '%(highlevel)s %(message)s'
        if change.new == logging.DEBUG:
            self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'
        self.log.level = self.log_level
        self.log.debug("changed default loglevel to {}".format(change.new))

#: Main application object that should not be called directly by a end user.
#: It is advisable to use the main `api` import to access all public methods of
#: this object.
app = _SpectroChemPy()

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
