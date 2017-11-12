# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
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
This module define the `application` on which the API rely


"""

# ==============================================================================
# standard library import
# ==============================================================================

import os
import glob
import sys
import logging
import warnings

# ==============================================================================
# third imports
# ==============================================================================

from traitlets.config.configurable import Configurable
from traitlets.config.application import Application, catch_config_error

from traitlets import (HasTraits, Instance,
                       Bool, Unicode, Int, List, Dict, default, observe
                       )

from IPython.core.magic import UsageError
from IPython import get_ipython

import matplotlib as mpl

# ==============================================================================
# local imports
# ==============================================================================

from spectrochempy.utils import is_kernel
from spectrochempy.utils import get_config_dir, get_pkg_data_dir, get_version
from spectrochempy.utils import get_pkg_data_filename
from spectrochempy.utils import install_styles

from spectrochempy.core.plotters.plottersoptions import PlotOptions
from spectrochempy.core.readers.readersoptions import ReadOptions
from spectrochempy.core.writers.writersoptions import WriteOptions
from spectrochempy.core.processors.processorsoptions import ProcessOptions

# doc info
# --------
_classes = [
    'SCPData',
    'SpectroChemPy',
]

__all__ = [

    # ## Helpers
    'log', 'log_level', 'DEBUG', 'WARN', 'ERROR', 'CRITICAL', 'INFO',
    'scpdata', 'list_scpdata',
    'options', 'plotoptions',
    'running',
    # 'pcl',

    # ## Info
    'copyright', 'version',
]

# add iparallel client

# pcl = ipp.Client()[:]  #TODO: parallelization


# ==============================================================================
# SCP class
# ==============================================================================

class SCPData(Configurable):
    """
    This class is used to determine the path to the scp_data directory.

    {attributes}

    Examples
    --------
    >>> scpdata = SCPData()
    >>> print(os.path.basename(scpdata.data))
    testdata
    >>> print(scpdata) # doctest: +ELLIPSIS
    testdata
    |__...
    <BLANKLINE>

    """

    data = Unicode(help="Directory where to look for data").tag(config=True)

    _data = Unicode()

    def listing(self):
        """
        Create a `str` representing a listing of the data repertory.

        Returns
        -------
        listing : `str`

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
        return get_pkg_data_dir('testdata', 'scp_data')


# ==============================================================================
# Main application and configurators
# ==============================================================================

class SpectroChemPy(Application):
    """
    SpectroChemPy is the main class, containing most of the setup, configuration,
    and more.

    {attributes}


    """

    # info ____________________________________________________________________

    name = Unicode('SpectroChemPy')
    description = Unicode('This is the main SpectroChemPy application ')

    version = Unicode('0.1').tag(config=True)

    @default('version')
    def _get_version(self):

        return get_version()

    copyright = Unicode('').tag(config=True)

    @default('copyright')
    def _get_copyright(self):
        copyright = '2014-2017'  # TODO put current year%
        copyright += ' - LCS (Laboratory for Catalysis and Spectrochempy)'
        return copyright

    classes = List([PlotOptions, ])

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

    _scpdata = Instance(SCPData,
                     help="Set a data directory where to look for data")

    csv_delimiter = Unicode(';',
                 help='set csv delimiter').tag(config=True)

    @default('_scpdata')
    def _get__data_default(self):
        # look for the testdata path in package tests
        return SCPData()

    @property
    def scpdata(self):
        return self._scpdata.data

    @property
    def list_scpdata(self):
        return self._scpdata


    aliases = Dict(
        dict(test='SpectroChemPy.test', log_level='SpectroChemPy.log_level'))

    flags = Dict(dict(
                      debug=(
                      {'SpectroChemPy': {'log_level': 10}}, "Set loglevel to DEBUG")
                      ))

    # --------------------------------------------------------------------------
    # Initialisation of the plot options
    # --------------------------------------------------------------------------

    def _init_plotoptions(self):

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

        Returns
        -------
        application : Application.
            The application handler.

        """
        # matplotlib use directive to set before calling matplotlib backends
        # ---------------------------------------------------------------------
        # we performs this before any call to matplotlib that are performed
        # later in this application


        # if we are building the docs, in principle it should be done using
        # the builddocs.py located in the scripts folder
        if not 'builddocs.py' in sys.argv[0]:
            # the normal backend
            backend = mpl.get_backend()
            if backend == 'module://ipykernel.pylab.backend_inline':
                mpl.use('Qt5Agg')
                mpl.rcParams['backend.qt5'] = 'PyQt5'
        else:
            # 'agg' backend is necessary to build docs with sphinx-gallery
            mpl.use('agg')

        ip = get_ipython()
        if ip is not None:
            #backend = mpl.get_backend()
            if is_kernel():
                # and backend == 'module://ipykernel.pylab.backend_inline':

                # set the ipython matplotlib environments
                try:
                    import ipympl
                    ip.magic('matplotlib notebook')  # nbagg')
                    #if 'IPython' in sys.modules:
                    #from IPython.core.pylabtools import backend2gui
                    #backend2gui['module://ipympl.backend_nbagg'] = 'ipympl'
                    #mpl.use('module://ipympl.backend_nbagg')
                    #
                    #print('OK FOR THE BACKEND')

                except UsageError as e:
                    print(e)
                    try:
                        ip.magic('matplotlib osx')
                    except:
                        try:
                            ip.magic('matplotlib qt5')
                        except:
                            pass

            else:
                try:
                    ip.magic('matplotlib osx')
                except:
                    try:
                        ip.magic('matplotlib qt5')
                    except:
                        pass

        # parse the argv
        # ---------------------------------------------------------------------

        # if we are running this under ipython and jupyter notebooks
        # deactivate potential command line arguments
        # (such that those from jupyter which cause problems here)

        _do_parse = True
        for arg in ['egg_info', '--egg-base',
                    'pip-egg-info', 'develop', '-f', '-x']:
            if arg in sys.argv:
                _do_parse = False

        # print("*" * 50, "\n", sys.argv, "\n", "*" * 50)

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

        for caller in ['builddocs.py', 'pytest', 'py.test']:

            if caller in sys.argv[0]:
                # this is necessary to build doc
                # with sphinx-gallery and doctests

                _do_not_block = self.plotoptions.do_not_block = True
                break
                
        # case we have passed -test arguments to a script
        if len(sys.argv)>1 and "-test" in sys.argv[1]:
            _do_not_block = self.plotoptions.do_not_block = True
            caller = sys.argv[0]

        if _do_not_block:
            self.log.warning(
                    'Running {} - set do_not_block: {}'.format(
                            caller, _do_not_block))

        # Possibly write the default config file
        # ---------------------------------------------------------------------
        self._make_default_config_file()

        # we catch warnings and error for a lighter display to the end-user.
        # except if we are in debugging mode

        # warning handler
        # ---------------------------------------------------------------------
        def send_warnings_to_log(message, category, filename, lineno,
                                 *args):
            self.log.warning(
                    '%s:  %s' %
                    (category.__name__, message))
            return

        warnings.showwarning = send_warnings_to_log

        # exception handler
        # ---------------------------------------------------------------------
        if ip is not None:

            def _custom_exc(shell, etype, evalue, tb, tb_offset=None):
                if self.log_level == logging.DEBUG:
                    shell.showtraceback((etype, evalue, tb),
                                        tb_offset=tb_offset)
                else:
                    self.log.error("%s: %s" % (etype.__name__, evalue))

            ip.set_custom_exc((Exception,), _custom_exc)


    # --------------------------------------------------------------------------
    # start the application
    # --------------------------------------------------------------------------

    def start(self, **kwargs):
        """

        Parameters
        ----------
        kwargs : options to pass to the application.

        Examples
        --------
        >>> app = SpectroChemPy()
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
                self.log_level = logging.CRITICAL

            if self.debug:
                self.log_level = logging.DEBUG
                self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'

            info_string = """
        SpectroChemPy's API
        Version   : {}
        Copyright : {}
            """.format(self.version, self.copyright)

            if self.info_on_loading and \
                    not self.plotoptions.do_not_block:
                print(info_string)

            self.log.debug(
                    "The application was launched with ARGV : %s" % str(
                            sys.argv))


            self.running = True

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

# ==============================================================================
# matplotlib use directive to set before calling matplotlib backends
# ==============================================================================

app = SpectroChemPy()
app.initialize()

# ==============================================================================
# API namespace
# ==============================================================================

running = app.running
version = app.version
copyright = app.copyright
log = app.log
log_level = app.log_level

# give a user friendly name to the objects containing configurables options
plotoptions = app.plotoptions
options = app

_do_not_block = plotoptions.do_not_block

scpdata = app.scpdata
list_scpdata = app.list_scpdata

# Log levels
# -----------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# TODO: look at the subcommands capabilities of traitlets
if __name__ == "__main__":
    pass
