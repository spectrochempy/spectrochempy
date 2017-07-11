# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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
This module define the application on which the API rely


"""

import os
import sys
import logging
from copy import deepcopy

from traitlets.config.configurable import Configurable
from traitlets.config.application import Application, catch_config_error

from traitlets import (
    Bool, Unicode, Int, List, Dict, default, observe
)
import matplotlib as mpl

from IPython.core.magic import UsageError
from IPython import get_ipython

# local imports
# =============================================================================

from spectrochempy.utils import is_kernel
from spectrochempy.utils import get_config_dir
from spectrochempy.version import get_version

# For wild importing using the *, we limit the methods, obetcs, ...
# that this method exposes
# ------------------------------------------------------------------------------
__all__ = ['scp']

# ==============================================================================
# PYTHONPATH
# ==============================================================================
# in case spectrochempy was not yet installed using setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# Plot Options
# =============================================================================
class PlotOptions(Configurable):
    """
    All options relative to plotting and views

    """
    name = Unicode(u'PlotsOptions')

    description = Unicode(u'')

    # -------------------------------------------------------------------------

    USE_LATEX = Bool(True, help='should we use latex for plotting labels and texts?').tag(
        config=True)

    @observe('USE_LATEX')
    def _USE_LATEX_changed(self, change):
        mpl.rc('text', usetex=change.new)

    # -------------------------------------------------------------------------

    LATEX_PREAMBLE = List(mpl.rcParams['text.latex.preamble'],
                          help='latex preamble for matplotlib outputs'
                          ).tag(config=True)

    @observe('LATEX_PREAMBLE')
    def _set_LATEX_PREAMBLE(self, change):
        mpl.rcParams['text.latex.preamble'] = change.new

    # -------------------------------------------------------------------------

    DO_NOT_BLOCK = Bool(False,
                        help="whether or not we show the plots "
                             "and stop after each of them").tag(config=True)



# ==============================================================================
# Main application and configurators
# ==============================================================================

class SpectroChemPy(Application):

    # info _____________________________________________________________________

    name = Unicode(u'SpectroChemPy')
    description = Unicode(u'This is the main SpectroChemPy application ')

    VERSION = Unicode('').tag(config=True)
    RELEASE = Unicode('').tag(config=True)
    COPYRIGHT = Unicode('').tag(config=True)

    classes = List([PlotOptions,])

    # configuration parameters  ________________________________________________

    RESET_CONFIG = Bool(False,
            help='should we restaure a default configuration?').tag(config=True)

    CONFIG_FILE_NAME = Unicode(None,
                                  help="Load this config file").tag(config=True)

    @default('CONFIG_FILE_NAME')
    def _set_config_file_name_default(self):
        return self.name + u'_config.py'

    CONFIG_DIR = Unicode(None,
                     help="Set the configuration dir location").tag(config=True)

    @default('CONFIG_DIR')
    def _set_config_dir_default(self):
        return get_config_dir()

    INFO_ON_LOADING = Bool(True,
                                help='display info on loading').tag(config=True)

    RUNNING = Bool(False,
                   help="Is SpectrochemPy running?").tag(config=True)

    DEBUG = Bool(False,
                 help='set DEBUG mode, with full outputs').tag(config=True)

    QUIET = Bool(False,
                 help='set Quiet mode, with minimal outputs').tag(config=True)


    # --------------------------------------------------------------------------
    # Initialisation of the plot options
    # --------------------------------------------------------------------------

    def _init_plotoptions(self):

        # Pass config to other classes for them to inherit the config.
        self.plotoptions = PlotOptions(config=self.config)

        # set default matplotlib options
        mpl.rc('text', usetex=self.plotoptions.USE_LATEX)

        if self.plotoptions.LATEX_PREAMBLE == []:
            self.plotoptions.LATEX_PREAMBLE = [
                r'\usepackage{siunitx}',
                r'\sisetup{detect-all}',
                r'\usepackage{times}',  # set the normal font here
                r'\usepackage{sansmath}',
                # load up the sansmath so that math -> helvet
                r'\sansmath'
            ]

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
        app : Application.
            The application handler.

        """
        # matplotlib use directive to set before calling matplotlib backends
        # ------------------------------------------------------------------
        # we performs this before any call to matplotlib that are performed
        # later in this application

        import matplotlib as mpl

        # if we are building the docs, in principle it should be done using
        # the make.py located in the scripts folder
        if not 'make.py' in sys.argv[0]:
            # the normal backend
            mpl.use('Qt5Agg')
            mpl.rcParams['backend.qt5'] = 'PyQt5'
        else:
            # 'agg' backend is necessary to build docs with sphinx-gallery
            mpl.use('agg')

        ip = get_ipython()
        if ip is not None:

            if is_kernel():

                # set the ipython matplotlib environments
                try:
                    ip.magic('matplotlib nbagg')
                except UsageError:
                    try:
                        ip.magic('matplotlib osx')
                    except:
                        ip.magic('matplotlib qt')
            else:
                try:
                    ip.magic('matplotlib osx')
                except:
                    ip.magic('matplotlib qt')

        # parse the argv
        # --------------

        # if we are running this under ipython and jupyter notebooks
        # deactivate potential command line arguments
        # (such that those from jupyter which cause problems here)

        _do_parse = True
        for arg in ['egg_info', '--egg-base', 'pip-egg-info', 'develop', '-f', '-x']:
            if arg in sys.argv:
                _do_parse = False

        # print("*" * 50, "\n", sys.argv, "\n", "*" * 50)

        if _do_parse:
            self.parse_command_line(argv)

        # Get options from the config file
        # --------------------------------

        if self.CONFIG_FILE_NAME :

            config_file = os.path.join(self.CONFIG_DIR, self.CONFIG_FILE_NAME)
            self.load_config_file(config_file)

        # add other options
        # -----------------

        self._init_plotoptions()

        # Test, Sphinx,  ...  detection
        # ------------------------------

        _DO_NOT_BLOCK = self.plotoptions.DO_NOT_BLOCK

        for caller in ['make.py', 'pytest', 'py.test', 'docrunner.py']:

            if caller in sys.argv[0]:

                # this is necessary to build doc
                # with sphinx-gallery and doctests

                _DO_NOT_BLOCK = self.plotoptions.DO_NOT_BLOCK = True
                self.log.warning(
                    'Running {} - set DO_NOT_BLOCK: {}'.format(
                                                         caller, _DO_NOT_BLOCK))

        self.log.debug("DO NOT BLOCK : %s " % _DO_NOT_BLOCK)

        # version
        # --------

        self.VERSION, self.RELEASE, self.COPYRIGHT = get_version()

        # Possibly write the default config file
        # ---------------------------------------
        self._make_default_config_file()

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
            RESET_CONFIG=True,   # option for restoring default configuration
            DEBUG=True,          # debugging logs
            )

        """
        if self.RUNNING:
            self.log.debug('API already started. Nothing done!')
            return

        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

        self.log_format = '%(highlevel)s %(message)s'

        if self.QUIET:
            self.log_level = logging.CRITICAL

        if self.DEBUG:
            self.log_level = logging.DEBUG
            self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'


        info_string = u"""
    SpectroChemPy's API
    Version   : {}
    Copyright : {}
        """.format(self.VERSION, self.COPYRIGHT)

        if self.INFO_ON_LOADING and \
                not self.plotoptions.DO_NOT_BLOCK:

            print(info_string)
            self.log.debug("argv0 : %s" % str(sys.argv[0]))

        self.RUNNING = True

    # --------------------------------------------------------------------------
    # Store default configuration file
    # --------------------------------------------------------------------------
    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = config_file = os.path.join(self.CONFIG_DIR,
                                           self.CONFIG_FILE_NAME)

        if not os.path.exists(fname) or self.RESET_CONFIG:
            s = self.generate_config_file()
            self.log.warning("Generating default config file: %r"%(fname))
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

scp = SpectroChemPy()
scp.initialize()

#TODO: look at the subcommands capabilities of traitlets
if __name__ == "__main__":

    scp.start(
            RESET_CONFIG=True,
            log_level = logging.INFO,
    )

    # ==============================================================================
    # Logger
    # ==============================================================================
    log = scp.log

    log.info('Name : %s ' % scp.name)

    scp.plotoptions.USE_LATEX = True

    log.info(scp.plotoptions.LATEX_PREAMBLE)
