"""A simple example of how to use traitlets.config.application.Application.

This should serve as a simple example that shows how the traitlets config
system works. The main classes are:

* traitlets.config.Configurable
* traitlets.config.SingletonConfigurable
* traitlets.config.Config
* traitlets.config.Application

To see the command line option help, run this program from the command line::

    $ python myapp.py -h

To make one of your classes configurable (from the command line and config
files) inherit from Configurable and declare class attributes as traits (see
classes Foo and Bar below). To make the traits configurable, you will need
to set the following options:

* ``config``: set to ``True`` to make the attribute configurable.
* ``shortname``: by default, configurable attributes are set using the syntax
  "Classname.attributename". At the command line, this is a bit verbose, so
  we allow "shortnames" to be declared. Setting a shortname is optional, but
  when you do this, you can set the option at the command line using the
  syntax: "shortname=value".
* ``help``: set the help string to display a help message when the ``-h``
  option is given at the command line. The help string should be valid ReST.

When the config attribute of an Application is updated, it will fire all of
the trait's events for all of the config=True attributes.
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


# local imports
# =============================================================================

from spectrochempy.utils import get_config_dir
from spectrochempy.version import get_version

_version_, _release_, _copyright_ = get_version()

# =============================================================================
# Plot Options
# =============================================================================
class PlotOptions(Configurable):
    """
    All options relative to plotting and views

    """

    name = Unicode(u'PlotsOptions')

    description = Unicode(u'')

    USETEX = Bool(True,
                  help='should we use latex for plotting labels and texts?').tag(
        config=True)

    DO_NOT_BLOCK = Bool(False,
                        help="whether or not we show the plots "
                             "and stop after each of them").tag(config=True)

# =============================================================================
# Main application and configurators
# =============================================================================

class SpectroChemPyApplication(Application):

    # info

    name = Unicode(u'SpectroChemPy')
    description = Unicode(u'This is the main SpectroChemPy application ')

    VERSION = Unicode(_version_)
    RELEASE = Unicode(_release_)
    COPYRIGHT = Unicode(_copyright_)

    classes = List([PlotOptions,])

    # configuration -----------------------------------------------------

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

    def init_ndio(self):

        # Pass config to other classes for them to inherit the config.
        self.plotoptions = PlotOptions(config=self.config)


    @catch_config_error
    def initialize(self, argv=None):

        self.parse_command_line(argv)

        self.init_ndio()

        cl_config = deepcopy(self.config)

        if self.CONFIG_FILE_NAME :

            config_file = os.path.join(self.CONFIG_DIR, self.CONFIG_FILE_NAME)
            self.load_config_file(config_file)

        self.update_config(cl_config)


    def start(self, **kwargs):

        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

        if self.QUIET:
            self.log_level = logging.CRITICAL

        if self.DEBUG:
            self.log_level = logging.DEBUG

        self._make_default_config_file()

        self.RUNNING = True

    # private methods
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

SCP = SpectroChemPyApplication()
SCP.initialize()
SCP.start(
         RESET_CONFIG=True,
         DEBUG=True
         )


if __name__ == "__main__":


    # ==============================================================================
    # Logger
    # ==============================================================================
    log = SCP.log

    log.debug('Name : %s '%SCP.name)

    for app in ['make.py','pytest', 'py.test', 'docrunner.py',]:
        if  app in sys.argv[0]:
            # this is necessary to buid doc with sphinx-gallery and doctests
            SCP._DO_NOT_BLOCK = True

    DO_NOT_BLOCK = SCP.plotoptions.DO_NOT_BLOCK

    log.debug("DO NOT BLOCK : %s "%DO_NOT_BLOCK)

    usetex = SCP.plotoptions.USETEX
    print(usetex)