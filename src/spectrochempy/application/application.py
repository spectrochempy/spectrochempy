# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Defines the `application` on which the API rely."""

import inspect
import io
import json
import logging
import sys
import traceback
import warnings
from contextlib import contextmanager
from os import environ
from pathlib import Path

import traitlets as tr
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config.application import Application
from traitlets.config.configurable import Config
from traitlets.config.manager import BaseJSONConfigManager

__all__ = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "error_",
    "info_",
    "debug_",
    "warning_",
    "NO_DISPLAY",
    "get_loglevel",
    "set_loglevel",
    "get_config_dir",
]


# --------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------
def get_config_dir():
    """
    Determine the SpectroChemPy configuration directory name and creates the directory if it doesn't exist.

    This directory is typically `$HOME/.spectrochempy/config` ,
    but if the
    SCP_CONFIG_HOME environment variable is set and the
    `$SCP_CONFIG_HOME` directory exists, it will be that
    directory.

    If neither exists, the former will be created.

    Returns
    -------
    config_dir : str
        The absolute path to the configuration directory.

    """
    from spectrochempy.utils.file import find_or_create_spectrochempy_dir

    # first look for SCP_CONFIG_HOME
    scp = environ.get("SCP_CONFIG_HOME")

    if scp is not None and Path(scp).exists():
        return Path(scp)

    config = find_or_create_spectrochempy_dir() / "config"
    if not config.exists():
        config.mkdir(exist_ok=True)

    return config


def get_log_dir():
    """
    Get log directory name or create SpectroChemPy log output file directory.

    This directory is typically ``$HOME/.spectrochempy/logs``,
    but if the
    SCPY_LOGS environment variable is set and the
    ``$SCPY_LOGS`` directory exists, it will be that
    directory.

    If neither exists, the former will be created.

    Returns
    -------
    log_dir : str
        The absolute path to the log directory.
    """
    from spectrochempy.utils.file import find_or_create_spectrochempy_dir

    # first look for SCP_LOGS
    log_dir = environ.get("SCP_LOGS")

    if log_dir is not None and Path(log_dir).exists():  # pragma: no cover
        return Path(log_dir)

    log_dir = find_or_create_spectrochempy_dir() / "logs"
    if not log_dir.exists():  # pragma: no cover
        log_dir.mkdir(exist_ok=True)

    return log_dir


# ======================================================================================
# Application
# ======================================================================================
class SpectroChemPy(Application):
    """
    SpectroChemPy is the main class, containing most of the setup.

    Configuration and more.
    """

    name = tr.Unicode("SpectroChemPy")
    description = tr.Unicode("Main application")

    # ----------------------------------------------------------------------------------
    # Private attributes
    # ----------------------------------------------------------------------------------
    _running = tr.Bool(False)
    _loaded_config_files = tr.List()
    _from_warning_ = False

    # ----------------------------------------------------------------------------------
    # Non configurable attributes
    # ----------------------------------------------------------------------------------

    # config
    config_file_name = tr.Unicode(None, help="Configuration file name")
    config_dir = tr.Instance(Path, help="Set the configuration directory location")
    config_manager = tr.Instance(BaseJSONConfigManager)

    @tr.default("config_dir")
    def _config_dir_default(self):
        return get_config_dir()

    @tr.default("log_dir")
    def _log_dir_default(self):
        return get_log_dir()

    @tr.default("config_manager")
    def _config_manager_default(self):
        return BaseJSONConfigManager(config_dir=str(self.config_dir))

    reset_config = tr.Bool(False, help="Should we restore a default configuration ?")

    # ----------------------------------------------------------------------------------
    # Configurables attributes
    # ----------------------------------------------------------------------------------

    # logger
    log_dir = tr.Instance(Path, help="The log output directory location").tag(
        config=True
    )
    log_format = tr.Unicode(
        "%(highlevel)s %(message)s",
        help="The Logging format template",
    ).tag(config=True)
    logging_config = tr.Dict(
        {
            "handlers": {
                "string": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": "INFO",
                    "stream": io.StringIO(),
                },
            },
            "loggers": {
                "SpectroChemPy": {
                    "level": "DEBUG",
                    "handlers": ["console", "string"],
                },
            },
        }
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Error/warning capture
    # ----------------------------------------------------------------------------------
    def _ipython_catch_exceptions(self, shell, etype, evalue, tb, tb_offset=None):
        # output the full traceback only in DEBUG mode or when under pytest
        # if self.log_level == logging.DEBUG:
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        # else:
        #    self.log.error(f"{etype.__name__}: {evalue}")

    def _catch_exceptions(self, etype, evalue, tb=None):
        # output the full traceback only in DEBUG mode
        with self._fmtcontext():
            # if self.log_level == logging.DEBUG:
            # print(etype, type(etype))
            if isinstance(etype, str):
                # probably the type was not provided!
                evalue = etype
                etype = Exception
            self.log.error(f"{etype.__name__}: {evalue}")
            if tb:
                format_exception = traceback.format_tb(tb)
                for line in format_exception:
                    parts = line.splitlines()
                    for p in parts:
                        self.log.error(p)
        # else:
        #    self.log.error(f"{etype.__name__}: {evalue}")

    def _custom_warning(
        self,
        message,
        category,
        filename,
        lineno,
        file=None,
        line=None,
    ):
        with self._fmtcontext():
            self._formatter(message)
            self.log.warning(f"({category.__name__}) {message}")

    def _formatter(self, *args):
        # do not format if not DEBUG mod
        if self.log_level != logging.DEBUG:
            return
        # We need a custom formatter (maybe there is a better way to do this suing
        # the logging library directly?)

        rootfolder = Path(__file__).parent
        st = 2
        if "_showwarnmsg" in inspect.stack()[2][3]:
            st = 4 if self._from_warning_ else 3

        filename = Path(inspect.stack()[st][1])
        try:
            module = filename.relative_to(rootfolder)
        except ValueError:
            module = filename
        line = inspect.stack()[st][2]
        func = inspect.stack()[st][3]

        # rotatingfilehandler formatter (DEBUG)
        formatter = logging.Formatter(
            f"<%(asctime)s:{module}/{func}::{line}> %(message)s"
        )
        self.log.handlers[1].setFormatter(formatter)

    @contextmanager
    def _fmtcontext(self):
        fmt = self.log_format, self.log.handlers[1].formatter
        try:
            yield fmt
        finally:
            self.log_format = fmt[0]
            self.log.handlers[1].setFormatter(fmt[1])

    # ----------------------------------------------------------------------------------
    # Initialisation of the configurables
    # ----------------------------------------------------------------------------------
    classes = tr.List(help="List of configurables")

    def _init_all_preferences(self):
        # Get preferences from the config files
        # ---------------------------------------------------------------------
        from spectrochempy.application._preferences.general_preferences import (
            GeneralPreferences,
        )
        from spectrochempy.application._preferences.plot_preferences import (
            PlotPreferences,
        )

        if not self.config:
            self.config = Config()

        configfiles = []
        if self.config_dir:
            lis = self.config_dir.iterdir()
            for fil in lis:
                if fil.suffix == ".py":
                    pyname = self.config_dir / fil
                    if self.reset_config:
                        # remove the py file to reset to defaults
                        pyname.unlink()
                    else:
                        configfiles.append(pyname)
                elif fil.suffix == ".json":
                    jsonname = self.config_dir / fil
                    if self.reset_config or fil == "PlotPreferences.json":
                        # remove the user json file to reset to defaults
                        jsonname.unlink()
                    else:
                        # check integrity of the file
                        with jsonname.open() as f:
                            try:
                                json.load(f)
                            except json.JSONDecodeError:
                                jsonname.unlink()
                                continue
                        configfiles.append(jsonname)

            for cfgname in configfiles:
                self.load_config_file(cfgname)
                if cfgname not in self._loaded_config_files:
                    self._loaded_config_files.append(cfgname)

        self.general_preferences = GeneralPreferences(config=self.config, parent=self)
        self.plot_preferences = PlotPreferences(config=self.config, parent=self)
        self.classes = [GeneralPreferences, PlotPreferences]

    # ----------------------------------------------------------------------------------
    # Initialisation of the application
    # ----------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log_level = kwargs.pop("log_level", None)
        self._initialize()
        if log_level is not None:
            self.log_level = log_level

    def _initialize(self):
        # Initialize the API applications.
        # warning handler
        # ----------------
        warnings.showwarning = self._custom_warning

        # exception handler
        # -----------------
        ipy = get_ipython() if InteractiveShell.initialized() else None

        if ipy is not None:  # pragma: no cover
            ipy.set_custom_exc((Exception,), self._ipython_catch_exceptions)
        else:
            # if environ.get("SCPY_TESTING", 0) == 0 and "pytest" not in sys.argv[0]:
            #    # catch exception only when pytest is not running
            sys.excepthook = self._catch_exceptions

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    def start(self):
        """
        Start the  `SpectroChemPy` API.

        All configuration must have been done before calling this function.
        """
        from matplotlib import pyplot as plt

        from spectrochempy.application.datadir import DataDir

        if self._running:
            # API already started. Nothing done!
            return True

        # datadir
        self.datadir = DataDir()

        # Get preferences from the config file and init everything
        self._init_all_preferences()

        # force update of rcParams
        import matplotlib as mpl

        for rckey in mpl.rcParams:
            key = rckey.replace("_", "__").replace(".", "_").replace("-", "___")
            try:
                mpl.rcParams[rckey] = getattr(self.plot_preferences, key)
            except ValueError:
                mpl.rcParams[rckey] = getattr(self.plot_preferences, key).replace(
                    "'",
                    "",
                )
            except AttributeError:
                # print(f'{e} -> you may want to add it to PlotPreferences.py')
                pass

        self.plot_preferences.set_latex_font(self.plot_preferences.font_family)

        # Eventually write the default config file
        # --------------------------------------
        self.make_default_config_file()

        # set the default style
        # --------------------------------------------------------------------------------------
        plt.style.use(["classic"])

        self.debug_(
            f"API loaded with log level set to "
            f"{logging.getLevelName(int(self.log_level))}- application is ready"
        )

        self._running = True
        return True

    def make_default_config_file(self, configurables=None):
        """Auto generate default config file."""
        # remove old configuration file spectrochempy_cfg.py
        # --------------------------------------------------
        fname = self.config_dir / "spectrochempy_cfg.py"  # Old configuration file
        fname2 = self.config_dir / "SpectroChemPy.cfg.py"
        if fname.exists() or fname2.exists():
            for file in list(self.config_dir.iterdir()):
                file.unlink()

        # create a configuration file for each configurables
        # --------------------------------------------------

        # first we will complete self.classes with a list of configurable in analysis
        if configurables:
            self.classes.extend(configurables)

        config_classes = list(self._classes_with_config_traits(self.classes))
        for cls in config_classes:
            name = cls.__name__
            fname = self.config_dir / f"{name}.py"
            if fname.exists() and not self.reset_config:
                continue
            """generate default config file from Configurables"""
            lines = [f"# Configuration file for SpectroChemPy::{name}"]
            lines.append("")
            lines.append("c = get_config()  # noqa")
            lines.append("")
            lines.append(cls.class_config_section([cls]))
            sfil = "\n".join(lines)
            self.log.info(f"Generating default config file: {fname}")
            with open(fname, "w") as fil:
                fil.write(sfil)

    # Logging methods
    def info_(self, msg, *args, **kwargs):
        """Format an info message."""
        with self._fmtcontext():
            self._formatter(msg)
            self.log.info(msg, *args, **kwargs)

    def debug_(self, msg, *args, **kwargs):
        """Format a debug message."""
        with self._fmtcontext():
            self._formatter(msg)
            self.log.debug("DEBUG | " + msg, *args, **kwargs)

    def error_(self, *args, **kwargs):
        """Format an error message."""
        if isinstance(args[0], Exception):
            e = args[0]
            etype = type(e)
            emessage = str(e)
        elif len(args) == 1 and isinstance(args[0], str):
            from spectrochempy.utils import exceptions

            etype = exceptions.SpectroChemPyError
            emessage = str(args[0])
        elif len(args) == 2:
            etype = args[0] if args else kwargs.get("type")
            emessage = args[1] if args and len(args) > 1 else kwargs.get("message")
        else:
            raise KeyError("wrong arguments have been passed to error_")
        self._catch_exceptions(etype, emessage, None)

    def warning_(self, msg, *args, **kwargs):
        """Format a warning message."""
        self._from_warning_ = True
        warnings.warn(msg, *args, **kwargs, stacklevel=2)
        self._from_warning_ = False


# --------------------------------------------------------------------------------------
# Setup environment
from .envsetup import setup_environment

NO_DISPLAY, SCPY_STARTUP_LOGLEVEL, is_pytest = setup_environment()

# Define an instance of the SpectroChemPy application.
app = SpectroChemPy(log_level=SCPY_STARTUP_LOGLEVEL)

# Start the application
app.start()

error_ = app.error_
info_ = app.info_
debug_ = app.debug_
warning_ = app.warning_

# Log levels
# ----------

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


# --------------------------------------------------------------------------------------
# Add some utility functions and variables.
# --------------------------------------------------------------------------------------
def set_loglevel(level: str | int = logging.WARNING) -> None:
    """
    Set the logging level for SpectroChemPy.

    Parameters
    ----------
    level : Union[str, int]
        Logging level (e.g. 'WARNING', 'DEBUG', etc. or logging constants)

    """
    from spectrochempy.application.preferences import preferences

    if isinstance(level, str):
        level = getattr(logging, level.upper())
    preferences.log_level = level


def get_loglevel() -> int:
    """
    Get current logging level.

    Returns
    -------
    int
        Current logging level

    """
    from spectrochempy.application.preferences import preferences

    return preferences.log_level


# --------------------------------------------------------------------------------------
# Check for new release in a separate thread
# --------------------------------------------------------------------------------------
import threading

if not NO_DISPLAY:
    from .check_update import check_update
    from .info import version

    check_update_frequency = app.general_preferences.check_update_frequency
    DISPLAY_UPDATE = threading.Thread(
        target=check_update, args=(version, check_update_frequency)
    )

    DISPLAY_UPDATE.start()

# --------------------------------------------------------------------------------------
# Download data in a separate thread
# --------------------------------------------------------------------------------------
if not is_pytest:
    from .testdata import download_full_testdata_directory

    DOWNLOAD_TESTDATA = threading.Thread(
        target=download_full_testdata_directory,
        args=(app.general_preferences.datadir,),
    )

    DOWNLOAD_TESTDATA.start()
