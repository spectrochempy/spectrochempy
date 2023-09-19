# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module defines the `application` on which the API rely.

It also defines
the default application preferences and IPython magic functions.
"""

import importlib
import io
import json
import logging
import pprint
import subprocess
import sys
import threading
import traceback
import warnings
from os import environ
from pathlib import Path
from zipfile import ZipFile

import matplotlib as mpl
import numpy as np
import requests
import traitlets as tr
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output, publish_display_data
from jinja2 import Template
from setuptools_scm import get_version
from traitlets.config.application import Application
from traitlets.config.configurable import Config
from traitlets.config.manager import BaseJSONConfigManager

from spectrochempy.application.datadir import DataDir
from spectrochempy.application.general_preferences import GeneralPreferences
from spectrochempy.utils.file import find_or_create_spectrochempy_dir, pathclean

# ======================================================================================
# Setup
# ======================================================================================

# --------------------------------------------------------------------------------------
# Log levels
# --------------------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


# --------------------------------------------------------------------------------------
# logo / copyright display
# --------------------------------------------------------------------------------------
def _display_info_string(**kwargs):  # pragma: no cover
    _template = """
    {{widgetcss}}
    <div>
    <table>
    <tr>
    <td>
    {% if logo %}
    <img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAAXNSR0IArs4c6QAAAAlw
    SFlzAAAJOgAACToB8GSSSgAAAetpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6
    bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8x
    OTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAg
    eG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMu
    YWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDx4bXA6Q3JlYXRvclRvb2w+bWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBo
    dHRwOi8vbWF0cGxvdGxpYi5vcmcvPC94bXA6Q3JlYXRvclRvb2w+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6
    T3JpZW50YXRpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgqNQaNYAAAGiUlE
    QVRIDY1We4xU1Rn/3XPuYx47u8w+hnU38hTcuoUEt/6D2y4RB0ME1BoEd9taJaKh9CFiN7YGp7appUAMNmktMZFoJTYVLVQ0smsy
    26CN0SU1QgsuFAaW3WVmx33N677O6XfuyoIxTXqSO/fec+75fd93vt/3/UbDV0aKSZmCpkFMLz3T9utuu2N+o98aDSMBKVAo89z5
    y+zEz3ZafcCOfvWdlGCalqKn1Bf71CygTd+mf1esSOnpdMpTb+vWpTZuWVfe3jLPa5tzHYNm0T5N0gpdkkHaDBeGBU6d1/t/fyS8
    +/CbqdfUvmsx1PuMgc2bNxv79u1zgd31r+7JH1jbIZKxWRXAcYUQ8IWvBfBXNjEuJWPgMA02NR7C3/pYT9fjdZ3A9tGrWF8YSJHn
    qcDz3y7q2T967PZv+gnYJdd1mEZ+62zGDQV/dQgKhmLzDNOXCEWM3j6eTT5Y3w78dOBKJLR1PQf+4ivPj76UPZnssBN+wbM9Aet/
    AV81Mf1EEULXYfOobvX2WWQk0aoioXwwSmirOlioY0mu8BIouzYl7P8GV3vpqCCEZvlFz769w08oLDWvyKIyL1asSm28d6WfzA97
    ztvvV1kexUMsmhlkULEkuGYmFYC6AvfUrITnwUKl5K79lkjeSSRRTCTbQPd95e1WzMbZSya74XoXAxctCllCnbECMOjZNGRwvzIX
    nD85wbkMmKK+U045Dtdi8Qp+SAxU2GTg2bYlC9224pgvmSb54vkVTBQYyhUt2KjAMyMmPjwRQW5Mh2WKwJhlBh6jVGagFM84wZnQ
    4bpC0Rt4pk1PbSt0NDcxDA5xryosDHWgtbM0DGZDWLSoiDMDYeQnGVrmOThxLozB0RAaahzkJzjKNqcIQBymJFMkOlN8Dqjpg0XY
    Tx5xO/QbmmUrqIjGJznq47TqTaClKYfjp+PInLMwnOdYvtQBZ2XcunQY+VwIo4U4muoFEjVEFE6lQyEUKzHYfgQG9ylCyngU+Cxj
    tOqxCDGHcCsOMCs6iQul5ZiStdATYxjMZXDLTUVwLY8Jey4uOh2IxjwsrP8UXJYxUrkZrghBahzV5iXU6gNkq0Z1EzIsUBUSCV2n
    EOHo0LVxHCpuxabJJdhi5PFnvw5vLXwXIfNZvD/+JNo/X40NegE54sUaazl+UL8XD1x+FB9Ijjt4EQfdGN6J/x131LwIV9ap/AYs
    0x1fz1ZKFbh6A7qKy/By9Dg6G36Ep91vUJJ15Cqr0Z67E8/HzmBrw1OwxWyM+3Mo6BAuSB17oyfx0Oyl2DN0Hqs/70Cx6hBCvESF
    UY1ShWXZZEE7OTAYxZzaPH4TuoiusZvRnunFy2NbiHYuBp2vB66srX4vMEjpRKPxKXmnoQ4+Mn4DPiv8CYcrs3GfNUXJLtM+alSO
    hrMj/KT+wBNW3+E/2liywNO3iSflbaFva/+stGDTxE0E9Sjaox8HBhxpEamzMGSEaFKg+mjEddzDh1MxTDq3YV1kGBsjfwW3S9Cq
    anjmko+ndlb1UR3s6K8JlfphNWq9Ew/7c61T2BB/EbcaNkb8GBaE0tANH7/M34PLdhJDzjIcL9xPbdTG6zyM72Y+wXPHmvB489No
    fm0b5HnbQ9Rgp/7DSSd29AeVvPeNyK6JcYl/yQVi5dBjuGvoV/gaJe47s45QUxrDmcYX0MBsdF7egvXZ7+O0vZA4X8QmOQWjlSK7
    RDz5wIM30gp9UbWcGjXxhzdDu1SiNSpx6kcQB57rPnr/3dlkZarWLnlRq5oPET1dOCIOk4wALib9eeS5iygfhkd09H0DWphB/+gs
    +PcOAS+ssrFmmXXgVfR0de9cpbAJfH3Q1jofW9DZk56dDcVsq9YcsoUMEd1qyLoT3BX1YiyHMJuk97hyjqIoE91t+NcTLeN0ZrfM
    oXatZbu6G0h4VG+ibqq0IJVK6cAjo6serG3vSUezCMct0yQeSOFJSUImqb2qbknUpDqlZxE0QZ+ZUpSlZx79h4Nda6zef9dlk121
    JDjbR5XggPRZlRnS6bRQRtLpn4++cuie/Yvn2svmNxuLw9WCcYIl4fEoTEGiSTUqJdfgU+8ROqf1iMkLzS389YtNPXc/PH8l8ONB
    JZkHD+4JtD04HmVEDWWErmBhzV2/2LB1bemJG6krzv2S6NOHUgtEP0Oif5pE/3fHoruP7N8RiP61GArzSwbUhJJQpXJKiKbfr/3b
    IhKq76sKPUdF9NW/LSqfSn6vjv8C45H/6FSgvZQAAAAASUVORK5CYII='
         style='height:25px; border-radius:12px; display:inline-block; float:left; vertical-align:middle'>
    </img>
    {% endif %}
    </td>
    <td>
    {% if message %}
    &nbsp;&nbsp;<span style='font-size:12px'>{{ message }}</span>
    {% endif %}
    </td>
    </tr>
    </table>
    </div>
    """

    clear_output()

    logo = kwargs.get("logo", True)
    message = kwargs.get("message", "info ")

    template = Template(_template)
    html = template.render(
        {"logo": logo, "message": message.strip().replace("\n", "<br/>")}
    )
    publish_display_data(data={"text/html": html})


# --------------------------------------------------------------------------------------
# Version
# --------------------------------------------------------------------------------------
try:
    release = importlib.metadata.version("spectrochempy").split("+")[0]
    "Release version string of this package"
except Exception:  # pragma: no cover
    # package is not installed
    release = "--not set--"

try:
    version = get_version(root="../..", relative_to=__file__)
    "Version string of this package"
except LookupError:  # pragma: no cover
    version = release


def _get_copyright():
    current_year = np.datetime64("now", "Y")
    right = f"2014-{current_year}"
    right += " - A.Travert & C.Fernandez @ LCS"
    return right


copyright = _get_copyright()
"Copyright string of this package"


def _get_release_date():
    return subprocess.getoutput("git log -1 --tags --date=short --format='%ad'")


release_date = _get_release_date()
"Last release date of this package"


# --------------------------------------------------------------------------------------
# Testdata
# --------------------------------------------------------------------------------------
def _download_full_testdata_directory(datadir):
    # this process is relatively long, so we do not want to do it several time:
    downloaded = datadir / "__downloaded__"
    if downloaded.exists():
        return

    url = "https://github.com/spectrochempy/spectrochempy_data/archive/refs/heads/master.zip"

    resp = requests.get(url, stream=True, allow_redirects=True)
    zipfile = ZipFile(io.BytesIO(resp.content))
    files = [zipfile.open(file_name) for file_name in zipfile.namelist()]

    for file in files:
        name = file.name
        if name.endswith("/") or "testdata/" not in name:  # dir
            continue
        uncompressed = zipfile.read(name)
        p = list(pathclean(name).parts)[2:]
        dst = datadir.joinpath("/".join(p))
        if not dst.parent.exists():
            # create the eventually missing subdirectory
            dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(uncompressed)

    # write the "__downloaded__" file to avoid this process to run several file.
    downloaded.touch(exist_ok=True)


# --------------------------------------------------------------------------------------
# Other info
# --------------------------------------------------------------------------------------
url = "https://www.spectrochempy.fr"
"URL for the documentation of this package"

authors = "C. Fernandez & A. Travert"
"First authors(s) of this package"

contributors = "A. Ait Blal, W. Guérin, M. Mailänder"
"contributor(s) to this package"

license = "CeCILL-B license"
"License of this package"

cite = (
    f"Arnaud Travert & Christian Fernandez (2021) SpectroChemPy (version"
    f" {'.'.join(version.split('.')[0:2])}). "
    f"Zenodo. https://doi.org/10.5281/zenodo.3823841"
)
"How to cite this package"


# Directories


def _get_config_dir():
    """
    Determines the SpectroChemPy configuration directory name and
    creates the directory if it doesn't exist.

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

    # first look for SCP_CONFIG_HOME
    scp = environ.get("SCP_CONFIG_HOME")

    if scp is not None and Path(scp).exists():
        return Path(scp)

    config = find_or_create_spectrochempy_dir() / "config"
    if not config.exists():
        config.mkdir(exist_ok=True)

    return config


# ======================================================================================
# Application
# ======================================================================================
class SpectroChemPy(Application):
    """
    This class SpectroChemPy is the main class, containing most of the setup,
    configuration and more.
    """

    icon = tr.Unicode("scpy.png")
    "Icon for the application"

    running = tr.Bool(False)
    "Running status of the  `SpectroChemPy` application"

    name = tr.Unicode("SpectroChemPy")
    "Running name of the application"

    description = tr.Unicode(
        "SpectroChemPy is a framework for processing, analysing and modelling "
        "Spectroscopic data for Chemistry with Python."
    )
    "Short description of the  `SpectroChemPy` application"

    long_description = tr.Unicode()
    "Long description of the  `SpectroChemPy` application"

    @tr.default("long_description")
    def _get_long_description(self):
        desc = f"""
<p><strong>SpectroChemPy</strong> is a framework for processing, analysing and modelling
 <strong>Spectro</>scopic data for <strong>Chem</strong>istry with
 <strong>Py</strong>thon.
 It is a cross platform software, running on Linux, Windows or OS X.</p><br><br>
<strong>Version:</strong> {release}<br>
<strong>Authors:</strong> {authors}<br>
<strong>License:</strong> {license}<br>
<div class='warning'> SpectroChemPy is still experimental and under active development.
Its current design and
 functionalities are subject to major changes, reorganizations, bugs and crashes!!!.
 Please report any issues
to the <a url='https://github.com/spectrochempy/spectrochempy/issues'>Issue Tracker<a>
</div><br><br>
When using <strong>SpectroChemPy</strong> for your own work,
you are kindly requested to cite it this way: <pre>{cite}</pre></p>.
"""

        return desc

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    # Config file setting
    _loaded_config_files = tr.List()

    reset_config = tr.Bool(
        False, help="Should we restore a default configuration ?"
    ).tag(config=True)
    """Flag: True if one wants to reset settings to the original config defaults."""

    # config_file_name = tr.Unicode(None, help="Configuration file name").tag(config=True)
    # """Configuration file name."""
    #
    # @tr.default("config_file_name")
    # def _get_config_file_name_default(self):
    #     return str(self.name).lower() + "_cfg"

    config_dir = tr.Instance(Path, help="Set the configuration directory location").tag(
        config=True
    )
    """Configuration directory."""

    @tr.default("config_dir")
    def _get_config_dir_default(self):
        return _get_config_dir()

    config_manager = tr.Instance(BaseJSONConfigManager)

    @tr.default("config_manager")
    def _get_default_config_manager(self):
        return BaseJSONConfigManager(config_dir=str(self.config_dir))

    debug = tr.Bool(True, help="Set DEBUG mode, with full outputs").tag(config=True)
    """Flag to set debugging mode."""

    info = tr.Bool(False, help="Set INFO mode, with msg outputs").tag(config=True)
    """Flag to set info mode."""

    quiet = tr.Bool(False, help="Set Quiet mode, with minimal outputs").tag(config=True)
    """Flag to set in fully quite mode (even no warnings)."""

    nodisplay = tr.Bool(
        False, help="Set NO DISPLAY mode, i.e., no graphics outputs"
    ).tag(config=True)
    """Flag to set in NO DISPLAY mode."""

    # last_project = tr.Unicode('', help='Last used project').tag(config=True, type='project')
    # """Last used project"""
    #
    # @tr.observe('last_project')
    # def _last_project_changed(self, change):
    #     if change.name in self.traits(config=True):
    #         self.config_manager.update(self.config_file_name, {self.__class__.__name__: {change.name: change.new, }})

    show_config = tr.Bool(help="Dump configuration to stdout at startup").tag(
        config=True
    )

    @tr.observe("show_config")
    def _show_config_changed(self, change):
        if change.new:
            self._save_start = self.start
            self.start = self.start_show_config

    show_config_json = tr.Bool(help="Dump configuration to stdout (as JSON)").tag(
        config=True
    )

    @tr.observe("show_config_json")
    def _show_config_json_changed(self, change):
        self.show_config = change.new

    test = tr.Bool(False, help="test flag").tag(config=True)
    """Flag to set the application in testing mode."""

    port = tr.Integer(7000, help="Dash server port").tag(config=True)
    """Dash server port."""

    # logger
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

    # Command line interface
    # ----------------------------------------------------------------------------------
    aliases = dict(
        test="SpectroChemPy.test",
        project="SpectroChemPy.last_project",
        f="SpectroChemPy.startup_filename",
        port="SpectroChemPy.port",
    )

    flags = dict(
        debug=(
            {"SpectroChemPy": {"log_level": DEBUG}},
            "Set log_level to DEBUG - most verbose mode.",
        ),
        info=(
            {"SpectroChemPy": {"log_level": INFO}},
            "Set log_level to INFO - verbose mode.",
        ),
        quiet=(
            {"SpectroChemPy": {"log_level": ERROR}},
            "Set log_level to ERROR - no verbosity at all.",
        ),
        nodisplay=(
            {"SpectroChemPy": {"nodisplay": True}},
            "Set NO DISPLAY mode to true - no graphics at all",
        ),
        reset_config=(
            {"SpectroChemPy": {"reset_config": True}},
            "Reset config to default",
        ),
        show_config=(
            {
                "SpectroChemPy": {
                    "show_config": True,
                }
            },
            "Show the application's configuration (human-readable format).",
        ),
        show_config_json=(
            {
                "SpectroChemPy": {
                    "show_config_json": True,
                }
            },
            "Show the application's configuration (json format).",
        ),
    )

    classes = tr.List(
        help="List of configurables",
    )

    _from_warning_ = tr.Bool(False)

    # ----------------------------------------------------------------------------------
    # Initialisation of the application
    # ----------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.initialize()

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
        self, message, category, filename, lineno, file=None, line=None
    ):
        with self._fmtcontext():
            self.log.warning(f"({category.__name__}) {message}")

    # ----------------------------------------------------------------------------------
    # Initialisation of the configurables
    # ----------------------------------------------------------------------------------
    def _init_all_preferences(self):
        # Get preferences from the config files
        # ---------------------------------------------------------------------
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

        self.datadir = (
            DataDir()
        )  # config=self.config)  -- passing args deprecated in traitlets 4.2
        self.preferences = GeneralPreferences(parent=self)
        from spectrochempy.application.plot_preferences import (
            PlotPreferences,  # slow : delayed import
        )

        self.plot_preferences = PlotPreferences(parent=self)
        self.classes.extend(
            [
                GeneralPreferences,
                PlotPreferences,
            ]
        )

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    def start_show_config(self):
        """
        Start function used when show_config is True.
        """
        config = self.config.copy()
        # exclude show_config flags from displayed config
        for cls in self.__class__.mro():
            if cls.__name__ in config:
                cls_config = config[cls.__name__]
                cls_config.pop("show_config", None)
                cls_config.pop("show_config_json", None)

        if self.show_config_json:
            json.dump(config, sys.stdout, indent=1, sort_keys=True, default=repr)
            # add trailing newlines
            sys.stdout.write("\n")
            return self._start()

        if self._loaded_config_files:
            print("Loaded config files:")
            for fil in self._loaded_config_files:
                print(f"  {fil}")

        for classname in sorted(config):
            class_config = config[classname]
            if not class_config:
                continue
            print(classname)
            pformat_kwargs = dict(indent=4)
            if sys.version_info >= (3, 4):
                # use compact pretty-print on Pythons that support it
                pformat_kwargs["compact"] = True
            for traitname in sorted(class_config):
                value = class_config[traitname]
                print(f"  {traitname} = {pprint.pformat(value, **pformat_kwargs)}")

        # now run the actual start function
        return self.start()

    def initialize(self, argv=None):
        """
        Initialisation function for the API applications.

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

        ipy = get_ipython() if InteractiveShell.initialized() else None

        if ipy is None:
            # remove argument not known by spectrochempy
            if (
                "sphinx-build" in sys.argv[0]
                or "make.py" in sys.argv[0]
                or "pytest" in sys.argv[0]
                or "validate_docstrings" in sys.argv[0]
            ):  # building docs
                options = []
                for item in sys.argv[:]:
                    for k in list(self.flags.keys()):
                        if item.startswith("--" + k) or k in ["--help", "--help-all"]:
                            options.append(item)
                        continue
                    for k in list(self.aliases.keys()):
                        if item.startswith("-" + k) or k in [
                            "h",
                        ]:
                            options.append(item)
                self.parse_command_line(options)
            else:  # pragma: no cover
                pass  # print("args", sys.argv)
                # self.parse_command_line(sys.argv)

        # Get preferences from the config file and init everything
        # ---------------------------------------------------------------------
        self._init_all_preferences()

        # we catch warnings and error for a lighter display to the end-user.
        # except if we are in debugging mode

        # warning handler
        # ----------------
        def send_warnings_to_log(message, category):
            self.log.warning(f"{category.__name__} - {message}")

        warnings.showwarning = send_warnings_to_log

        # Warning handler
        # we catch warnings and error for a lighter display to the end-user.
        # except if we are in debugging mode

        warnings.showwarning = self._custom_warning

        # exception handler
        # -----------------
        if ipy is not None:  # pragma: no cover
            ipy.set_custom_exc((Exception,), self._ipython_catch_exceptions)
        else:
            # if environ.get("SCPY_TESTING", 0) == 0 and "pytest" not in sys.argv[0]:
            #    # catch exception only when pytest is not running
            sys.excepthook = self._catch_exceptions

    def reset_preferences(self):
        """
        Reset all preferences to default.
        """
        self.reset_config = True
        self._init_all_preferences()
        self.reset_config = False

    # ----------------------------------------------------------------------------------
    # start the application
    # ----------------------------------------------------------------------------------
    def start(self):
        """
        Start the  `SpectroChemPy` API.

        All configuration must have been done before calling this function.
        """
        if self.running:
            # API already started. Nothing done!
            return True

        if self.preferences.show_info_on_loading:
            info_string = f"SpectroChemPy's API - v.{version}\n© Copyright {copyright}"
            ipy = get_ipython()
            if ipy is not None and "TerminalInteractiveShell" not in str(ipy):
                _display_info_string(message=info_string.strip())

            else:
                if "/bin/scpy" not in sys.argv[0]:  # deactivate for console scripts
                    print(info_string.strip())

        # force update of rcParams
        for rckey in mpl.rcParams.keys():
            key = rckey.replace("_", "__").replace(".", "_").replace("-", "___")
            try:
                mpl.rcParams[rckey] = getattr(self.plot_preferences, key)
            except ValueError:
                mpl.rcParams[rckey] = getattr(self.plot_preferences, key).replace(
                    "'", ""
                )
            except AttributeError:
                # print(f'{e} -> you may want to add it to PlotPreferences.py')
                pass

        self.plot_preferences.set_latex_font(self.plot_preferences.font_family)

        # Eventually write the default config file
        # --------------------------------------
        self._make_default_config_file()

        # set the default style
        # --------------------------------------------------------------------------------------
        from matplotlib import pyplot as plt  # <-- slow!  delayed import

        plt.style.use(["classic"])

        debug_("API loaded - application is ready")
        self.running = True
        return True

    from contextlib import contextmanager

    @contextmanager
    def _fmtcontext(self):
        fmt = self.log_format
        try:
            yield fmt
        finally:
            self.log_format = fmt

    def info_(self, msg, *args, **kwargs):
        """
        Formatted info message.
        """
        with self._fmtcontext():
            self.log.info(msg, *args, **kwargs)

    def debug_(self, msg, *args, **kwargs):
        """
        Formatted debug message.
        """
        with self._fmtcontext():
            self.log.debug("DEBUG | " + msg, *args, **kwargs)

    def error_(self, *args, **kwargs):
        """
        Formatted error message.
        """
        if isinstance(args[0], Exception):
            e = args[0]
            etype = type(e)
            emessage = str(e)
        elif len(args) == 1 and isinstance(args[0], str):
            from spectrochempy.utils import exceptions

            etype = exceptions.SpectroChemPyError
            emessage = str(args[0])
        elif len(args) == 2:
            etype = args[0] if args else kwargs.get("type", None)
            emessage = (
                args[1] if args and len(args) > 1 else kwargs.get("message", None)
            )
        else:
            raise KeyError("wrong arguments have been passed to error_")
        self._catch_exceptions(etype, emessage, None)

    def warning_(self, msg, *args, **kwargs):
        """
        Formatted warning message.
        """
        self._from_warning_ = True
        warnings.warn(msg, *args, **kwargs)
        self._from_warning_ = False

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _make_default_config_file(self):
        # auto generate default config file.

        # remove old configuration file spectrochempy_cfg.py
        # --------------------------------------------------
        fname = self.config_dir / "spectrochempy_cfg.py"  # Old configuration file
        fname2 = self.config_dir / "SpectroChemPy.cfg.py"
        if fname.exists() or fname2.exists():
            for file in list(self.config_dir.iterdir()):
                file.unlink()

        # create a configuration file for each configurables
        # --------------------------------------------------
        from spectrochempy.analysis.api import __configurables__

        # first we will complete self.classes with a list of configurable in analysis
        self.classes.extend(__configurables__)
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


# ======================================================================================
# Start instance of Spectrochempy and expose public members in all
# ======================================================================================
app = SpectroChemPy()
preferences = app.preferences
error_ = app.error_
warning_ = app.warning_
info_ = app.info_
debug_ = app.debug_
preferences = app.preferences
plot_preferences = app.plot_preferences
description = app.description
long_description = app.long_description
config_manager = app.config_manager
config_dir = app.config_dir
reset_preferences = app.reset_preferences


# --------------------------------------------------------------------------------------
# Check for new release in a separate thread
# --------------------------------------------------------------------------------------
from spectrochempy.application._check_update import check_update

DISPLAY_UPDATE = threading.Thread(target=check_update, args=(version,))
DISPLAY_UPDATE.start()


# --------------------------------------------------------------------------------------
# Download data in a separate thread
# --------------------------------------------------------------------------------------
DOWNLOAD_TESTDATA = threading.Thread(
    target=_download_full_testdata_directory, args=(preferences.datadir,)
)
DOWNLOAD_TESTDATA.start()
