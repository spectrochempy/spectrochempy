# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module defines the `application` on which the API rely.

It also defines
the default application preferences and IPython magic functions.
"""

__all__ = []

import re
import sys
import logging
import subprocess
import datetime
import warnings
import pprint
import json
from os import environ
from pathlib import Path
import threading

from pkg_resources import get_distribution, DistributionNotFound
import requests
from setuptools_scm import get_version

from traitlets.config.configurable import Config
from traitlets.config.application import Application
from traitlets import (
    Bool,
    Unicode,
    List,
    Integer,
    Enum,
    Union,
    HasTraits,
    Instance,
    default,
    observe,
)
from traitlets.config.manager import BaseJSONConfigManager
import matplotlib as mpl
from matplotlib import pyplot as plt
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import Magics, magics_class, line_cell_magic
from IPython.core.magics.code import extract_symbols
from IPython.core.error import UsageError
from IPython.utils.text import get_text_list
from IPython.display import publish_display_data, clear_output
from jinja2 import Template

from spectrochempy.utils import MetaConfigurable, pathclean, get_pkg_path, Version
from spectrochempy.plot_preferences import PlotPreferences

# set the default style
plt.style.use(["classic"])

# ------------------------------------------------------------------
# Log levels
# ------------------------------------------------------------------

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


# ------------------------------------------------------------------
# logo / copyright display
# ------------------------------------------------------------------


def display_info_string(**kwargs):  # pragma: no cover
    _template = """
    {{widgetcss}}
    <table><tr><td>
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
         style='height:25px; border-radius:12px; display:inline-block; float:left; vertical-align:middle'></img>
    {% endif %}
    </td><td>
    {% if message %}
    &nbsp;&nbsp;<span style='font-size:12px'>{{ message }}</span>
    {% endif %}
    </td></tr></table>
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


# ------------------------------------------------------------------
# Version
# ------------------------------------------------------------------
try:
    __release__ = get_distribution("spectrochempy").version.split("+")[0]
    "Release version string of this package"
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    __release__ = "--not set--"

try:
    __version__ = get_version(root="..", relative_to=__file__)
    "Version string of this package"
except LookupError:  # pragma: no cover
    __version__ = __release__


# ............................................................................
def _get_copyright():
    current_year = datetime.date.today().year
    right = f"2014-{current_year}"
    right += " - A.Travert & C.Fernandez @ LCS"
    return right


__copyright__ = _get_copyright()
"Copyright string of this package"


# .............................................................................
def _get_release_date():
    return subprocess.getoutput("git log -1 --tags --date=short --format='%ad'")


__release_date__ = _get_release_date()
"Last release date of this package"


def _get_conda_package_version():
    """
    Get last conda package version
    """
    # Get version
    conda_url = "https://anaconda.org/spectrocat/spectrochempy/files"
    try:
        response = requests.get(conda_url)
    except requests.exceptions.RequestException:  # pragma: no cover
        return

    regex = (
        r"\/\d{1,2}\.\d{1,2}\.\d{1,2}\/download\/noarch"
        r"\/spectrochempy-(\d{1,2}\.\d{1,2}\.\d{1,2})\-(dev\d{1,"
        r"2}|stable).tar.bz2"
    )
    matches = re.finditer(regex, response.text, re.MULTILINE)
    vavailables = []
    for match in matches:
        vers = match[1] if match[2] == "stable" else f"{match[1]}.{match[2]}"
        vavailables.append(vers)

    return sorted(map(Version, vavailables))


def _check_for_updates(*args):

    old = Version(__version__)
    conda_versions = _get_conda_package_version()

    new_release = None
    new_version = None

    for new in conda_versions:
        if new > old:  # pragma: no cover
            new_version = new.public
            if not new.is_devrelease:
                new_release = new_version

    fil = Path.home() / ".scpy_update"
    if new_release and environ.get("DOC_BUILDING") is not None:  # pragma: no cover
        fil.write_text(
            f"You are running SpectrocChemPy-{__version__} but version {new_release} is available."
            f"Please consider updating for bug fixes and new features! "
        )

    else:  # pragma: no cover
        if fil.exists():
            fil.unlink()


CHECK_UPDATE = threading.Thread(target=_check_for_updates, args=(1,))
CHECK_UPDATE.start()

# other info
# ............................................................................

__url__ = "https://www.spectrochempy.fr"
"URL for the documentation of this package"

__author__ = "C. Fernandez & A. Travert"
"First authors(s) of this package"

__contributor__ = "A. Ait Blal, W. Guérin, M. Mailänder"
"contributor(s) to this package"

__license__ = "CeCILL-B license"
"License of this package"

__cite__ = (
    f"Arnaud Travert & Christian Fernandez (2021) SpectroChemPy (version"
    f" {'.'.join(__version__.split('.')[0:2])}). "
    f"Zenodo. https://doi.org/10.5281/zenodo.3823841"
)
"How to cite this package"


# ..........................................................................
def _find_or_create_spectrochempy_dir():
    directory = Path.home() / ".spectrochempy"

    directory.mkdir(exist_ok=True)  # Create directory only if it does not exist

    if directory.is_file():  # pragma: no cover
        msg = "Intended SpectroChemPy directory `{0}` is actually a file."
        raise IOError(msg.format(directory))

    return directory


# ======================================================================================================================
# Magic ipython function
# ======================================================================================================================
@magics_class
class SpectroChemPyMagics(Magics):
    """
    This class implements the addscript ipython magic function.
    """

    @line_cell_magic
    def addscript(self, pars="", cell=None):
        """
        This works both as **%addscript** and as **%%addscript**.

        This magic command can either take a local filename, element in the
        namespace or history range (see %history),
        or the current cell content.


        Usage:

            %addscript  -p project  n1-n2 n3-n4 ... n5 .. n6 ...

            or

            %%addscript -p project
            ...code lines ...


        Options:

            -p <string>         Name of the project where the script will be stored.
                                If not provided, a project with a standard
                                name : `proj` is searched.
            -o <string>         script name.
            -s <symbols>        Specify function or classes to load from python
                                source.
            -a                  append to the current script instead of
                                overwriting it.
            -n                  Search symbol in the current namespace.


        Examples
        --------

        .. sourcecode:: ipython

           In[1]: %addscript myscript.py

           In[2]: %addscript 7-27

           In[3]: %addscript -s MyClass,myfunction myscript.py
           In[4]: %addscript MyClass

           In[5]: %addscript mymodule.myfunction
        """
        opts, args = self.parse_options(pars, "p:o:s:n:a")

        # append = 'a' in opts
        # mode = 'a' if append else 'w'
        search_ns = "n" in opts

        if not args and not cell and not search_ns:  # pragma: no cover
            raise UsageError(
                "Missing filename, input history range, "
                "or element in the user namespace.\n "
                "If no argument are given then the cell content "
                "should "
                "not be empty"
            )
        name = "script"
        if "o" in opts:
            name = opts["o"]

        proj = "proj"
        if "p" in opts:
            proj = opts["p"]
        if proj not in self.shell.user_ns:  # pragma: no cover
            raise ValueError(
                f"Cannot find any project with name `{proj}` in the namespace."
            )
        # get the proj object
        projobj = self.shell.user_ns[proj]

        contents = ""
        if search_ns:
            contents += (
                "\n" + self.shell.find_user_code(opts["n"], search_ns=search_ns) + "\n"
            )

        args = " ".join(args)
        if args.strip():
            contents += (
                "\n" + self.shell.find_user_code(args, search_ns=search_ns) + "\n"
            )

        if "s" in opts:  # pragma: no cover
            try:
                blocks, not_found = extract_symbols(contents, opts["s"])
            except SyntaxError:
                # non python code
                logging.error("Unable to parse the input as valid Python code")
                return None

            if len(not_found) == 1:
                warnings.warn(f"The symbol `{not_found[0]}` was not found")
            elif len(not_found) > 1:
                sym = get_text_list(not_found, wrap_item_with="`")
                warnings.warn(f"The symbols {sym} were not found")

            contents = "\n".join(blocks)

        if cell:
            contents += "\n" + cell

        # import delayed to avoid circular import error
        from spectrochempy.core.scripts.script import Script

        script = Script(name, content=contents)
        projobj[name] = script

        return f"Script {name} created."

        # @line_magic  # def runscript(self, pars=''):  #     """  #  #  # """  #     opts,
        # args = self.parse_options(pars, '')  #  #     if  # not args:  #         raise UsageError('Missing script
        # name')  #  #  # return args


# ======================================================================================================================
# DataDir class
# ======================================================================================================================


class DataDir(HasTraits):
    """
    A class used to determine the path to the testdata directory.
    """

    path = Instance(Path)

    @default("path")
    def _get_path_default(self, **kwargs):  # pragma: no cover

        super().__init__(**kwargs)

        # create a directory testdata in .spectrochempy to avoid an error if the following do not work
        path = _find_or_create_spectrochempy_dir() / "testdata"
        path.mkdir(exist_ok=True)

        # try to use the conda installed testdata (spectrochempy_data package)
        try:
            conda_env = environ["CONDA_PREFIX"]
            _path = Path(conda_env) / "share" / "spectrochempy_data" / "testdata"
            if not _path.exists():
                _path = (
                    Path(conda_env) / "share" / "spectrochempy_data"
                )  # depending on the version of spectrochempy_data
            if _path.exists():
                path = _path

        except KeyError:
            pass

        return path

    def listing(self):
        """
        Create a str representing a listing of the testdata folder.

        Returns
        -------
        listing : str
            Display of the datadir content
        """
        strg = f"{self.path.name}\n"  # os.path.basename(self.path) + "\n"

        def _listdir(strg, initial, nst):
            nst += 1
            for fil in pathclean(initial).glob(
                "*"
            ):  # glob.glob(os.path.join(initial, '*')):
                filename = fil.name  # os.path.basename(f)
                if filename.startswith("."):  # pragma: no cover
                    continue
                if (
                    not filename.startswith("acqu")
                    and not filename.startswith("pulse")
                    and filename not in ["ser", "fid"]
                ):
                    strg += "   " * nst + f"|__{filename}\n"
                if fil.is_dir():
                    strg = _listdir(strg, fil, nst)
            return strg

        return _listdir(strg, self.path, -1)

    @classmethod
    def class_print_help(cls):
        # to work with --help-all
        """"""  # TODO: make some useful help

    def __str__(self):
        return self.listing()

    def _repr_html_(self):  # pragma: no cover
        # _repr_html is needed to output in notebooks
        return self.listing().replace("\n", "<br/>").replace(" ", "&nbsp;")


# ======================================================================================================================
# General Preferences
# ======================================================================================================================


class GeneralPreferences(MetaConfigurable):
    """
    Preferences that apply to the |scpy| application in general.

    They should be accessible from the main API.
    """

    name = Unicode("GeneralPreferences")
    description = Unicode("General options for the SpectroChemPy application")
    updated = Bool(False)

    # ------------------------------------------------------------------------
    # Configuration entries
    # ------------------------------------------------------------------------

    # NON GUI
    show_info_on_loading = Bool(True, help="Display info on loading").tag(config=True)
    use_qt = Bool(
        False,
        help="Use QT for dialog instead of TK which is the default. "
        "If True the PyQt libraries must be installed",
    ).tag(config=True)

    # GUI
    databases_directory = Union(
        (Instance(Path), Unicode()),
        help="Directory where to look for database files such as csv",
    ).tag(config=True, gui=True, kind="folder")

    datadir = Union(
        (Instance(Path), Unicode()), help="Directory where to look for data by default"
    ).tag(config=True, gui=True, kind="folder")

    workspace = Union(
        (Instance(Path), Unicode()), help="Workspace directory by default"
    ).tag(config=True, gui=True, kind="folder")

    # ------------------------------------------------------------------------
    # Configuration entries
    # ------------------------------------------------------------------------

    autoload_project = Bool(
        True, help="Automatic loading of the last project at startup"
    ).tag(config=True, gui=True)

    autosave_project = Bool(True, help="Automatic saving of the current project").tag(
        config=True, gui=True
    )

    project_directory = Union(
        (Instance(Path), Unicode()),
        help="Directory where projects are stored by default",
    ).tag(config=True, kind="folder")

    last_project = Union(
        (Instance(Path, allow_none=True), Unicode()), help="Last used project"
    ).tag(config=True, gui=True, kind="file")

    show_close_dialog = Bool(
        True,
        help="Display the close project dialog project changing or on application exit",
    ).tag(config=True, gui=True)

    csv_delimiter = Enum(
        [",", ";", r"\t", " "], default_value=",", help="CSV data delimiter"
    ).tag(config=True, gui=True)

    @default("project_directory")
    def _get_default_project_directory(self):
        # Determines the SpectroChemPy project directory name and creates the directory if it doesn't exist.
        # This directory is typically ``$HOME/spectrochempy/projects``, but if the SCP_PROJECTS_HOME environment
        # variable is set and the `$SCP_PROJECTS_HOME` directory exists, it will be that directory.
        # If neither exists, the former will be created.

        # first look for SCP_PROJECTS_HOME
        pscp = environ.get("SCP_PROJECTS_HOME")
        if pscp is not None and Path(pscp).exists():
            return Path(pscp)

        pscp = Path.home() / ".spectrochempy" / "projects"

        pscp.mkdir(exist_ok=True)

        if pscp.is_file():
            raise IOError("Intended Projects directory is actually a file.")

        return pscp

    # ..........................................................................
    @default("workspace")
    def _get_workspace_default(self):
        # the spectra path in package data
        return Path.home()

    # ..........................................................................
    @default("databases_directory")
    def _get_databases_directory_default(self):
        # the spectra path in package data
        return Path(get_pkg_path("databases", "scp_data"))

    # ..........................................................................
    @default("datadir")
    def _get_default_datadir(self):
        return self.parent.datadir.path

    # ..........................................................................
    @observe("datadir")
    def _datadir_changed(self, change):
        self.parent.datadir.path = pathclean(change["new"])

    # ..........................................................................
    @property
    def log_level(self):
        """
        Logging level (int).
        """
        return self.parent.log_level

    # ..........................................................................
    @log_level.setter
    def log_level(self, value):
        if isinstance(value, str):
            value = getattr(logging, value, None)
            if value is None:  # pragma: no cover
                warnings.warn(
                    "Log level not changed: invalid value given\n"
                    "string values must be DEBUG, INFO, WARNING, "
                    "or ERROR"
                )
        self.parent.log_level = value

    # ..........................................................................
    def __init__(self, **kwargs):
        super().__init__(jsonfile="GeneralPreferences", **kwargs)


# ======================================================================================================================
# Application
# ======================================================================================================================


class SpectroChemPy(Application):
    """
    This class SpectroChemPy is the main class, containing most of the setup,
    configuration and more.
    """

    icon = Unicode("scpy.png")
    "Icon for the application"

    running = Bool(False)
    "Running status of the |scpy| application"

    name = Unicode("SpectroChemPy")
    "Running name of the application"

    description = Unicode(
        "SpectroChemPy is a framework for processing, analysing and modelling Spectroscopic data for "
        "Chemistry with Python."
    )
    "Short description of the |scpy| application"

    long_description = Unicode()
    "Long description of the |scpy| application"

    @default("long_description")
    def _get_long_description(self):
        desc = f"""
<p><strong>SpectroChemPy</strong> is a framework for processing, analysing and modelling
 <strong>Spectro</>scopic data for <strong>Chem</strong>istry with <strong>Py</strong>thon.
 It is a cross platform software, running on Linux, Windows or OS X.</p><br><br>
<strong>Version:</strong> {__release__}<br>
<strong>Authors:</strong> {__author__}<br>
<strong>License:</strong> {__license__}<br>
<div class='warning'> SpectroChemPy is still experimental and under active development. Its current design and
 functionalities are subject to major changes, reorganizations, bugs and crashes!!!. Please report any issues
to the <a url='https://github.com/spectrochempy/spectrochempy/issues'>Issue Tracker<a>
</div><br><br>
When using <strong>SpectroChemPy</strong> for your own work,
you are kindly requested to cite it this way: <pre>{__cite__}</pre></p>.
"""

        return desc

    # ------------------------------------------------------------------------
    # Configuration parameters
    # ------------------------------------------------------------------------

    # Config file setting
    # ------------------------------------------------------------------------
    _loaded_config_files = List()

    reset_config = Bool(False, help="Should we restore a default configuration ?").tag(
        config=True
    )
    """Flag: True if one wants to reset settings to the original config defaults."""

    config_file_name = Unicode(None, help="Configuration file name").tag(config=True)
    """Configuration file name."""

    @default("config_file_name")
    def _get_config_file_name_default(self):
        return str(self.name).lower() + "_cfg"

    config_dir = Instance(Path, help="Set the configuration directory location").tag(
        config=True
    )
    """Configuration directory."""

    @default("config_dir")
    def _get_config_dir_default(self):
        return self.get_config_dir()

    config_manager = Instance(BaseJSONConfigManager)

    @default("config_manager")
    def _get_default_config_manager(self):
        return BaseJSONConfigManager(config_dir=str(self.config_dir))

    log_format = Unicode(
        "%(highlevel)s %(message)s",
        help="The Logging format template",
    ).tag(config=True)

    debug = Bool(True, help="Set DEBUG mode, with full outputs").tag(config=True)
    """Flag to set debugging mode."""

    info = Bool(False, help="Set INFO mode, with msg outputs").tag(config=True)
    """Flag to set info mode."""

    quiet = Bool(False, help="Set Quiet mode, with minimal outputs").tag(config=True)
    """Flag to set in fully quite mode (even no warnings)."""

    nodisplay = Bool(False, help="Set NO DISPLAY mode, i.e., no graphics outputs").tag(
        config=True
    )
    """Flag to set in NO DISPLAY mode."""

    # last_project = Unicode('', help='Last used project').tag(config=True, type='project')
    # """Last used project"""
    #
    # @observe('last_project')
    # def _last_project_changed(self, change):
    #     if change.name in self.traits(config=True):
    #         self.config_manager.update(self.config_file_name, {self.__class__.__name__: {change.name: change.new, }})

    show_config = Bool(help="Dump configuration to stdout at startup").tag(config=True)

    @observe("show_config")
    def _show_config_changed(self, change):
        if change.new:
            self._save_start = self.start
            self.start = self.start_show_config

    show_config_json = Bool(help="Dump configuration to stdout (as JSON)").tag(
        config=True
    )

    @observe("show_config_json")
    def _show_config_json_changed(self, change):
        self.show_config = change.new

    test = Bool(False, help="test flag").tag(config=True)
    """Flag to set the application in testing mode."""

    port = Integer(7000, help="Dash server port").tag(config=True)
    """Dash server port."""

    # Command line interface
    # ------------------------------------------------------------------------

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

    classes = List(
        [
            GeneralPreferences,
            PlotPreferences,
            DataDir,
        ]
    )

    # ------------------------------------------------------------------------
    # Initialisation of the application
    # ------------------------------------------------------------------------

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.logs = (
            self.log
        )  # we change the no name in order to avoid latter conflict with numpy.log

        self.initialize()

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

        in_python = False
        if InteractiveShell.initialized():
            in_python = True

        if not in_python:
            # remove argument not known by spectrochempy
            if (
                "make.py" in sys.argv[0]
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
                self.parse_command_line(sys.argv)

        # Get preferences from the config file and init everything
        # ---------------------------------------------------------------------

        self._init_all_preferences()

        # we catch warnings and error for a lighter display to the end-user.
        # except if we are in debugging mode

        # warning handler
        # --------------------------------------------------------------------
        def send_warnings_to_log(message, category):
            self.logs.warning(f"{category.__name__} - {message}")

        warnings.showwarning = send_warnings_to_log

        # exception handler
        # --------------------------------------------------------------------

        if in_python:  # pragma: no cover

            ipy = get_ipython()

            def _custom_exc(shell, etype, evalue, tb, tb_offset=None):

                if self.log_level == logging.DEBUG:
                    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
                else:
                    self.logs.error(f"{etype.__name__}: {evalue}")

            ipy.set_custom_exc((Exception,), _custom_exc)

            # load our custom magic extensions
            # --------------------------------------------------------------------
            if ipy is not None:
                ipy.register_magics(SpectroChemPyMagics)

    def _init_all_preferences(self):

        # Get preferences from the config file
        # ---------------------------------------------------------------------

        if not self.config:
            self.config = Config()

        configfiles = []
        if self.config_file_name:
            config_file = self.config_dir / self.config_file_name
            configfiles.append(config_file)

            lis = self.config_dir.iterdir()
            for fil in lis:
                if fil.suffix == ".json":
                    jsonname = self.config_dir / fil
                    if self.reset_config or fil == "PlotPreferences.json":
                        # remove the user json file to reset to defaults
                        jsonname.unlink()
                    else:
                        configfiles.append(jsonname)

            for cfgname in configfiles:
                self.load_config_file(cfgname)
                if cfgname not in self._loaded_config_files:
                    self._loaded_config_files.append(cfgname)

        # Eventually write the default config file
        # --------------------------------------
        self._make_default_config_file()

        self.datadir = (
            DataDir()
        )  # config=self.config)  -- passing args deprecated in traitlets 4.2
        self.preferences = GeneralPreferences(config=self.config, parent=self)
        self.plot_preferences = PlotPreferences(config=self.config, parent=self)

    # ..........................................................................
    @staticmethod
    def get_config_dir():
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
        scp = environ.get("SCP_CONFIG_HOME")

        if scp is not None and Path(scp).exists():
            return Path(scp)

        config = _find_or_create_spectrochempy_dir() / "config"
        if not config.exists():
            config.mkdir(exist_ok=True)

        return config

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
            print()
            return self._start()

        if self._loaded_config_files:
            print("Loaded config files:")
            for fil in self._loaded_config_files:
                print(f"  {fil}")
            print()

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
        print()

        # now run the actual start function
        return self._start()

    def reset_preferences(self):
        """
        Reset all preferences to default.
        """
        self.reset_config = True
        self._init_all_preferences()
        self.reset_config = False

    # ------------------------------------------------------------------------
    # start the application
    # ------------------------------------------------------------------------

    def start(self):
        """
        Start the |scpy| API.

        All configuration must have been done before calling this function.
        """

        # print(f'{sys.argv}')

        return self._start()

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    def _start(self):

        if self.running:
            # API already started. Nothing done!
            return True

        if self.preferences.show_info_on_loading:
            info_string = (
                f"SpectroChemPy's API - v.{__version__}\n© Copyright {__copyright__}"
            )
            ipy = get_ipython()
            if ipy is not None and "TerminalInteractiveShell" not in str(ipy):
                display_info_string(message=info_string.strip())

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

        self.running = True

        # display needs for update
        # time.sleep(1)
        fil = Path.home() / ".scpy_update"
        if fil.exists():
            try:
                msg = fil.read_text()
                self.logs.warning(msg)
            except Exception:
                pass

        return True

    # ..........................................................................
    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = self.config_dir / self.config_file_name
        fname = fname.with_suffix(".py")

        if not fname.exists() or self.reset_config:
            sfil = self.generate_config_file()
            self.logs.info(f"Generating default config file: {fname}")
            with open(fname, "w") as fil:
                fil.write(sfil)

    # ------------------------------------------------------------------------
    # Events from Application
    # ------------------------------------------------------------------------

    @observe("log_level")
    def _log_level_changed(self, change):

        self.log_format = "%(message)s"
        if change.new == DEBUG:
            self.log_format = "[%(filename)s-%(funcName)s %(levelname)s] %(message)s"
        self.logs._cache = {}
        self.logs.level = self.log_level
        for handler in self.logs.handlers:
            handler.level = self.log_level
        self.logs.info(
            f"changed default log_level to {logging.getLevelName(change.new)}"
        )


# ======================================================================================================================

if __name__ == "__main__":
    pass
