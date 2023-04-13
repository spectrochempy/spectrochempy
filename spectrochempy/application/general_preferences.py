# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import logging
import warnings
from os import environ
from pathlib import Path

import traitlets as tr

from spectrochempy.application.metaconfigurable import MetaConfigurable
from spectrochempy.utils.file import pathclean


# ======================================================================================
# General Preferences
# ======================================================================================
class GeneralPreferences(MetaConfigurable):
    """
    Preferences that apply to the  `SpectroChemPy` application in general.

    They should be accessible from the main API.
    """

    # ----------------------------------------------------------------------------------
    # Configuration entries
    # ----------------------------------------------------------------------------------
    # NON GUI
    show_info_on_loading = tr.Bool(True, help="Display info on loading").tag(
        config=True
    )
    use_qt = tr.Bool(
        False,
        help="Use QT for dialog instead of TK which is the default. "
        "If True the PyQt libraries must be installed",
    ).tag(config=True)

    # GUI
    # databases_directory = tr.Union(
    #     (tr.Instance(Path), tr.Unicode()),
    #     help="Directory where to look for database files such as csv",
    # ).tag(config=True, gui=True, kind="folder")

    datadir = tr.Union(
        (tr.Instance(Path), tr.Unicode()),
        help="Directory where to look for data by default",
    ).tag(config=True, gui=True, kind="folder")

    workspace = tr.Union(
        (tr.Instance(Path), tr.Unicode()), help="Workspace directory by default"
    ).tag(config=True, gui=True, kind="folder")

    autoload_project = tr.Bool(
        True, help="Automatic loading of the last project at startup"
    ).tag(config=True, gui=True)

    autosave_project = tr.Bool(
        True, help="Automatic saving of the current project"
    ).tag(config=True, gui=True)

    project_directory = tr.Union(
        (tr.Instance(Path), tr.Unicode()),
        help="Directory where projects are stored by default",
    ).tag(config=True, kind="folder")

    last_project = tr.Union(
        (tr.Instance(Path, allow_none=True), tr.Unicode()), help="Last used project"
    ).tag(config=True, gui=True, kind="file")

    show_close_dialog = tr.Bool(
        True,
        help="Display the close project dialog project changing or on application exit",
    ).tag(config=True, gui=True)

    csv_delimiter = tr.Enum(
        [",", ";", r"\t", " "], default_value=",", help="CSV data delimiter"
    ).tag(config=True, gui=True)

    check_update_frequency = tr.Enum(
        ["day", "week", "month"],
        default_value="day",
        help="Frequency of checking for update",
    )

    @tr.default("project_directory")
    def _get_default_project_directory(self):
        # Determines the SpectroChemPy project directory name and creates the directory
        # if it doesn't exist.
        # This directory is typically `$HOME/spectrochempy/projects` , but if the
        # SCP_PROJECTS_HOME environment
        # variable is set and the `$SCP_PROJECTS_HOME` directory exists, it will be
        # that directory.
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

    @tr.default("workspace")
    def _get_workspace_default(self):
        # the spectra path in package data
        return Path.home()

    # @tr.default("databases_directory")
    # def _get_databases_directory_default(self):
    #     # the spectra path in package data
    #     return pathclean(get_pkg_path("databases", "scp_data"))

    @tr.default("datadir")
    def _get_default_datadir(self):
        return pathclean(self.parent.datadir.path)

    @tr.observe("datadir")
    def _datadir_changed(self, change):
        self.parent.datadir.path = pathclean(change["new"])

    @tr.validate("datadir")
    def _data_validate(self, proposal):
        # validation of the datadir attribute
        datadir = proposal["value"]
        if isinstance(datadir, str):
            datadir = pathclean(datadir)
        return datadir

    @property
    def log_level(self):
        """
        Logging level (int).
        """
        return self.parent.log_level

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

    # ----------------------------------------------------------------------------------
    # Class Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
