# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
SpectroChemPy Core Package.

This package defines the core functionality of the SpectroChemPy API, including:

- Base classes for spectroscopic data handling
- Plotting capabilities
- Data processing and analysis tools
- File I/O operations
- Project management
- Scripting utilities
- Interactive widgets
- Analysis methods

The module uses performance timing during import to track initialization of different components.

Notes
-----
The imports are organized in logical groups and timed using the `timeit` context manager.
All public API elements are collected in the __all__ list.

"""
# ruff: noqa: F405

import logging

from spectrochempy.utils.timeutils import timeit

# Initialize public API list
__all__: list[str] = []

# ------------------------------------------------------------------------------
# Application Components
# ------------------------------------------------------------------------------
with timeit("application"):
    from spectrochempy.application import CRITICAL
    from spectrochempy.application import DEBUG
    from spectrochempy.application import ERROR
    from spectrochempy.application import INFO
    from spectrochempy.application import WARNING
    from spectrochempy.application import authors
    from spectrochempy.application import config_dir
    from spectrochempy.application import config_manager
    from spectrochempy.application import contributors
    from spectrochempy.application import copyright
    from spectrochempy.application import debug_
    from spectrochempy.application import description
    from spectrochempy.application import error_
    from spectrochempy.application import info_
    from spectrochempy.application import license
    from spectrochempy.application import long_description
    from spectrochempy.application import plot_preferences
    from spectrochempy.application import preferences
    from spectrochempy.application import release
    from spectrochempy.application import release_date
    from spectrochempy.application import reset_preferences
    from spectrochempy.application import url
    from spectrochempy.application import version
    from spectrochempy.application import warning_

    def set_loglevel(level: str | int = logging.WARNING) -> None:
        """
        Set the logging level for SpectroChemPy.

        Parameters
        ----------
        level : Union[str, int]
            Logging level (e.g. 'WARNING', 'DEBUG', etc. or logging constants)

        """
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
        return preferences.log_level

    __all__ += [
        # Helpers
        "DEBUG",
        "WARNING",
        "ERROR",
        "CRITICAL",
        "INFO",
        "error_",
        "warning_",
        "debug_",
        "info_",
        "preferences",
        "plot_preferences",
        "config_manager",
        "config_dir",
        "reset_preferences",
        "set_loglevel",
        "get_loglevel",
        # Info
        "copyright",
        "version",
        "release",
        "license",
        "url",
        "release_date",
        "authors",
        "contributors",
        "description",
        "long_description",
    ]

# ------------------------------------------------------------------------------
# Core Components
# ------------------------------------------------------------------------------

# Constants
with timeit("constants"):
    from spectrochempy.utils.constants import EPSILON
    from spectrochempy.utils.constants import INPLACE
    from spectrochempy.utils.constants import MASKED
    from spectrochempy.utils.constants import NOMASK
    from spectrochempy.utils.plots import show
    from spectrochempy.utils.print_versions import show_versions

    __all__ += ["show", "MASKED", "NOMASK", "EPSILON", "INPLACE", "show_versions"]

# Units
with timeit("units"):
    from spectrochempy.core.units import *  # noqa: F403

    __all__ += [
        "Unit",
        "Quantity",
        "ur",
        "set_nmr_context",
        "DimensionalityError",
    ]  # noqa: F405

# Dataset
with timeit("dataset"):
    from spectrochempy.core.dataset import api
    from spectrochempy.core.dataset.api import *  # noqa: F403

    __all__ += api.__all__

# ------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------

# Plotting
with timeit("plotter"):
    from spectrochempy.core.plotters import api
    from spectrochempy.core.plotters.api import *  # noqa: F403

    __all__ += api.__all__

# I/O Operations
with timeit("readers"):
    from spectrochempy.core.readers import api
    from spectrochempy.core.readers.api import *  # noqa: F403

    __all__ += api.__all__

with timeit("writers"):
    from spectrochempy.core.writers import api
    from spectrochempy.core.writers.api import *  # noqa: F403

    __all__ += api.__all__

# Project Management
with timeit("project"):
    from spectrochempy.core.project.project import Project

    __all__ += ["Project"]

# Scripting
with timeit("script"):
    from spectrochempy.core.script import *  # noqa: F403

    __all__ += ["Script", "run_script", "run_all_scripts"]  # noqa: F405

# Interactive Components
with timeit("widgets"):
    from spectrochempy.widgets import api
    from spectrochempy.widgets.api import *  # noqa: F403

    __all__ += api.__all__

# Analysis Tools
with timeit("analysis"):
    from spectrochempy.analysis import api
    from spectrochempy.analysis.api import *  # noqa: F403

    __all__ += api.__all__

# Data Processing
with timeit("processing"):
    from spectrochempy.processing import api
    from spectrochempy.processing.api import *  # noqa: F403

    __all__ += api.__all__

# ------------------------------------------------------------------------------
# Application Startup
# ------------------------------------------------------------------------------
with timeit("start app"):
    from spectrochempy.application import app

    app.start()
