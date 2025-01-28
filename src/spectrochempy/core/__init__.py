# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
SpectroChemPy Core Package
=========================

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

import logging

from spectrochempy.utils.timeutils import timeit

# Initialize public API list
__all__: list[str] = []

# ------------------------------------------------------------------------------
# Application Components
# ------------------------------------------------------------------------------
with timeit("application"):
    from spectrochempy.application import CRITICAL  # noqa: E402
    from spectrochempy.application import DEBUG  # noqa: E402
    from spectrochempy.application import ERROR  # noqa: E402
    from spectrochempy.application import INFO  # noqa: E402
    from spectrochempy.application import WARNING  # noqa: E402
    from spectrochempy.application import authors  # noqa: E402
    from spectrochempy.application import config_dir  # noqa: E402
    from spectrochempy.application import config_manager  # noqa: E402
    from spectrochempy.application import contributors  # noqa: E402
    from spectrochempy.application import copyright  # noqa: E402
    from spectrochempy.application import debug_  # noqa: E402
    from spectrochempy.application import description  # noqa: E402
    from spectrochempy.application import error_  # noqa: E402
    from spectrochempy.application import info_  # noqa: E402
    from spectrochempy.application import license  # noqa: E402
    from spectrochempy.application import long_description  # noqa: E402
    from spectrochempy.application import plot_preferences  # noqa: E402
    from spectrochempy.application import preferences  # noqa: E402
    from spectrochempy.application import release  # noqa: E402
    from spectrochempy.application import release_date  # noqa: E402
    from spectrochempy.application import reset_preferences  # noqa: E402
    from spectrochempy.application import url  # noqa: E402
    from spectrochempy.application import version  # noqa: E402
    from spectrochempy.application import warning_  # noqa: E402

    def set_loglevel(level: str | int = logging.WARNING) -> None:
        """Set the logging level for SpectroChemPy.

        Parameters
        ----------
        level : Union[str, int]
            Logging level (e.g. 'WARNING', 'DEBUG', etc. or logging constants)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        preferences.log_level = level

    def get_loglevel() -> int:
        """Get current logging level.

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
    from spectrochempy.utils.constants import EPSILON  # noqa: E402
    from spectrochempy.utils.constants import INPLACE  # noqa: E402
    from spectrochempy.utils.constants import MASKED  # noqa: E402
    from spectrochempy.utils.constants import NOMASK  # noqa: E402
    from spectrochempy.utils.plots import show
    from spectrochempy.utils.print_versions import show_versions  # noqa: E402

    __all__ += ["show", "MASKED", "NOMASK", "EPSILON", "INPLACE", "show_versions"]

# Units
with timeit("units"):
    from spectrochempy.core.units import *  # noqa: E402,F403,F401

    __all__ += [
        "Unit",  # noqa: F405
        "Quantity",  # noqa: F405
        "ur",  # noqa: F405
        "set_nmr_context",  # noqa: F405
        "DimensionalityError",  # noqa: F405
    ]

# Dataset
with timeit("dataset"):
    from spectrochempy.core.dataset import api  # noqa: E402
    from spectrochempy.core.dataset.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# ------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------

# Plotting
with timeit("plotter"):
    from spectrochempy.core.plotters import api  # noqa: E402
    from spectrochempy.core.plotters.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# I/O Operations
with timeit("readers"):
    from spectrochempy.core.readers import api  # noqa: E402
    from spectrochempy.core.readers.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

with timeit("writers"):
    from spectrochempy.core.writers import api  # noqa: E402
    from spectrochempy.core.writers.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# Project Management
with timeit("project"):
    from spectrochempy.core.project.project import Project  # noqa: E402,F403,F401

    __all__ += ["Project"]

# Scripting
with timeit("script"):
    from spectrochempy.core.script import *  # noqa: E402,F403,F401

    __all__ += ["Script", "run_script", "run_all_scripts"]  # noqa: F405

# Interactive Components
with timeit("widgets"):
    from spectrochempy.widgets import api  # noqa: E402
    from spectrochempy.widgets.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# Analysis Tools
with timeit("analysis"):
    from spectrochempy.analysis import api  # noqa: E402
    from spectrochempy.analysis.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# Data Processing
with timeit("processing"):
    from spectrochempy.processing import api  # noqa: E402
    from spectrochempy.processing.api import *  # noqa: E402,F403,F401

    __all__ += api.__all__

# ------------------------------------------------------------------------------
# Application Startup
# ------------------------------------------------------------------------------
with timeit("start app"):
    from spectrochempy.application import app

    app.start()
