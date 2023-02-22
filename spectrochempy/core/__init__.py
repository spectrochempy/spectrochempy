# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Package defining the *core* methods of the |scpy| API.

Most the API methods such as plotting, processing, analysis, etc...

isort:skip_file
"""

__all__ = []  # modified below

from os import environ
import sys
import warnings

warnings.filterwarnings("ignore")

# ======================================================================================
# Progress bar
# ======================================================================================================================
PBAR_COUNT = 0

USE_TQDM = (
    environ.get("USE_TQDM", "Yes") == "Yes"
    and "DOC_BUILDING" not in environ
    and "/bin/scpy" not in sys.argv[0]
)

if USE_TQDM:
    from tqdm import tqdm

    pbar = tqdm(total=100)
    pbar.set_description("Loading SpectroChemPy API")
    val_tqdm = [0.1, 3.9, 4.5, 5.3, 5.4, 7.5, 8.4, 8.4, 8.5, 8.6, 9.7, 9.8, 9.9, 10]


def _pbar_update(close=None):
    global PBAR_COUNT

    if USE_TQDM:

        if close:
            pbar.clear()
            pbar.close()
            PBAR_COUNT = 0
        else:
            pbar.update(val_tqdm[PBAR_COUNT])
            PBAR_COUNT += 1
    else:
        pass


# ======================================================================================================================
# loading module libraries
# here we also construct the __all__ list automatically
# ======================================================================================================================

_pbar_update()
from spectrochempy.application import app  # noqa: E402

__all__ += ["app"]

from spectrochempy.application import (
    __version__ as version,
    __release__ as release,
    __copyright__ as copyright,
    __license__ as license,
    __release_date__ as release_date,
    __author__ as authors,
    __contributor__ as contributors,
    __url__ as url,
    DEBUG,
    WARNING,
    ERROR,
    CRITICAL,
    INFO,
    error_,
    warning_,
    debug_,
    info_,
    preferences,
    plot_preferences,
    description,
    long_description,
    config_dir,
    config_manager,
    reset_preferences,
)  # noqa: E402


# datadir = app.datadir


def set_loglevel(level=WARNING):
    if isinstance(level, str):
        import logging

        level = getattr(logging, level)
    preferences.log_level = level


def get_loglevel():
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

# IPython methods
# ---------------
# we put them before so that we can eventually overwrite them

_pbar_update()

# constants
# ---------
from spectrochempy.utils import (
    show,
    MASKED,
    NOMASK,
    EPSILON,
    INPLACE,
    show_versions,
)

__all__ += ["show", "MASKED", "NOMASK", "EPSILON", "INPLACE", "show_versions"]

# dataset
# -------
_pbar_update()
from spectrochempy.core.dataset import api  # noqa: E402
from spectrochempy.core.dataset.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# plotters
# --------
_pbar_update()
from spectrochempy.core.plotters import api  # noqa: E402
from spectrochempy.core.plotters.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# processors
# ----------
_pbar_update()
from spectrochempy.core.processors import api  # noqa: E402
from spectrochempy.core.processors.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# readers
# -------
_pbar_update()
from spectrochempy.core.readers import api  # noqa: E402
from spectrochempy.core.readers.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# writers
# -------
_pbar_update()
from spectrochempy.core.writers import api  # noqa: E402
from spectrochempy.core.writers.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# units
# -----
_pbar_update()
from spectrochempy.core.units import api  # noqa: E402
from spectrochempy.core.units.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# databases
# ---------
_pbar_update()
from spectrochempy.databases import api  # noqa: E402
from spectrochempy.databases.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# analysis
# --------
_pbar_update()
from spectrochempy.analysis import api  # noqa: E402
from spectrochempy.analysis.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# project
# -------
_pbar_update()
from spectrochempy.core.project import api  # noqa: E402
from spectrochempy.core.project.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# script
# ------
_pbar_update()
from spectrochempy.core.scripts import api  # noqa: E402
from spectrochempy.core.scripts.api import *  # noqa: E402,F403,F401

__all__ += api.__all__

# widgets
# -------
_pbar_update()
from spectrochempy.widgets import api  # noqa: E402
from spectrochempy.widgets.api import *  # noqa: E402,F403,F401

__all__ += api.__all__


# Helpers
# -------
def APIref():
    """
    Helper to display public objects and methods from the API
    """
    a = __all__[:]
    a = sorted(a)
    return a


APIref = APIref()

__all__.append("APIref")

# START THE app

_pbar_update()

_pbar_update(close=True)

_started = app.start()

warnings.filterwarnings(action="ignore", module="matplotlib", category=UserWarning)

# ======================================================================================================================
if __name__ == "__main__":
    pass
