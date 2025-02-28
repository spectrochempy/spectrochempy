# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: I001
"""
SpectroChemPy Application Programming Interface.

This module initializes the SpectroChemPy environment and exposes the public API.
It handles:
1. Environment detection (IPython, PyCharm, Jupyter, etc.)
2. Display settings configuration
3. Matplotlib backend and style setup
4. Core API exposure

The module adapts its behavior based on the execution context:
- Interactive vs non-interactive environment
- Test environment
- Documentation building
- Display availability
"""

import sys
from pathlib import Path

import matplotlib as mpl
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

# Environment detection
# --------------------
IN_IPYTHON = False
KERNEL = None
IP = None
if InteractiveShell.initialized():  # pragma: no cover
    IN_IPYTHON = True
    IP = get_ipython()
    KERNEL = getattr(IP, "kernel", None)

# Display configuration
# -------------------
NO_DISPLAY = False
NO_DIALOG = False

# Check execution context
if Path(sys.argv[0]).name in ["make.py", "validate_docstrings.py"]:  # pragma: no cover
    # Documentation building mode
    NO_DISPLAY = True
    NO_DIALOG = True
    mpl.use("agg", force=True)

# Check command line flags
if "--nodisplay" in sys.argv:  # pragma: no cover
    NO_DISPLAY = True
    NO_DIALOG = True

# Test environment detection
if "pytest" in sys.argv[0] or "py.test" in sys.argv[0]:
    NO_DISPLAY = True
    NO_DIALOG = True

    # Enable display for individual test runs
    if (
        len(sys.argv) > 1
        and any(arg.split("::")[0].endswith(".py") for arg in sys.argv[1:])
        and "--nodisplay" not in sys.argv
    ):  # pragma: no cover
        NO_DISPLAY = False
        NO_DIALOG = False

# PyCharm scientific mode detection
IN_PYCHARM_SCIMODE = mpl.get_backend() == "module://backend_interagg"

# Configure matplotlib backend
if (
    not (IN_IPYTHON and KERNEL) and not IN_PYCHARM_SCIMODE and not NO_DISPLAY
):  # pragma: no cover
    backend = mpl.rcParams["backend"]
    mpl.use(backend, force=True)

ALL = ["NO_DISPLAY", "NO_DIALOG"]

# Initialize matplotlib styles and fonts
# -----------------------------------
from spectrochempy.data.setup import setup_mpl

setup_mpl()

# Load core API
# ------------
from spectrochempy import core
from spectrochempy.core import *  # noqa: F403

ALL += core.__all__

# Terminal color support
if not IN_IPYTHON:
    from colorama import init as initcolor

    initcolor()

# Jupyter environment configuration
# -------------------------------
RUNNING_IN_COLAB = "google.colab" in str(get_ipython())

if IN_IPYTHON and KERNEL and not NO_DISPLAY:  # pragma: no cover
    try:
        if (
            "ipykernel_launcher" in sys.argv[0]
            and "--InlineBackend.rc={'figure.dpi': 96}" in sys.argv
        ):
            # Jupyter environment
            IP.run_line_magic("matplotlib", "inline")
        elif RUNNING_IN_COLAB:  # pragma: no cover
            # Google Colab environment
            IP.run_line_magic("matplotlib", "inline")
    except Exception:
        IP.run_line_magic("matplotlib", "qt")

# Path utilities and data directory setup
# ------------------------------------
from spectrochempy.utils.file import pathclean

DATADIR = pathclean(preferences.datadir)  # noqa: F405

__all__ = ["pathclean", "DATADIR"] + ALL

# Test environment setup
if NO_DISPLAY:
    mpl.use("agg", force=True)

    from os import environ

    # Set test files and folders
    environ["TEST_FILE"] = str(DATADIR / "irdata" / "nh4y-activation.spg")
    environ["TEST_FOLDER"] = str(DATADIR / "irdata" / "subdir")
    environ["TEST_NMR_FOLDER"] = str(
        DATADIR / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d",
    )

# For new version, where dialog are removed,
# NO_DIALOG = True
