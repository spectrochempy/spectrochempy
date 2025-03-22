# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import importlib.util
import sys
from os import environ

import matplotlib as mpl
from traitlets import import_item

from spectrochempy.application.jupyter import setup_jupyter_css
from spectrochempy.data.setup import setup_mpl
from spectrochempy.utils.system import get_ipython
from spectrochempy.utils.system import is_notebook
from spectrochempy.utils.system import is_terminal


# --------------------------------------------------------------------------------------
def setup_environment():
    NO_DISPLAY = False

    is_docs = environ.get("DOC_BUILDING") is not None
    is_pytest = "pytest" in sys.argv[0] or "py.test" in sys.argv[0]

    # Terminal output colors
    # ----------------------
    if is_terminal():
        # needed in Windows terminal - but must not be inited in Jupyter notebook
        from colorama import init as initcolor

        initcolor()

    # Are we running pytest?
    # ----------------------
    if is_pytest or is_docs:
        # if we are testing or doc building we like a silent work with no figure popup!
        NO_DISPLAY = True

    # Matplotlib setup
    # -----------------
    if NO_DISPLAY:
        # Setup for pytest and sphinx
        mpl.use("template", force=True)

        # set test file and folder in environment
        environ["TEST_FILE"] = "irdata/nh4y-activation.spg"
        environ["TEST_FOLDER"] = "irdata/subdir"
        environ["TEST_NMR_FOLDER"] = "nmrdata/bruker/tests/nmr/topspin_2d"

    elif is_notebook():  # pragma: no cover
        IP = get_ipython()
        try:
            if (
                "ipykernel_launcher" in sys.argv[0]
                and "--InlineBackend.rc={'figure.dpi': 96}" in sys.argv
            ):
                # We are running from NBSphinx - the plot must be inline to show up.
                IP.run_line_magic("matplotlib", "inline")

            elif "google.colab" in str(IP):  # pragma: no cover
                # allow using matplotlib widget
                output = import_item("google.colab").output
                output.enable_custom_widget_manager()

            elif importlib.util.find_spec("ipympl") is not None:
                IP.run_line_magic("matplotlib", "widget")

            elif importlib.util.find_spec("PyQt6") is not None:
                IP.run_line_magic("matplotlib", "qt")

            else:
                IP.run_line_magic("matplotlib", "inline")
        except Exception:
            IP.run_line_magic("matplotlib", "inline")

        # Jupyter integration
        # -------------------
        setup_jupyter_css()

    # Initialize matplotlib styles and fonts
    # --------------------------------------
    setup_mpl()

    SCPY_STARTUP_LOGLEVEL = environ.get("SCPY_STARTUP_LOGLEVEL", None)

    if SCPY_STARTUP_LOGLEVEL is None:
        SCPY_STARTUP_LOGLEVEL = "DEBUG" if is_pytest else "INFO"

    return NO_DISPLAY, SCPY_STARTUP_LOGLEVEL, is_pytest
