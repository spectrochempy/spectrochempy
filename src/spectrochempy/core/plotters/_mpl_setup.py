# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# spectrochempy/core/plotters/_mpl_setup.py

"""
Minimal, side-effect-free Matplotlib bootstrap.

- DOES NOT import pyplot at module import time
- DOES NOT touch rcParams
- DOES NOT create figures
- Safe in pytest / CI / notebooks / GUI
"""

import os
import sys

_MPL_INITIALIZED = False


def ensure_mpl_setup():
    """
    Ensure Matplotlib is safely initialized.

    This function is:
    - idempotent
    - silent (no stdout / no logging)
    - safe in headless / CI environments
    """
    global _MPL_INITIALIZED

    if _MPL_INITIALIZED:
        return

    try:
        import matplotlib
    except Exception:
        # Matplotlib not available or broken
        raise

    # Headless safety (Linux / CI)
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            raise

    # Sanity check: pyplot must be importable
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        # Clean partial imports to avoid poisoned state
        sys.modules.pop("matplotlib.pyplot", None)
        sys.modules.pop("matplotlib", None)
        raise

    _MPL_INITIALIZED = True
