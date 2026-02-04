# ======================================================================================
# Copyright (©) 2014–2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Matplotlib plotting setup and state restoration utilities.

This module is responsible for:
- Applying the SpectroChemPy Matplotlib plotting style
- Preserving the user's original matplotlib rcParams
- Restoring the rcParams on demand

IMPORTANT DESIGN NOTES
----------------------
Matplotlib uses *global mutable state* (mpl.rcParams).

Once modified, rcParams cannot be "reconstructed" unless a snapshot
was taken beforehand.

For this reason:
- We DO NOT snapshot rcParams at import time
- We snapshot rcParams *lazily*, exactly when SpectroChemPy is about to
  modify them for the first time
- Restoration is guaranteed ONLY to the state *just before the first SCP plot*

This is the strongest guarantee possible without breaking user expectations
or delaying matplotlib initialization.
"""
import contextlib
import copy

import matplotlib as mpl

# -----------------------------------------------------------------------------
# Internal storage for the user's original rcParams
#
# This snapshot is taken ONCE, lazily, just before SpectroChemPy modifies
# matplotlib global state.
# -----------------------------------------------------------------------------

_USER_RCPARAMS = None


def _snapshot_user_rcparams() -> None:
    """
    Snapshot the current matplotlib rcParams exactly once.

    This function must be called immediately BEFORE SpectroChemPy applies
    any matplotlib style or modifies rcParams.

    Why lazy?
    ---------
    - Matplotlib rcParams may already be modified at import time by:
        * the environment (Jupyter, pytest-mpl, IDEs)
        * user code
        * other libraries

    - Capturing rcParams at import time would freeze an *unknown* state

    - Capturing them just-in-time ensures we restore exactly what the user
      was seeing before the first SpectroChemPy plot.
    """
    global _USER_RCPARAMS

    if _USER_RCPARAMS is None:
        # Deepcopy is required: rcParams values contain mutable objects
        _USER_RCPARAMS = copy.deepcopy(mpl.rcParams)


def ensure_spectrochempy_plot_style() -> None:
    """
    Apply the SpectroChemPy Matplotlib plotting style.

    This is a HIGH-LEVEL entry point called by SpectroChemPy plotting code.

    Responsibilities:
    -----------------
    1. Ensure Matplotlib is fully initialized (backend-safe)
    2. Snapshot the user's rcParams BEFORE modification
    3. Install SpectroChemPy matplotlib assets (stylesheets, fonts)
    4. Apply the configured SpectroChemPy matplotlib style

    This function is:
    - Safe to call multiple times
    - Idempotent
    - Non-destructive to the user's environment
    """

    # ------------------------------------------------------------------
    # Ensure matplotlib is initialized safely (backend-safe, no pyplot side effects)
    # ------------------------------------------------------------------
    from spectrochempy.core.plotters._mpl_setup import ensure_mpl_setup

    ensure_mpl_setup()

    # ------------------------------------------------------------------
    # Snapshot user rcParams BEFORE touching anything
    # ------------------------------------------------------------------
    _snapshot_user_rcparams()

    # ------------------------------------------------------------------
    # High-level imports (safe AFTER ensure_mpl_setup)
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    from spectrochempy.application.preferences import preferences
    from spectrochempy.core.plotters._mpl_assets import ensure_mpl_assets_installed

    # ------------------------------------------------------------------
    # Install matplotlib assets (best effort)
    #
    # Asset installation must NEVER break plotting:
    # - failure is logged elsewhere
    # - plotting must continue even without custom assets
    # ------------------------------------------------------------------
    with contextlib.suppress(Exception):
        ensure_mpl_assets_installed()

    # ------------------------------------------------------------------
    # Apply SpectroChemPy style if defined
    # ------------------------------------------------------------------
    plot_prefs = preferences.get("plot", None)
    style = getattr(plot_prefs, "style", None) if plot_prefs else None

    if style:
        plt.style.use(style)


def restore_rcparams() -> None:
    """
    Restore matplotlib rcParams to the user's original state.

    Guarantees:
    -----------
    - Restores rcParams to the exact state they had
      *just before the first SpectroChemPy plot*
    - Safe to call multiple times
    - Safe to call even if no SCP plot occurred
    - Does NOT require matplotlib pyplot state

    Limitations (documented behavior):
    ----------------------------------
    - This does NOT restore the rcParams state prior to importing SpectroChemPy
    - This is intentional and unavoidable due to matplotlib global state
    """

    mpl.rcParams.clear()
    mpl.rcParams.update(_USER_RCPARAMS)


def get_import_time_rcparams():
    """Return a copy of the import-time matplotlib rcParams snapshot."""
    return copy.deepcopy(_USER_RCPARAMS)
