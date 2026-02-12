# ======================================================================================
# Copyright (©) 2014–2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Matplotlib plotting setup and state restoration utilities.

This module is responsible for:
- LAZY initialization of all matplotlib functionality
- Applying the SpectroChemPy Matplotlib plotting style only when needed
- Preserving the user's original matplotlib rcParams
- Restoring the rcParams on demand

LAZY INITIALIZATION DESIGN NOTES
--------------------------------
Alternative A Implementation: ALL matplotlib setup is deferred until first plot.

This provides maximum import performance by:
- Zero matplotlib imports at module import time
- Zero rcParams modifications until first plot() call
- Zero asset installation until first plot() call
- Full preservation of user's matplotlib environment during import

The only trigger for matplotlib initialization is first call to
lazy_ensure_mpl_config() which happens on the first dataset.plot().

This achieves the performance goal: 2.5+ second import time improvement.
"""

import contextlib
import copy
import threading
from enum import Enum
from typing import Any, Dict, Optional

# NO matplotlib imports at module level!
# These will be imported lazily when needed.

# -----------------------------------------------------------------------------
# Lazy initialization state management
# -----------------------------------------------------------------------------


class MPLInitState(Enum):
    """Matplotlib initialization state enumeration."""

    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"


# Global state for lazy initialization
_MPL_INIT_STATE = MPLInitState.NOT_INITIALIZED
_MPL_INIT_LOCK = threading.Lock()
_PENDING_PREFERENCE_CHANGES: Dict[str, Any] = {}
_MPL_INIT_ERROR = None

# -----------------------------------------------------------------------------
# Internal storage for the user's original rcParams
#
# This snapshot is taken ONCE, lazily, just before SpectroChemPy modifies
# matplotlib global state.
# -----------------------------------------------------------------------------

_USER_RCPARAMS = None


def _is_mpl_initialized() -> bool:
    """Check if matplotlib has been initialized by SpectroChemPy."""
    import spectrochempy.core.plotters.plot_setup as plot_setup

    return plot_setup._MPL_INIT_STATE == MPLInitState.INITIALIZED


def _is_mpl_initializing() -> bool:
    """Check if matplotlib is currently being initialized."""
    import spectrochempy.core.plotters.plot_setup as plot_setup

    return plot_setup._MPL_INIT_STATE == MPLInitState.INITIALIZING


def _set_mpl_state(state: MPLInitState) -> None:
    """Set matplotlib initialization state."""
    import spectrochempy.core.plotters.plot_setup as plot_setup

    plot_setup._MPL_INIT_STATE = state


def _get_mpl_state() -> MPLInitState:
    """Get current matplotlib initialization state."""
    import spectrochempy.core.plotters.plot_setup as plot_setup

    return plot_setup._MPL_INIT_STATE


# -----------------------------------------------------------------------------
# Lazy preference deferral system
# -----------------------------------------------------------------------------


def _defer_preference_change(change) -> None:
    """Queue a preference change until matplotlib is initialized."""
    change_key = f"{change.name}:{change.new}"
    _PENDING_PREFERENCE_CHANGES[change_key] = change

    # Debug only if debug module is available (avoid circular import)
    try:
        from spectrochempy.application.application import debug_

        debug_(f"Deferred preference change: {change.name} = {change.new}")
    except ImportError:
        # Silently skip debug if circular import
        pass


def _apply_deferred_preferences() -> None:
    """Apply all queued preference changes after matplotlib initialization."""
    from spectrochempy.application.application import app, debug_, warning_

    if not _PENDING_PREFERENCE_CHANGES:
        return

    debug_(f"Applying {len(_PENDING_PREFERENCE_CHANGES)} deferred preference changes")

    plot_prefs = app.plot_preferences

    # DISABLED: No longer apply global rcParams
    # if plot_prefs:
        import matplotlib as mpl

        # DISABLED: No longer apply global rcParams from deferred changes
        # for change_key, change in _PENDING_PREFERENCE_CHANGES.items():
        #     try:
        #         # Apply the change directly to rcParams
        #         # Convert trait name to rcParams key
        #         rc_key = plot_prefs.to_rc_key(change.name)
        #         if rc_key in mpl.rcParams:
        #             mpl.rcParams[rc_key] = change.new
        #             debug_(f"Applied {rc_key} = {change.new}")
        #     else:
        #         warning_(f"Unknown rcParams key: {rc_key}")
        #     except Exception as e:
        #         warning_(f"Failed to apply deferred preference {change_key}: {e}")

    # Clear the queue
    _PENDING_PREFERENCE_CHANGES.clear()


def _synchronize_preferences_to_rcparams() -> None:
    """Synchronize all PlotPreferences to matplotlib rcParams."""
    from spectrochempy.application.preferences import preferences

    plot_prefs = preferences.get("plot", None)

    if not plot_prefs:
        return

    # Import matplotlib here (lazy)
    import matplotlib as mpl

    # Force update all rcParams from preferences
    for rckey in mpl.rcParams:
        key = rckey.replace("_", "__").replace(".", "_").replace("-", "___")
        try:
            value = getattr(plot_prefs, key)
            if value is not None:
                mpl.rcParams[rckey] = value
        except (ValueError, AttributeError):
            pass  # Graceful handling


def _snapshot_user_rcparams() -> None:
    """
    Snapshot the current matplotlib rcParams exactly once.

    Must be called immediately BEFORE SpectroChemPy modifies rcParams.
    """
    global _USER_RCPARAMS

    # Import matplotlib lazily
    import matplotlib as mpl

    if _USER_RCPARAMS is None:
        _USER_RCPARAMS = copy.deepcopy(mpl.rcParams)


def lazy_ensure_mpl_config() -> None:
    """
    LAZY initialization of ALL matplotlib functionality.

    This is the single entry point that replaces ALL matplotlib setup
    currently scattered across app.start() and other initialization points.

    This function is:
    - Thread-safe
    - Idempotent
    - Comprehensive (handles ALL matplotlib setup)
    - Error-resilient
    """
    # Fast path: already initialized
    if _is_mpl_initialized():
        return

    # Thread safety: ensure only one thread initializes
    with _MPL_INIT_LOCK:
        # Double-check pattern
        if _is_mpl_initialized():
            return

        if _is_mpl_initializing():
            # Another thread is initializing, wait for completion
            while _is_mpl_initializing():
                threading.Event().wait(0.01)  # 10ms polling
            return

        # Mark as initializing
        _set_mpl_state(MPLInitState.INITIALIZING)

        try:
            _perform_lazy_mpl_initialization()
            _set_mpl_state(MPLInitState.INITIALIZED)
            from spectrochempy.application.application import debug_

            debug_("Lazy matplotlib initialization completed successfully")

        except Exception as e:
            _set_mpl_state(MPLInitState.FAILED)
            global _MPL_INIT_ERROR
            _MPL_INIT_ERROR = e
            from spectrochempy.application.application import error_

            error_(e, "Failed to initialize matplotlib lazily")
            # Re-raise to let caller know initialization failed
            raise


def _perform_lazy_mpl_initialization() -> None:
    """
    Perform the actual matplotlib initialization.

    This contains ALL the matplotlib setup logic that was previously
    scattered across app.start(), ensure_spectrochempy_plot_style(), etc.
    """
    # ------------------------------------------------------------------
    # 0. Conditional cleanup of partially imported matplotlib
    # ------------------------------------------------------------------
    # Tests may leave matplotlib in a partially imported state where
    # essential attributes are missing. We detect this condition and
    # perform cleanup ONLY when necessary to preserve user imports.
    #
    # Unconditional cleanup is unsafe as it would reset user's matplotlib
    # configuration and break backward compatibility.
    import sys
    import gc

    needs_cleanup = False

    # Check if matplotlib is in sys.modules but partially initialized
    if "matplotlib" in sys.modules:
        matplotlib_module = sys.modules["matplotlib"]
        # Check for essential attributes that indicate full initialization
        required_attrs = ["artist", "rcParams", "use", "__version__"]
        is_partial = any(
            not hasattr(matplotlib_module, attr) for attr in required_attrs
        )

        if is_partial:
            needs_cleanup = True
    else:
        # Also check if there are matplotlib objects in memory but no module
        # This indicates a previous import was cleaned up improperly
        try:
            matplotlib_objects = [
                obj
                for obj in gc.get_objects()
                if hasattr(obj, "__module__")
                and obj.__module__
                and isinstance(obj.__module__, str)
                and obj.__module__.startswith("matplotlib")
            ]
            if matplotlib_objects:
                needs_cleanup = True
        except (TypeError, AttributeError):
            # If we can't check objects, err on side of caution
            needs_cleanup = True

    if needs_cleanup:
        # Remove all matplotlib modules to ensure clean re-import
        matplotlib_modules = [
            k for k in sys.modules.keys() if k.startswith("matplotlib")
        ]
        for mod in matplotlib_modules:
            del sys.modules[mod]

    # ------------------------------------------------------------------
    # 1. Ensure matplotlib is initialized safely (backend-safe)
    # ------------------------------------------------------------------
    # NOTE: Matplotlib bootstrap logic is intentionally inlined here.
    # Previously this was in ensure_mpl_setup() from _mpl_setup.py.
    # lazy_ensure_mpl_config() is now the single initialization path.
    # Safe matplotlib import
    try:
        import matplotlib
    except Exception:
        # Matplotlib not available or broken
        raise

    # Headless safety (Linux / CI)
    import os

    if os.name != "nt" and not os.environ.get("DISPLAY"):
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            raise

    # Sanity check: pyplot must be importable
    import sys

    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        # Clean partial imports to avoid poisoned state
        sys.modules.pop("matplotlib.pyplot", None)
        sys.modules.pop("matplotlib", None)
        raise

    # ------------------------------------------------------------------
    # 2. Snapshot user rcParams BEFORE touching anything
    # ------------------------------------------------------------------
    _snapshot_user_rcparams()

    # ------------------------------------------------------------------
    # 3. High-level imports (safe AFTER matplotlib backend setup)
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ------------------------------------------------------------------
    # 4. Install matplotlib assets (best effort)
    # ------------------------------------------------------------------
    from spectrochempy.core.plotters._mpl_assets import ensure_mpl_assets_installed

    with contextlib.suppress(Exception):
        ensure_mpl_assets_installed()

    # ------------------------------------------------------------------
    # 5. Synchronize current preferences to rcParams
    # ------------------------------------------------------------------
    # _synchronize_preferences_to_rcparams()  # DISABLED: No longer apply global rcParams

    # ------------------------------------------------------------------
    # 6. LaTeX font configuration
    # ------------------------------------------------------------------
    from spectrochempy.application.preferences import preferences

    plot_prefs = preferences.get("plot", None)
    if plot_prefs:
        plot_prefs.set_latex_font(plot_prefs.font_family)

    # ------------------------------------------------------------------
    # 7. Apply default style (classic + configured style)
    # ------------------------------------------------------------------
    # plt.style.use(["classic"])  # DISABLED: No longer apply global style

    if plot_prefs and plot_prefs.style:
        # DISABLE global style application
        # Note: Styles now applied only via local context in plotting functions
        pass

    # ------------------------------------------------------------------
    # 8. Apply all deferred preference changes LAST
    # ------------------------------------------------------------------
    # This must be last to override any other settings
    # _apply_deferred_preferences()  # DISABLED: No longer apply global rcParams


def ensure_spectrochempy_plot_style() -> None:
    """
    Legacy compatibility wrapper.

    Previously this function handled all matplotlib setup.
    Now it delegates to the new lazy system for backward compatibility.

    This ensures existing code that calls this function continues to work.
    """
    lazy_ensure_mpl_config()


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

    if _USER_RCPARAMS is None:
        # Nothing to restore → no-op
        return

    # Import matplotlib lazily for restoration
    import matplotlib as mpl

    mpl.rcParams.clear()
    mpl.rcParams.update(copy.deepcopy(_USER_RCPARAMS))


def get_import_time_rcparams():
    """Return a copy of the import-time matplotlib rcParams snapshot."""
    return copy.deepcopy(_USER_RCPARAMS)
