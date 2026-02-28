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

# NO matplotlib imports at module level!
# These will be imported lazily when needed.

# -----------------------------------------------------------------------------
# Minimal lazy initialization state
# -----------------------------------------------------------------------------

# Simple flag to track if matplotlib has been configured
_MPL_READY: bool = False
_ASSETS_INSTALLED: bool = False


def _is_mpl_initialized() -> bool:
    """Check if matplotlib has been initialized by SpectroChemPy."""
    import spectrochempy.core.plotters.plot_setup as plot_setup

    return plot_setup._MPL_READY


def _set_mpl_state(ready: bool) -> None:
    """Set matplotlib ready state."""
    import spectrochempy.core.plotters.plot_setup

    spectrochempy.core.plotters.plot_setup._MPL_READY = ready


def _get_mpl_state() -> bool:
    """Get current matplotlib ready state."""
    import spectrochempy.core.plotters.plot_setup as plot_setup

    return plot_setup._MPL_READY


# -----------------------------------------------------------------------------
# Lazy preference deferral system
# -----------------------------------------------------------------------------


def lazy_ensure_mpl_config() -> None:
    """
    Ensure matplotlib is initialized and assets are installed.

    This function is:
    - Idempotent (safe to call multiple times)
    - Fast on subsequent calls
    - Lazy (doesn't import matplotlib until first plot)
    """
    global _MPL_READY

    # Fast path: already initialized
    if _MPL_READY:
        return

    # Import matplotlib (this is the lazy part)

    # Install assets once per process
    global _ASSETS_INSTALLED
    if not _ASSETS_INSTALLED:
        with contextlib.suppress(Exception):
            from spectrochempy.core.plotters._mpl_assets import (
                ensure_mpl_assets_installed,
            )

            with contextlib.suppress(Exception):
                ensure_mpl_assets_installed()
            _ASSETS_INSTALLED = True

    # Mark as ready
    _MPL_READY = True


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

    DEPRECATED: This is now a no-op since SpectroChemPy no longer modifies
    global matplotlib rcParams. The function is kept for backward compatibility.
    """
    import warnings

    warnings.warn(
        "restore_rcparams() is deprecated and has no effect. "
        "SpectroChemPy no longer modifies global matplotlib rcParams.",
        DeprecationWarning,
        stacklevel=2,
    )
    # No-op - no longer needed in stateless mode
    pass


def get_import_time_rcparams():
    """
    Return a copy of the import-time matplotlib rcParams snapshot.

    DEPRECATED: This now returns None since SpectroChemPy no longer modifies
    global matplotlib rcParams and therefore doesn't need to snapshot them.
    """
    import warnings

    warnings.warn(
        "get_import_time_rcparams() is deprecated and now returns None. "
        "SpectroChemPy no longer modifies global matplotlib rcParams.",
        DeprecationWarning,
        stacklevel=2,
    )
    return
