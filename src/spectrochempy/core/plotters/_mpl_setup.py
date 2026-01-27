# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================


"""
Lazy and reversible Matplotlib setup for SpectroChemPy.

- Silent by default
- Thread-safe
- Respects user Matplotlib configuration
- Uses SpectroChemPy logging helpers
"""

from os import environ
from threading import Lock

from spectrochempy.application.application import debug_
from spectrochempy.application.application import error_

_MPL_INITIALIZED = False
_MPL_ORIGINAL_RCPARAMS = None
_LOCK = Lock()


def ensure_mpl_setup():
    """
    Lazily initialize Matplotlib for SpectroChemPy plotting.

    This function:
    - Imports Matplotlib only when needed
    - Saves the user's original rcParams
    - Installs assets if missing (best-effort)
    - Applies the SpectroChemPy plotting style
    """
    global _MPL_INITIALIZED, _MPL_ORIGINAL_RCPARAMS

    with _LOCK:
        if _MPL_INITIALIZED:
            return

        try:
            import matplotlib as mpl
            import matplotlib.backend_bases
        except ImportError:
            debug_(debug_("Matplotlib not available; skipping plotting setup"))
            return

        # If disable: we stop there (but matplotlib is imported)
        if environ.get("SCPY_DISABLE_MPL_SETUP"):
            debug_(
                "Matplotlib setup disabled via SCPY_DISABLE_MPL_SETUP (imports done)"
            )
            _MPL_INITIALIZED = True
            return

        debug_("Initializing Matplotlib for SpectroChemPy")

        # Save user configuration
        _MPL_ORIGINAL_RCPARAMS = mpl.rcParams.copy()

        # Install styles/fonts if needed (silent, best-effort)
        from ._mpl_assets import ensure_mpl_assets_installed

        ensure_mpl_assets_installed()

        # Apply SpectroChemPy style (strict)
        _apply_scpy_style()

        _MPL_INITIALIZED = True


def _apply_scpy_style():
    """
    Apply the SpectroChemPy Matplotlib style.

    Raises
    ------
    RuntimeError
        If the SpectroChemPy style is unavailable.
    """
    import matplotlib.pyplot as plt

    try:
        plt.style.use("scpy")
        debug_("SpectroChemPy matplotlib style applied")
    except OSError as exc:
        error_(
            "SpectroChemPy matplotlib style is unavailable. "
            "The style installation may have failed."
        )
        raise RuntimeError(
            "scpy matplotlib style is unavailable. "
            "Please reinstall SpectroChemPy or install the styles manually."
        ) from exc


def reset_mpl(close_figures: bool = True):
    """
    Restore Matplotlib to its original state prior to SpectroChemPy usage.

    Parameters
    ----------
    close_figures : bool, default=True
        Whether to close all open Matplotlib figures.
    """
    global _MPL_INITIALIZED, _MPL_ORIGINAL_RCPARAMS

    with _LOCK:
        if not _MPL_INITIALIZED:
            return

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        debug_("Resetting Matplotlib state (close_figures=%s)", close_figures)

        if close_figures:
            plt.close("all")

        if _MPL_ORIGINAL_RCPARAMS is not None:
            mpl.rcParams.update(_MPL_ORIGINAL_RCPARAMS)

        _MPL_INITIALIZED = False
        _MPL_ORIGINAL_RCPARAMS = None
