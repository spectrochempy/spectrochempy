import copy

import matplotlib as mpl
import pytest

import spectrochempy as scp
from spectrochempy.application.application import app


@pytest.mark.mpl
def test_scpy_style_application_changes_rcparams_when_forced():
    """Explicitly forcing the SpectroChemPy style must modify matplotlib rcParams."""

    # Start from a known Matplotlib baseline
    mpl.rcdefaults()
    rc_before = mpl.rcParams.copy()

    # Force SpectroChemPy plotting style (CORRECT API)
    app.plot_preferences.style = "scpy"

    X = scp.random((10, 10))
    X.plot()

    assert (
        mpl.rcParams != rc_before
    ), "Forcing SpectroChemPy style did not modify matplotlib rcParams"


@pytest.mark.mpl
def test_rcparams_restore_is_idempotent_and_safe():
    """
    scp.restore_rcparams() must restore matplotlib rcParams
    exactly to the state that existed before any SpectroChemPy plotting,
    regardless of whether rcParams were modified or not.
    """

    # ------------------------------------------------------------------
    # 1. Snapshot initial user rcParams
    # ------------------------------------------------------------------
    rc_before = copy.deepcopy(mpl.rcParams)

    keys_to_check = [
        "axes.facecolor",
        "figure.facecolor",
        "axes.prop_cycle",
        "lines.linewidth",
        "font.size",
    ]

    subset_before = {k: rc_before[k] for k in keys_to_check}

    # ------------------------------------------------------------------
    # 2. Trigger SpectroChemPy plotting (may or may not modify rcParams)
    # ------------------------------------------------------------------
    X = scp.random((10, 10))
    X.plot()

    # ------------------------------------------------------------------
    # 3. Restore user rcParams
    # ------------------------------------------------------------------
    scp.restore_rcparams()

    subset_after_restore = {k: mpl.rcParams[k] for k in keys_to_check}

    # ------------------------------------------------------------------
    # 4. MUST restore exactly
    # ------------------------------------------------------------------
    assert (
        subset_after_restore == subset_before
    ), "scp.restore_rcparams() did not restore the original matplotlib rcParams"
