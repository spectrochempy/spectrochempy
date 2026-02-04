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


import spectrochempy.core.plotters.plot_setup as plot_setup


@pytest.mark.mpl
def test_rcparams_restore_restores_import_time_state():
    """
    scp.restore_rcparams() must restore matplotlib rcParams
    to the state captured BEFORE SpectroChemPy modifies them.
    """

    mpl.rcdefaults()

    # Explicitly trigger the lazy snapshot
    plot_setup._snapshot_user_rcparams()

    keys_to_check = [
        "axes.facecolor",
        "figure.facecolor",
        "axes.prop_cycle",
        "lines.linewidth",
        "font.size",
    ]

    reference = {k: plot_setup._USER_RCPARAMS[k] for k in keys_to_check}

    # Force SpectroChemPy style
    scp.preferences.plot_preferences.style = "scpy"

    # Restore
    scp.restore_rcparams()

    for k, v in reference.items():
        assert mpl.rcParams[k] == v
