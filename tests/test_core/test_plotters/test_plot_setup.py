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


import pytest

from spectrochempy.core.plotters.plot_setup import get_import_time_rcparams


@pytest.mark.mpl
def test_rcparams_restore_restores_import_time_state():
    """
    scp.restore_rcparams() must restore matplotlib rcParams exactly
    to the state that existed when SpectroChemPy was imported.
    """

    keys_to_check = [
        "axes.facecolor",
        "figure.facecolor",
        "axes.prop_cycle",
        "lines.linewidth",
        "font.size",
    ]

    # Reference = import-time snapshot
    import_time = get_import_time_rcparams()
    reference = {k: import_time[k] for k in keys_to_check}

    # Trigger SCP plotting
    X = scp.random((10, 10))
    X.plot()

    # Restore
    scp.restore_rcparams()

    restored = {k: mpl.rcParams[k] for k in keys_to_check}

    assert restored == reference
