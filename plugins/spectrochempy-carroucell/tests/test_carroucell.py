# ruff: noqa: S101  # assert allowed in tests

"""Tests for spectrochempy-carroucell."""

import pytest

import spectrochempy as scp
from spectrochempy import NDDataset
from spectrochempy import preferences as prefs

DATADIR = prefs.datadir
CARROUCELL_FOLDER = DATADIR / "irdata/carroucell_samp"


def dialog_carroucell(*args, **kwargs):
    # mock opening a dialog
    return CARROUCELL_FOLDER


@pytest.mark.skipif(
    not CARROUCELL_FOLDER.exists(),
    reason="Experimental data not available for testing",
)
def test_read_carroucell(monkeypatch):
    nd = scp.carroucell.read_carroucell("irdata/carroucell_samp", spectra=(1, 2))
    assert len(nd) == 11
    assert nd[3].shape == (2, 11098)

    nd = scp.carroucell.read_carroucell("irdata/carroucell_samp", spectra=(1, 1))
    assert isinstance(nd, NDDataset)

    nd = scp.carroucell.read_carroucell("irdata/carroucell_samp", spectra=(1, 3))
    assert nd[3].shape == (3, 11098)

    nd = scp.carroucell.read_carroucell(
        "irdata/carroucell_samp", spectra=(2, 3), discardbg=False
    )
    assert nd[3].shape == (2, 11098)
