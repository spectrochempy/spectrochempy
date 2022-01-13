# -*- coding: utf-8 -*-
# flake8: noqa

# TODO: reduce the size of the data to accelerate the test
# TODO: Add example to imporve coverage

import spectrochempy
from spectrochempy import NDDataset, preferences as prefs, pathclean, info_

DATADIR = prefs.datadir


def dialog_carroucell(*args, **kwargs):
    # mock opening a dialog
    return DATADIR / "irdata/carroucell_samp"


def test_read_carroucell(monkeypatch):
    monkeypatch.setattr(spectrochempy.core, "open_dialog", dialog_carroucell)
    monkeypatch.setenv("KEEP_DIALOGS", "True")
    B = NDDataset.read_carroucell()
    assert B[3].shape == (6, 11098)

    A = NDDataset.read_carroucell("irdata/carroucell_samp", spectra=(1, 2))
    for x in A:
        info_("  " + x.name + ": " + str(x.shape))
    assert len(A) == 11
    assert A[3].shape == (2, 11098)
