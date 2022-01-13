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

    nd = NDDataset.read_carroucell("irdata/carroucell_samp", spectra=(1, 2))
    for x in nd:
        info_("  " + x.name + ": " + str(x.shape))
    assert len(nd) == 11
    assert nd[3].shape == (2, 11098)

    nd = NDDataset.read_carroucell("irdata/carroucell_samp", spectra=(1, 1))
    assert isinstance(nd, NDDataset)

    monkeypatch.setattr(spectrochempy.core, "open_dialog", dialog_carroucell)
    monkeypatch.setenv("KEEP_DIALOGS", "True")
    nd = NDDataset.read_carroucell(spectra=(1, 3))
    assert nd[3].shape == (3, 11098)

    nd = NDDataset.read_carroucell(spectra=(2, 3), discardbg=False)
    assert nd[3].shape == (2, 11098)
