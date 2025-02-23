# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

# TODO: reduce the size of the data to accelerate the test
# TODO: Add example to improve coverage
import pytest

import spectrochempy as scp
from spectrochempy import NDDataset, info_
from spectrochempy import preferences as prefs

try:
    from spectrochempy.core.common import dialogs
except ImportError:
    pytest.skip("dialogs not available with act", allow_module_level=True)

DATADIR = prefs.datadir
CARROUCELL_FOLDER = DATADIR / "irdata/carroucell_samp"


def dialog_carroucell(*args, **kwargs):
    # mock opening a dialog
    return CARROUCELL_FOLDER


# @pytest.mark.skipif(
#         not CARROUCELL_FOLDER.exists(), reason="Experimental data not available for testing"
#         )
def test_read_carroucell(monkeypatch):
    # # Before testing we need to download the data locally if not yet done:
    # # because read carrousel is not designed to download itself.
    # # Use the read for that:
    # scp.read("irdata/carroucell_samp", replace_existing=False)

    nd = scp.read_carroucell("irdata/carroucell_samp", spectra=(1, 2))
    for x in nd:
        info_("  " + x.name + ": " + str(x.shape))
    assert len(nd) == 11
    assert nd[3].shape == (2, 11098)

    nd = scp.read_carroucell("irdata/carroucell_samp", spectra=(1, 1))
    assert isinstance(nd, NDDataset)

    monkeypatch.setattr(dialogs, "open_dialog", dialog_carroucell)
    monkeypatch.setenv("KEEP_DIALOGS", "True")
    nd = scp.read_carroucell(spectra=(1, 3))
    assert nd[3].shape == (3, 11098)

    nd = scp.read_carroucell(spectra=(2, 3), discardbg=False)
    assert nd[3].shape == (2, 11098)
