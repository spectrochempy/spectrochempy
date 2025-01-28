# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

import spectrochempy as scp
from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset

DATADIR = prefs.datadir
AGIR_FOLDER = DATADIR / "agirdata"
IR_FOLDER = DATADIR / "irdata"


# @pytest.mark.skipif(
#     not AGIR_FOLDER.exists() or not IR_FOLDER.exists(),
#     reason="Experimental data not available for testing",
# )
def test_read_csv():
    prefs.csv_delimiter = ","

    A = scp.read_csv("agirdata/P350/TGA/tg.csv", directory=DATADIR, origin="tga")
    assert A.shape == (1, 3247)

    B = scp.read_csv("irdata/IR.CSV", origin="omnic")
    assert B.shape == (1, 3736)

    # Read CSV content
    p = DATADIR / "irdata" / "IR.CSV"
    content = p.read_bytes()
    C = scp.read_csv({"somename.csv": content})
    assert C.shape == (1, 3736)

    # wrong origin parameters
    D = scp.read_csv("irdata/IR.CSV", origin="opus")
    assert not D
