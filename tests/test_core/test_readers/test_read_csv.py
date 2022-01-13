# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path
import pytest

import spectrochempy as scp
from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset

DATADIR = prefs.datadir
AGIR_FOLDER = DATADIR / "agirdata"
IR_FOLDER = DATADIR / "irdata"


@pytest.mark.skipif(
    not AGIR_FOLDER.exists() or not IR_FOLDER.exists(),
    reason="Experimental data not available for testing",
)
def test_read_csv():
    datadir = prefs.datadir
    prefs.csv_delimiter = ","

    A = NDDataset.read_csv("agirdata/P350/TGA/tg.csv", directory=datadir, origin="tga")
    assert A.shape == (1, 3247)

    B = NDDataset.read_csv("irdata/IR.CSV", origin="omnic")
    assert B.shape == (1, 3736)

    # without directory
    C = NDDataset.read_csv("irdata/IR.CSV")
    assert C.shape == (1, 3736)

    # pathlib.Path objects can be used instead of string for filenames
    p = Path(datadir) / "irdata" / "IR.CSV"
    D = scp.read_csv(p)
    assert D == C

    # Read CSV content
    content = p.read_bytes()
    E = scp.read_csv({"somename.csv": content})
    assert E == C

    F = NDDataset.read_csv("irdata/IR.CSV", origin="opus")
    assert not F
