# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

from spectrochempy import read_quadera
from spectrochempy import preferences as prefs

DATADIR = prefs.datadir
MSDATA = DATADIR / "msdata"


def test_read_quadera():
    # single file
    A = read_quadera(MSDATA / "ion_currents.asc")
    assert str(A) == "NDDataset: [float64] A (shape: (y:16975, x:10))"
