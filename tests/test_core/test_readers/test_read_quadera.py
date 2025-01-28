# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

from spectrochempy import NDDataset
from spectrochempy import preferences as prefs

DATADIR = prefs.datadir
MSDATA = DATADIR / "msdata"


# @pytest.mark.skipif(
#     not MSDATA.exists(),
#     reason="Experimental data not available for testing",
# )


def test_read_quadera():
    # single file
    A = NDDataset.read_quadera(MSDATA / "ion_currents.asc")
    assert str(A) == "NDDataset: [float64] A (shape: (y:16975, x:10))"
