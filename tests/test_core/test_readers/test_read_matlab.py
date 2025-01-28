# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

from spectrochempy import NDDataset
from spectrochempy import preferences as prefs

MATLABDATA = prefs.datadir / "matlabdata"


# @pytest.mark.skipif(
#     not MATLABDATA.exists(),
#     reason="Experimental data not available for testing",
# )
def test_read_matlab():
    A = NDDataset.read_matlab(MATLABDATA / "als2004dataset.MAT")
    assert len(A) == 6
    assert A[3].shape == (4, 96)

    A = NDDataset.read_matlab(MATLABDATA / "dso.mat")
    assert A.name == "Group sust_base line withoutEQU.SPG"
    assert A.shape == (20, 426)
