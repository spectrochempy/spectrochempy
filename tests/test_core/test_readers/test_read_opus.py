# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa


import pytest

import spectrochempy as scp
from spectrochempy import NDDataset
from spectrochempy import preferences as prefs
from spectrochempy.utils.testing import assert_dataset_equal

DATADIR = prefs.datadir
OPUSDATA = DATADIR / "irdata" / "OPUS"


# @pytest.mark.skipif(
#     not OPUSDATA.exists(),
#     reason="Experimental data not available for testing",
# )


def test_read_opus():
    # single file
    A = NDDataset.read_opus(OPUSDATA / "test.0000")
    assert A.shape == (1, 2567)
    assert A[0, 2303.8694].data == pytest.approx(2.72740, 0.00001)
    assert A.units == "absorbance"

    # background
    B = NDDataset.read_opus(OPUSDATA / "background.0", type="RF")
    assert B.shape == (1, 4096)
    assert B.units is None

    # Test if the background.0 file is correctly inferered as a reference spectrum
    C = NDDataset.read_opus(OPUSDATA / "background.0")
    assert C.shape == (1, 4096)
    assert_dataset_equal(B, C)

    # read contents
    p = OPUSDATA / "test.0000"
    content = p.read_bytes()
    F = NDDataset.read_opus({p.name: content})
    assert F.name == p.name
    assert F.shape == (1, 2567)


if __name__ == "__main__":
    pytest.main([__file__])
