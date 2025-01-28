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

    # read contents
    p = OPUSDATA / "test.0000"
    content = p.read_bytes()
    F = NDDataset.read_opus({p.name: content})
    assert F.name == p.name
    assert F.shape == (1, 2567)

    assert NDDataset.read_opus(OPUSDATA / "background.0") is None
