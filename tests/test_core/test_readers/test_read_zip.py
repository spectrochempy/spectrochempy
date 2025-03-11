# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa


import pytest

from spectrochempy.application.preferences import preferences as prefs
from spectrochempy import read_zip

DATADIR = prefs.datadir
AGIRDATA = DATADIR / "agirdata"


# @pytest.mark.skipif(
#     not AGIRDATA.exists(),
#     reason="Experimental data not available for testing",
# )
def test_read_zip():
    A = read_zip(
        "agirdata/P350/FTIR/FTIR.zip",
        origin="omnic",
        only=10,
        csv_delimiter=";",
        merge=True,
    )
    assert A.shape == (10, 2843)

    # Test bytes contents for ZIP files
    z = DATADIR / "agirdata" / "P350" / "FTIR" / "FTIR.zip"
    content2 = z.read_bytes()
    B = read_zip(
        {"name.zip": content2}, origin="omnic", only=10, csv_delimiter=";", merge=True
    )
    assert B.shape == (10, 2843)

    # Test read_zip with several contents
    C = read_zip(
        {"name1.zip": content2, "name2.zip": content2},
        origin="omnic",
        only=10,
        csv_delimiter=";",
        merge=True,
    )
    assert C.shape == (20, 2843)
