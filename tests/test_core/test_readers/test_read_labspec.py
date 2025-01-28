# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from pathlib import Path

import pytest

import spectrochempy as scp

RAMANDIR = scp.preferences.datadir / "ramandata/labspec"


# @pytest.mark.skipif(
#     not RAMANDIR.exists(),
#     reason="Experimental data not available for testing",
# )
def test_read_labspec():
    # single file
    nd = scp.read_labspec("Activation.txt", directory=RAMANDIR)
    assert nd.shape == (532, 1024)

    # with read_dir
    # First download data as read_dir will not
    scp.read(RAMANDIR / "subdir", replace_existing=False)

    nd = scp.read_dir(directory=RAMANDIR / "subdir")
    assert nd.shape == (6, 1024)

    # empty txt file
    Path("i_am_empty.txt").touch()
    f = Path("i_am_empty.txt")
    nd = scp.read_labspec(f)
    f.unlink()
    assert nd is None

    # non labspec txt file
    f = Path("i_am_not_labspec.txt")
    f.write_text("blah")
    nd = scp.read_labspec(f)
    f.unlink()
    assert nd is None
