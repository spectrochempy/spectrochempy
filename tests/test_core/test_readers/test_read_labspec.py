# -*- coding: utf-8 -*-
# flake8: noqa

import pytest

import spectrochempy as scp
from pathlib import Path

RAMANDIR = scp.preferences.datadir / "ramandata"


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
    scp.read_remote(RAMANDIR / "subdir", replace_existing=False)

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
