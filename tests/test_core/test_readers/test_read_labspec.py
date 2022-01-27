# -*- coding: utf-8 -*-
# flake8: noqa

import pytest

import spectrochempy as scp
import os

RAMANDIR = scp.preferences.datadir / "ramandata"


@pytest.mark.skipif(
    not RAMANDIR.exists(),
    reason="Experimental data not available for testing",
)
def test_read_labspec():

    # single file
    nd = scp.read_labspec("Activation.txt", directory=RAMANDIR)
    assert nd.shape == (532, 1024)

    # with read_dir
    nd = scp.read_dir(directory=RAMANDIR / "subdir")
    assert nd.shape == (6, 1024)

    # empty txt file
    with open("i_am_empty.txt", "w") as f:
        f.close()
    nd = scp.read_labspec("i_am_empty.txt")
    os.remove("i_am_empty.txt")
    assert nd is None

    # non labspec txt file
    with open("i_am_not_labspec.txt", "w") as f:
        f.write("blabla")
        f.close()
    nd = scp.read_labspec("i_am_not_labspec.txt")
    os.remove("i_am_not_labspec.txt")
    assert nd is None
