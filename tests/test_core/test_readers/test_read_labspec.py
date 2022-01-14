# -*- coding: utf-8 -*-
# flake8: noqa

import pytest

import spectrochempy as scp
from spectrochempy.utils import show

RAMANDIR = scp.preferences.datadir / "ramandata"


@pytest.mark.skipif(
    not RAMANDIR.exists(),
    reason="Experimental data not available for testing",
)
def test_read_labspec():

    nd = scp.read_labspec("Activation.txt", directory=RAMANDIR)
    assert nd.shape == (532, 1024)

    nd = scp.read_dir(directory=RAMANDIR / "subdir")
    assert nd.shape == (6, 1024)
