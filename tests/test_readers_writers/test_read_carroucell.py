# -*- coding: utf-8 -*-
# flake8: noqa


import os

import pytest

from spectrochempy.core import info_
from spectrochempy.core.dataset.nddataset import NDDataset


# uncomment the next line to test it manually
@pytest.mark.skip("interactive so cannot be used with full testing")
def test_read_carroucell_without_dirname():
    NDDataset.read_carroucell()


def test_read_carroucell_with_dirname():
    A = NDDataset.read_carroucell(os.path.join("irdata", "carroucell_samp"))
    for x in A:
        info_("  " + x.name + ": " + str(x.shape))
    assert len(A) == 11
    assert A[3].shape == (6, 11098)
