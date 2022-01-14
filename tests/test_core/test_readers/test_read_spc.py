# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path

import spectrochempy as spc
from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset


def test_read_spc():

    A = spc.read_spc("galacticdata/000001_Spectrum.spc")
    assert A.description == "Dataset from spc file."
