# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path

import spectrochempy as spc


def test_read_spc():
    A = spc.read_spc("galacticdata/000001_Spectrum.spc")
    assert "Dataset from spc file." in A.description
