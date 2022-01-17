# -*- coding: utf-8 -*-
# flake8: noqa


from pathlib import Path

import spectrochempy as scp


def test_read_spc():

    A = scp.read_spc("galacticdata/BARBITUATES.spc")
    # multi file, can't be read yet
    assert A == []

    B = scp.read_spc("galacticdata/barbsvd.spc")
    # multi file, can't be read yet
    assert B == []

    C = scp.read_spc("galacticdata/BARBITUATES.spc")
    assert "Dataset from spc file." in A.description
