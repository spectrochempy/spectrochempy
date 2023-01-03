# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import spectrochempy as scp


def test_fileselector():

    datadir = scp.preferences.datadir
    fs = scp.FileSelector(path=datadir, filters=["spg", "spa"])
    # no selection possible if we are not in a notebook
    assert fs.value is None
    assert fs.path.name == "testdata"
    assert fs.fullpath.parent.name == datadir.parent.name

    fs.up()
    assert fs.path.name == datadir.parent.name
