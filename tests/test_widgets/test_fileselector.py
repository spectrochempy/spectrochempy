# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import pytest

import spectrochempy as scp

pytest.mark.skipif(
    pytest.importorskip("ipywidgets", reason="ipywidgets not installed") is None,
    reason="ipywidgets not installed",
)


def test_fileselector():
    datadir = scp.preferences.datadir
    fs = scp.FileSelector(path=datadir, filters=["spg", "spa"])
    # no selection possible if we are not in a notebook
    assert fs.value is None
    assert fs.path.name == "testdata"
    assert fs.fullpath.parent.name == datadir.parent.name

    fs.up()
    assert fs.path.name == datadir.parent.name
