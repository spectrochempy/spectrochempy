# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

import spectrochempy as scp
from spectrochempy.core import preferences as prefs

DATADIR = prefs.datadir
IRDATA = DATADIR / "irdata"


def test_read():
    filename = IRDATA / "CO@Mo_Al2O3.SPG"

    # read normally
    nd1 = scp.read_omnic(filename)
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

    # delete file to simulate its absence:
    filename.unlink()

    # now try to download from github s not found locally (use _read_remote)
    nd2 = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

    # delete file to simulate its absence:
    filename.unlink()

    # now try to download from github s not found locally (use _read_remote)
    # but file doesn't exist on github
    with pytest.raises(FileNotFoundError):
        scp.read_omnic("irdata/nh4y-active.spg")

    # now try a using generic read
    assert not filename.exists()
    nd2 = scp.read("irdata/CO@Mo_Al2O3.SPG")
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"
    assert filename.exists()

    # now try a using generic read with a missing
    with pytest.raises(FileNotFoundError):
        scp.read("irdata/nh4y-acti.spg")

    # download from url
    ds = scp.read("http://www.eigenvector.com/data/Corn/corn.mat")
    assert len(ds) == 7
    # assert "Downloaded from " in str(ds[0].history)  # FIXME: add history

    # generic read
    ds1 = scp.read("http://www.eigenvector.com/data/Corn/corn.mat")
    assert len(ds1) == 7

    # specific
    ds2 = scp.read_mat("http://www.eigenvector.com/data/Corn/corn.mat")
    assert len(ds2) == 7

    # generic read of a zipped file
    ds3 = scp.read("https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip")
    assert len(ds3) == 7

    # doesn't exist
    with pytest.raises(FileNotFoundError):
        scp.read("http://www.eigenvector.com/does_not_exist.mat")

    # not a scpy readable type
    with pytest.raises(TypeError):
        scp.read("https://www.spectrochempy.fr/latest/index.html")
