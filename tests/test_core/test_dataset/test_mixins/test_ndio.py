# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the ndplugin module"""

import pytest

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.testing import assert_array_equal, assert_dataset_equal

# Basic
# --------------------------------------------------------------------------------------


def test_ndio_generic(ndataset_1d, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ir = ndataset_1d
    ir.name = "IR_1D"

    # save with a default filename derived from the dataset name
    f = ir.save_as(tmp_path / ir.name)
    assert ir.filename.name == f.name
    assert ir.directory == tmp_path

    # load back this  file : the full path f is given so no dialog is opened
    nd = NDDataset.load(f)
    assert_dataset_equal(nd, ir)

    # as it has been already saved,
    f = nd.save()
    assert nd.filename.name == "IR_1D.scp"

    # now save it with a new name
    f = ir.save_as(tmp_path / "essai")
    assert ir.filename.name == f.name

    # remove these files
    f.unlink()

    # save in a specified directory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    ir.save_as(subdir / "essai")  # save essai.scp
    assert ir.directory == subdir
    assert ir.filename.name == "essai.scp"
    (subdir / ir.filename.name).unlink()

    # save in the current directory
    f = ir.save_as(tmp_path / "essai")

    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load("essai")
    # assert dl.directory == cwd
    assert_array_equal(dl.data, ir.data)
    f.unlink()


def test_ndio_2D(ndataset_2d, tmp_path):
    # test with a 2D

    ir2 = ndataset_2d.copy()
    f = ir2.save_as(tmp_path / "essai2D", confirm=False)
    assert ir2.directory == tmp_path
    with pytest.raises(FileNotFoundError):
        NDDataset.load("essai2D")
    nd = NDDataset.load(tmp_path / "essai2D")
    assert nd.directory == tmp_path
    f.unlink()


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
