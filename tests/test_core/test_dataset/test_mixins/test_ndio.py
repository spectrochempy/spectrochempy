# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the ndplugin module"""

import pathlib
import tempfile

import pytest

from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.file import pathclean
from spectrochempy.utils.testing import assert_array_equal, assert_dataset_equal


irdatadir = pathclean(prefs.datadir) / "irdata"
nmrdatadir = pathclean(prefs.datadir) / "nmrdata" / "bruker" / "tests" / "nmr"
cwd = pathlib.Path.cwd()

# Basic
# --------------------------------------------------------------------------------------


def test_ndio_generic(IR_dataset_1D):
    ir = IR_dataset_1D
    assert ir.directory == irdatadir

    # save with the default filename or open a dialog if it doesn't exists
    # ----------------------------------------------------------------------------------
    # save with the default name (equivalent to save_as in this case)
    # as this file (IR_1D.scp)  doesn't yet exist a confirmation dialog is opened
    f = ir.save()
    assert ir.filename.name == f.name
    assert ir.directory == irdatadir

    # load back this  file : the full path f is given so no dialog is opened
    nd = NDDataset.load(f)
    assert_dataset_equal(nd, ir)

    # as it has been already saved,
    f = nd.save()
    assert nd.filename.name == "IR_1D.scp"
    # return

    # now save it with a new name
    f = ir.save_as("essai")
    assert ir.filename.name == f.name

    # remove these files
    f.unlink()

    # save in a specified directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        ir.save_as(tmpdir / "essai")  # save essai.scp
        assert ir.directory == tmpdir
        assert ir.filename.name == "essai.scp"
        (tmpdir / ir.filename.name).unlink()

    # save in the current directory
    f = ir.save_as(cwd / "essai")

    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load("essai")
    # assert dl.directory == cwd
    assert_array_equal(dl.data, ir.data)
    f.unlink()


def test_ndio_2D(IR_dataset_2D):
    # test with a 2D

    ir2 = IR_dataset_2D.copy()
    f = ir2.save_as("essai2D", confirm=False)
    assert ir2.directory == irdatadir
    with pytest.raises(FileNotFoundError):
        NDDataset.load("essai2D")
    nd = NDDataset.load(prefs.datadir / "irdata/essai2D")
    assert nd.directory == irdatadir
    f.unlink()


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
