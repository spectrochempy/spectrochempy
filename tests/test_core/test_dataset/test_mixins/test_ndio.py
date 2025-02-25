# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the ndplugin module"""

import pathlib

import pytest

from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.file import pathclean
from spectrochempy.utils.testing import assert_array_equal, assert_dataset_equal

irdatadir = pathclean(prefs.datadir) / "irdata"
nmrdatadir = pathclean(prefs.datadir) / "nmrdata" / "bruker" / "tests" / "nmr"
cwd = pathlib.Path.cwd()

try:
    from spectrochempy.core.common import dialogs
except ImportError:
    pytest.skip("dialogs not available with act", allow_module_level=True)


# Basic
# --------------------------------------------------------------------------------------


def test_ndio_generic(NMR_dataset_1D):
    nmr = NMR_dataset_1D
    assert nmr.directory == nmrdatadir

    # save with the default filename or open a dialog if it doesn't exists
    # ----------------------------------------------------------------------------------
    # save with the default name (equivalent to save_as in this case)
    # as this file (IR_1D.scp)  doesn't yet exist a confirmation dialog is opened
    f = nmr.save()
    assert nmr.filename.name == f.name
    assert nmr.directory == nmrdatadir

    # load back this  file : the full path f is given so no dialog is opened
    nd = NDDataset.load(f)
    assert_dataset_equal(nd, nmr)

    # as it has been already saved, we should not get dialogs
    f = nd.save()
    assert nd.filename.name == "NMR_1D.scp"
    # return

    # now it opens a dialog and the name can be changed
    f1 = nmr.save_as()
    assert nmr.filename.name == f1.name

    # remove these files
    f.unlink()
    f1.unlink()

    # save in a specified directory
    nmr.save_as(irdatadir / "essai")  # save essai.scp
    assert nmr.directory == irdatadir
    assert nmr.filename.name == "essai.scp"
    (irdatadir / nmr.filename.name).unlink()

    # save in the current directory
    f = nmr.save_as(cwd / "essai")

    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load("essai")
    # assert dl.directory == cwd
    assert_array_equal(dl.data, nmr.data)
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
