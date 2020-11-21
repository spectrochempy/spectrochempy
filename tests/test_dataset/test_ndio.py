# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

""" Tests for the ndplugin module

"""


import pathlib

import pytest

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import general_preferences as prefs
from spectrochempy.utils.testing import assert_array_equal
from spectrochempy.utils import pathclean, SpectroChemPyException

# Basic
# ----------------------------------------------------------------------------------------------------------------------
def test_ndio_generic(IR_dataset_1D, IR_dataset_2D):

    ir = IR_dataset_1D
    assert ir.directory == pathclean(prefs.datadir) / "irdata"

    # save in the self.directory
    path = ir.save('essai')               # save essai.scp
    assert ir.directory == pathclean(prefs.datadir) / "irdata" # should not change
    assert path.name == "essai.scp"
    assert path.parent == pathlib.Path.cwd()
    assert path.suffix == ".scp"

    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)
    assert dl.directory == pathlib.Path.cwd()

    path.unlink()

    # save in the same directory as original
    path = ir.save('essai', same_dir=True)
    assert path.parent == ir.directory

    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load('irdata/essai')
    assert_array_equal(dl.data, ir.data)

    # or with extension
    dl = NDDataset.load('irdata/essai.scp')
    assert_array_equal(dl.data, ir.data)

    # this should fail as the file is not saved at the root of data_dir
    with pytest.raises(SpectroChemPyException):
        dl = NDDataset.load('essai.scp')
    path.unlink()                         # remove this test file

    # test with a 2D
    ir2 = IR_dataset_2D.copy()
    path = ir2.save('essai2D')
    assert path.parent == pathlib.Path.cwd()
    dl2 = NDDataset.load("essai2D")
    assert dl2.directory == pathlib.Path.cwd()
    path.unlink()

    # save with no filename
    path = ir2.save()
    assert path.stem == IR_dataset_2D.name

