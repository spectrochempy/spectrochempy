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
from spectrochempy.utils import pathclean

irdatadir = pathclean(prefs.datadir) / "irdata"
cwd = pathlib.Path.cwd()


# Basic
# ----------------------------------------------------------------------------------------------------------------------
def test_ndio_generic(IR_dataset_1D):
    ir = IR_dataset_1D
    assert ir.filename == 'nh4y-activation.spg'
    assert ir.directory == irdatadir

    # save with the default name (equivalent to save_as in this case)
    # as this file doesn't yet exist a confirmation is opened
    ir.save()
    assert ir.filename == 'nh4y-activation.scp'
    assert ir.directory == irdatadir

    ir.save()  # as it has been already saved, we should not get dialogs
    assert ir.filename == 'nh4y-activation.scp'

    f = ir.save_as()  # now it opens a dialog and the name can be changed
    assert ir.filename == f.name

    # save in the self.directory with a new name without dialog
    ir.save_as('essai')  # save essai.scp
    assert ir.directory == cwd  # should not change
    assert ir.filename == "essai.scp"
    f.unlink()

    # save in a specified directory
    ir.save_as(irdatadir / 'essai')  # save essai.scp
    assert ir.directory == irdatadir
    assert ir.filename == "essai.scp"

    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load('essai')
    assert dl.directory == cwd
    assert_array_equal(dl.data, ir.data)

    for f in ['essai.scp', 'nh4y-activation.scp']:
        if (irdatadir / f).is_file():
            (irdatadir / f).unlink()


def test_ndio_2D(IR_dataset_2D):
    # test with a 2D

    ir2 = IR_dataset_2D.copy()
    f = ir2.save_as('essai2D')
    assert ir2.directory == irdatadir
    with pytest.raises(FileNotFoundError):
        nd = NDDataset.load("essai2D")
    nd = NDDataset.load("irdata/essai2D")
    assert nd.directory == irdatadir
    f.unlink()

# EOF
