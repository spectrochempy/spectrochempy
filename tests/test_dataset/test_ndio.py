# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

""" Tests for the ndplugin module

"""

import pathlib
import pytest

from spectrochempy.utils.testing import assert_dataset_equal
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import general_preferences as prefs
from spectrochempy.utils.testing import assert_array_equal
from spectrochempy.utils import pathclean, json_decoder

irdatadir = pathclean(prefs.datadir) / "irdata"
nmrdatadir = pathclean(prefs.datadir) / "nmrdata" / "bruker" / "tests" / "nmr"
cwd = pathlib.Path.cwd()


# Basic
# ----------------------------------------------------------------------------------------------------------------------
def test_ndio_generic(NMR_dataset_1D):

    nmr = NMR_dataset_1D
    assert nmr.directory == nmrdatadir

    # save with the default filename or open a dialog if it doesn't exists
    # ------------------------------------------------------------------------------------------------------------------
    # save with the default name (equivalent to save_as in this case)
    # as this file (IR_1D.scp)  doesn't yet exist a confirmation dialog is opened
    f = nmr.save()
    assert nmr.filename == f.name
    assert nmr.directory == nmrdatadir

    # load back this  file : the full path f is given so no dialog is opened
    nd = NDDataset.load(f)
    assert_dataset_equal(nd, nmr)

    # as it has been already saved, we should not get dialogs
    f = nd.save()
    assert nd.filename == 'NMR_1D.scp'
    #return

    # now it opens a dialog and the name can be changed
    f1 = nmr.save_as()
    assert nmr.filename == f1.name

    # remove these files
    f.unlink()
    f1.unlink()

    # save in a specified directory
    nmr.save_as(irdatadir / 'essai')  # save essai.scp
    assert nmr.directory == irdatadir
    assert nmr.filename == "essai.scp"

    # save if the current directory
    f = nmr.save_as(cwd / 'essai')
    print(f)
    # try to load without extension specification (will first assume it is scp)
    dl = NDDataset.load('essai')
    # assert dl.directory == cwd
    assert_array_equal(dl.data, nmr.data)

    for f in list(cwd.parent.parent.glob('**/*.scp')):
        f.unlink()


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
