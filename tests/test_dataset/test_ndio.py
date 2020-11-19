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


def test_generic_read():
    import time

    # filename + extension specified
    start = time.time()
    ds = NDDataset.read('wodger.spg')
    t1 = (time.time() - start)
    assert ds.name == 'wodger'

    # save with no filename (should save wodger.scp)
    path = ds.save()

    assert isinstance(path, pathlib.Path)
    assert path.stem == ds.name
    assert path.parent == ds.directory

    # should be équivalent to load (but read is a more general function
    start = time.time()
    dataset = NDDataset.read('wodger.scp')
    t2 =(time.time() - start)

    p =  (t2 - t1) * 100./ t1
    assert p<0


def test_generic_read_content():

    # Test bytes content reading
    datadir = prefs.datadir
    filename = pathclean(datadir) / 'wodger.spg'
    content = filename.read_bytes()

    # change the filename to be sure that the file will be read from the passed content
    filename = 'try.spg'

    # The most direct way to pass the byte content information
    nd = NDDataset.read(filename, content=content)
    assert str(nd) == 'NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    # It can also be passed using a dictionary structure {filename:content, ....}
    nd = NDDataset.read({filename:content})
    assert str(nd) == 'NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    # Case where the filename is not provided
    nd = NDDataset.read(content)
    assert str(nd) == 'NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    # Try with an .spa file
    filename = pathclean(datadir) / 'irdata/subdir/7_CZ0-100 Pd_101.SPA'
    content = filename.read_bytes()
    filename = 'try.spa'

    filename2 = pathclean(datadir) / 'irdata/subdir/7_CZ0-100 Pd_102.SPA'
    content2 = filename2.read_bytes()
    filename = 'try2.spa'

    nd = NDDataset.read({filename:content})
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:1, x:5549))'

    # Try with only a .spa content
    nd = NDDataset.read(content)
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:1, x:5549))'

    # Try with several .spa content (should be stacked into a single nddataset)
    nd = NDDataset.read({filename:content, filename2:content2})
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

    nd = NDDataset.read(content, content2)
    assert str(nd)=='NDDataset: [float32] a.u. (shape: (y:2, x:5549))'

