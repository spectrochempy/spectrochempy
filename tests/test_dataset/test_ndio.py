# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# ======================================================================================================================


""" Tests for the ndplugin module

"""

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core import general_preferences as prefs

import os
import pytest

from spectrochempy.core import info_, debug_
from spectrochempy.utils.testing import assert_array_equal


# Basic
# ----------------------------------------------------------------------------------------------------------------------
def test_ndio_basic():
    ir = NDDataset([1.1, 2.2, 3.3], coords=[[1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)

    ir = NDDataset([[1.1, 2.2, 3.3]], coords=[[0], [1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)

    ir = NDDataset([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]], coords=[[1, 2], [1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)

def test_ndio_less_basic(coord2, coord2b, dsm):  # dsm is defined in conftest
    
    coordm = CoordSet(coord2, coord2b)
    
    # for multiple coordinates
    assert dsm.coords['x'] == coordm

    info_(dsm)
    
    dsm.save('essai')
    da = NDDataset.load('essai')
    
    info_(da)
    
    assert da == dsm

def test_ndio_save1D_load(IR_dataset_1D):
    dataset = IR_dataset_1D.copy()
    #debug_(dataset)
    dataset.save('essai')
    ir = NDDataset.load("essai")
    #debug_(ir)
    os.remove(os.path.join(prefs.datadir, 'essai.scp'))

def test_ndio_save2D_load(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    #debug_(dataset)
    dataset.save('essai')
    ir = dataset.load("essai")
    #debug_(ir)
    os.remove(os.path.join(prefs.datadir, 'essai.scp'))

def test_ndio_save_and_load_mydataset(IR_dataset_2D):
    ds = IR_dataset_2D.copy()
    ds.save('mydataset')
    dl = NDDataset.load('mydataset')
    assert_array_equal(dl.data, ds.data)
    assert_array_equal(dl.x.data, ds.x.data)
    assert (dl == ds)
    assert (dl.meta == ds.meta)
    assert (dl.plotmeta == ds.plotmeta)

