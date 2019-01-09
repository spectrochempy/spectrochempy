# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# =============================================================================


""" Tests for the ndplugin module

"""

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.application import datadir
import os

from spectrochempy.utils.testing import assert_array_equal


# Basic
# -------
def test_basic():
    ir = NDDataset([1.1, 2.2, 3.3], coordset=[[1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)

    ir = NDDataset([[1.1, 2.2, 3.3]], coordset=[[1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)

    ir = NDDataset([[1.1, 2.2, 3.3], [1.1, 2.2, 3.3]], coordset=[[1, 2], [1, 2, 3]])
    ir.save('essai')
    dl = NDDataset.load('essai')
    assert_array_equal(dl.data, ir.data)


def test_save(IR_dataset_2D):
    dataset = IR_dataset_2D.copy()
    dataset.save('essai')

    dataset.plot_stack()
    dataset.save('essai')  # there was a bug due to the saving of mpl axes

    os.remove(os.path.join(datadir.path, 'essai.scp'))


def test_save_and_load_mydataset(IR_dataset_2D):
    ds = IR_dataset_2D.copy()
    ds.save('mydataset')
    dl = NDDataset.load('mydataset')
    assert_array_equal(dl.data, ds.data)
    assert_array_equal(dl.x.data, ds.x.data)
    assert (dl == ds)
    assert (dl.meta == ds.meta)
    assert (dl.plotmeta == ds.plotmeta)
