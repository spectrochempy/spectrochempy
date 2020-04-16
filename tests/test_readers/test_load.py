# -*- coding: utf-8 -*-
#
# ======================================================================================================================

# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================





from spectrochempy import NDDataset, general_preferences as prefs

from spectrochempy.utils.testing import assert_approx_equal
import os
import pytest

#datasets are defined in conftest as fixture


def test_load(IR_dataset_2D):

    dataset = IR_dataset_2D
    assert_approx_equal(dataset.data[0,0], 2.05, significant=2)
    B = dataset * 1.98
    assert_approx_equal(B.data[0, 0], 2.05 * 1.98, significant=2)
    assert "Binary operation mul with `1.98` has been performed" in B.history

    filename = os.path.join('irdata', 'nh4.scp')
    dataset.save(filename)
    dataset2 = NDDataset.read(filename)

    assert dataset == dataset2

def test_methods_read_access():

    path = os.path.join(prefs.datadir, 'nmrdata', 'bruker', 'tests',
                        'nmr','bruker_1d')

    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

    # alternatively
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
