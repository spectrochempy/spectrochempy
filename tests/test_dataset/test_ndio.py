# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# =============================================================================





""" Tests for the ndplugin module

"""

from spectrochempy.scp import NDDataset, preferences
from spectrochempy.utils import SpectroChemPyWarning

import pytest
import os

from tests.utils import assert_array_equal


# Basic
# -------
def test_save_and_load(IR_source_2D):

    A = IR_source_2D.copy()
    A.save('tartempion.scp')
    # no directory for saving passed ... it must be in data
    path = os.path.join(preferences.datadir, 'tartempion.scp')
    assert os.path.exists(path)

    B = NDDataset.load('tartempion.scp')
    assert B.description == A.description
    assert_array_equal(A.data,B.data)
    os.remove(path)

    #B.save()

    #C=NDDataset.load()

def test_save(IR_source_2D):

    source = IR_source_2D.copy()
    source.save('essai')

    source.plot_stack()
    source.save('essai')  # there was a bug due to the saving of mpl axes

    os.remove(os.path.join(preferences.datadir, 'essai.scp'))
