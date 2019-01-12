# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================




from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.application import  general_preferences as prefs

from spectrochempy.utils.testing import assert_approx_equal
import os
import pytest


def test_nmr_reader_1D():
    path = os.path.join(prefs.datadir, 'nmrdata','bruker', 'tests',
                        'nmr','bruker_1d')

    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    print(ndd.__str__())
    print()
    print(ndd._repr_html_())

def test_nmr_reader_2D():
    path = os.path.join(prefs.datadir, 'nmrdata','bruker', 'tests',
                        'nmr','bruker_2d')

    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    print(ndd.__str__())
    print()
    print(ndd._repr_html_())