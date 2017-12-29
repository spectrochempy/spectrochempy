# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================




from spectrochempy import *

from tests.utils import assert_approx_equal
import pytest


def test_read_csv():

    A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip',
                           origin='omnic_export',
                           only=10)
    print(A)
    assert A.shape == (10,3736)

    A.plot_stack()

    B = NDDataset.read_csv('agirdata/A350/TGA/tg.csv',
                           directory=preferences.datadir,
                           origin='tga')
    assert B.shape == (3446,)
    print(B)
    B.plot()
    show()