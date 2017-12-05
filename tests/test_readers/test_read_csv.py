# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================




from spectrochempy.api import *

from tests.utils import assert_approx_equal, show_do_not_block
import pytest

@show_do_not_block
def test_read_csv():

    A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip',
                           directory=scpdata,
                           origin='omnic_export',
                           only=10)
    print(A)
    assert A.shape == (10,3736)

    A.plot_stack()

    B = NDDataset.read_csv('agirdata/A350/TGA/tg.csv',
                           directory=scpdata,
                           origin='tga')
    assert B.shape == (3446,)
    print(B)
    B.plot()
    show()