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




from spectrochempy.api import NDDataset, scpdata, log, DEBUG, ERROR

from tests.utils import assert_approx_equal
import os
import pytest

#sources are defined in conftest as fixture

def test_load(IR_source_2D):

    source = IR_source_2D
    assert_approx_equal(source.data[0,0], 2.05, significant=2)
    B = source * 1.98
    assert_approx_equal(B.data[0, 0], 2.05 * 1.98, significant=2)
    assert "binary operation mul with `1.98` has been performed" in B.history

    filename = os.path.join(scpdata, 'irdata', 'nh4.scp')
    source.save(filename)
    source2 = NDDataset.read(filename)

    assert source == source2

def test_methods_read_access():

    path = os.path.join(scpdata, 'nmrdata', 'bruker', 'tests', 'nmr',
                        'bruker_1d')

    # load the data in a new dataset
    ndd = NDDataset()
    ndd.read_bruker_nmr(path, expno=1, remove_digital_filter=True)

    # alternatively
    ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
