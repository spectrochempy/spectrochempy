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



import spectrochempy

from spectrochempy.api import *


def test_api():
    # test version
    from spectrochempy.__version__ import version
    assert version.split('.')[0] == '0'
    assert version.split('.')[1][:2] == '1a'
    # TODO: modify this for each release

    # test application

    print(('\n\nRunning : ', spectrochempy.api.running))
    assert version.startswith('0.1')
    assert "Laboratory for Catalysis and Spectrochempy" in copyright

    log.warning('Ok, this is nicely executing!')

    assert 'np' in APIref
    assert 'NDDataset' in APIref
    assert 'abs' in APIref
