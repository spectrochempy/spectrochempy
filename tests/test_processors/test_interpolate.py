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




""" Tests for the interpolate module

"""
from spectrochempy.api import align

# align
#-------
def test_align(ds1, ds2):

    ds3 = ds2.align(ds1, axis=1, inplace=True)
    assert(ds3.shape == (9,100,4))
    assert(ds3 is ds2)


