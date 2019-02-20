# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================


""" Tests for the interpolate module

"""
from spectrochempy.core.processors.interpolate import align

# align
#-------
def test_align(ds1, ds2):

    ds3 = ds2.align(ds1, dim=1, inplace=True)
    assert(ds3.shape == (9,100,4))
    assert(ds3 is ds2)


