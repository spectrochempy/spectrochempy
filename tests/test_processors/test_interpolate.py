# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B) 
# See full LICENSE agreement in the root directory
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


