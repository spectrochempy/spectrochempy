# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================


""" Tests for the interpolate module

"""
from spectrochempy.core import info_
from spectrochempy.core.processors.align import align

# align
#-------

def test_align(ds1, ds2):
    info_()
    info_('---')
    info_(ds1)     # (z:10, y:100, x:3)
    info_('---')
    info_(ds2)     # ( z:9,  y:50, x:4)
    
    ds1c = ds1.copy()
    dss = ds1c.align(ds2, dim='x')   # first syntax
    #TODO: flag copy=False raise an error
    assert dss is not None
    ds3, ds4 = dss  # a tuple is returned
    
    assert(ds3.shape == (10,100,6))
    
    info_()
    info_('---')
    info_(ds3)     # (z:10, y:100, x:6)
    info_('---')
    info_(ds4)     # ( z:9,  y:50, x:6)
    #TODO: labels are not aligned

    dss2 = align(ds1, ds2, dim='x')   # second syntax
    assert dss2 == dss
    assert dss2[0] == dss[0]
    ds5, ds6 = dss2
    info_()
    info_('---')
    info_(ds5)     # (z:10, y:100, x:6)
    info_('---')
    info_(ds6)     # ( z:9,  y:50, x:6)
    
    # align another dim
    dss3 = align(ds1, ds2, dim='z') # by default it would be the 'x' dim
    ds7, ds8 = dss3
    assert ds7.shape == (17, 100, 3)
    info_()
    info_('---')
    info_(ds7)     # (z:17, y:100, x:3)
    info_('---')
    info_(ds8)     # ( z:17,  y:50, x:4)
    
    # align two dims
    dss4 = align(ds1, ds2, dims=['x','z'])
    ds9, ds10 = dss4
    info_()
    info_('---')
    info_(ds9)     # (z:17, y:100, x:6)
    info_('---')
    info_(ds10)     # ( z:17,  y:50, x:6)
    
    # align inner
    a, b = align(ds1, ds2, method='inner')
    info_()
    info_('---')
    info_(a)
    info_('---')
    info_(b)