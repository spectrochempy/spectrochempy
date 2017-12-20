# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================




""" Tests for the svd module

"""
from spectrochempy.api import *

# test svd
#-----------

def test_svd(IR_source_2D):

    source = IR_source_2D.copy()
    print(source)

    svd = Svd(source)

    print()
    print((svd.U))
    print((svd.Vt))
    print((svd.s))
    print((svd.ev))
    print((svd.ev_cum))
    print((svd.ev_ratio))

    assert np.around(svd.ev_ratio[0].data, decimals=3) == 94.539

    #TODO: add round function to NDDataset


    # with masks
    source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing
    ax = source.plot_stack()

    svd = Svd(source)

    print()
    print((svd.U))
    print((svd.Vt))
    print((svd.s))
    print((svd.ev))
    print((svd.ev_cum))
    print((svd.ev_ratio))

    assert np.around(svd.ev_ratio[0].data, decimals=3) == 93.887
