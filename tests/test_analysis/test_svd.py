# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

""" Tests for the SVD class

"""
from spectrochempy import *
from numpy.testing import assert_allclose

# test svd
#-----------

def test_svd(IR_source_2D):

    source = IR_source_2D.copy()
    print(source)

    svd = SVD(source)

    print()
    print((svd.U))
    print((svd.VT))
    print((svd.s))
    print((svd.ev))
    print((svd.ev_cum))
    print((svd.ev_ratio))

    assert_allclose( svd.ev_ratio[0].data, 94.539, rtol=1e-5, atol=0.0001)

    #TODO: add round function to NDDataset


    # with masks
    source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing
    source[10:12] = masked

    ax = source.plot_stack()

    svd = SVD(source)

    print()
    print((svd.U))
    print((svd.VT))
    print((svd.s))
    print((svd.ev))
    print((svd.ev_cum))
    print((svd.ev_ratio))

    assert_allclose(svd.ev_ratio[0].data, 93.803, rtol=1e-5, atol=0.0001)

    # with masks
    source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing
    source[10:12] = masked

    ax = source.plot_stack()

    svd = SVD(source, full_matrices = True)

    print()
    print((svd.U))
    print((svd.VT))
    print((svd.s))
    print((svd.ev))
    print((svd.ev_cum))
    print((svd.ev_ratio))

    assert_allclose(svd.ev_ratio[0].data, 93.803, rtol=1e-5, atol=0.0001)