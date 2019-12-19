# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

""" Tests for the SVD class

"""
from spectrochempy.core.analysis.svd import SVD
from spectrochempy.utils import MASKED
from spectrochempy.core import info_
from numpy.testing import assert_allclose

# test svd
#-----------

def test_svd(IR_dataset_2D):

    dataset = IR_dataset_2D.copy()
    info_(dataset)

    svd = SVD(dataset)

    info_()
    info_((svd.U))
    info_((svd.VT))
    info_((svd.s))
    info_((svd.ev))
    info_((svd.ev_cum))
    info_((svd.ev_ratio))

    assert_allclose( svd.ev_ratio[0].data, 94.539, rtol=1e-5, atol=0.0001)

    #TODO: add round function to NDDataset


    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing
    dataset[10:12] = MASKED

    ax = dataset.plot_stack()

    svd = SVD(dataset)

    info_()
    info_((svd.U))
    info_((svd.VT))
    info_((svd.s))
    info_((svd.ev))
    info_((svd.ev_cum))
    info_((svd.ev_ratio))

    assert_allclose(svd.ev_ratio.data[0], 93.803, rtol=1e-5, atol=0.001)

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing
    dataset[10:12] = MASKED

    ax = dataset.plot_stack()

    svd = SVD(dataset, full_matrices = True)

    info_()
    info_((svd.U))
    info_((svd.VT))
    info_((svd.s))
    info_((svd.ev))
    info_((svd.ev_cum))
    info_((svd.ev_ratio))

    assert_allclose(svd.ev_ratio.data[0], 93.803, rtol=1e-5, atol=0.001)