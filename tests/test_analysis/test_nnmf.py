# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import numpy as np
import spectrochempy as scp
from spectrochempy.utils.testing import assert_array_almost_equal

# scp.set_loglevel(scp.DEBUG)


def test_NNMF():

    w1 = scp.NDDataset([[1, 2, 3], [4, 5, 6]])
    h1 = scp.NDDataset([[1, 2], [3, 4], [5, 6]])

    w2 = scp.NDDataset([[1, 1, 1], [4, 4, 4]])
    h2 = scp.NDDataset([[1, 1], [3, 3], [5, 5]])

    v = scp.dot(w1, h1)

    nnmf = scp.NNMF(v, w2, h2, tol=0.0001, maxtime=60, maxiter=100, verbose=True)

    scp.info_("------")
    scp.info_(nnmf.C)
    scp.info_("------")
    scp.info_(nnmf.St)

    assert_array_almost_equal(
        nnmf.C.data, np.array([[1.4, 2.1, 2.9], [4.4, 5.2, 6.0]]), decimal=1
    )
    assert_array_almost_equal(
        nnmf.St.data, [[0.8, 1.9], [2.9, 3.9], [5.1, 5.9]], decimal=1
    )
