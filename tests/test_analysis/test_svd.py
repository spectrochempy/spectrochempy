# -*- coding: utf-8 -*-
# flake8: noqa


""" Tests for the SVD class

"""
from numpy.testing import assert_allclose

from spectrochempy.analysis.svd import SVD
from spectrochempy.utils import MASKED


# test svd
# -----------


def test_svd(IR_dataset_2D):
    dataset = IR_dataset_2D

    svd = SVD(dataset)

    assert_allclose(svd.ev_ratio[0].data, 94.539, rtol=1e-5, atol=0.0001)

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing
    dataset[10:12] = MASKED

    dataset.plot_stack()

    svd = SVD(dataset)

    assert_allclose(svd.ev_ratio.data[0], 93.8, rtol=1e-4, atol=0.01)

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing
    dataset[10:12] = MASKED

    dataset.plot_stack()

    svd = SVD(dataset, full_matrices=True)

    assert_allclose(svd.ev_ratio.data[0], 93.8, rtol=1e-4, atol=0.01)
