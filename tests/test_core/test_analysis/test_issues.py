# -*- coding: utf-8 -*-
# flake8: noqa


"""
Tests for general issues

"""
from spectrochempy import PCA, read_omnic


def _test_issue_15():
    x = read_omnic("irdata/nh4y-activation.spg")
    my_pca = PCA(x)
    my_pca.reconstruct(n_pc=3)
