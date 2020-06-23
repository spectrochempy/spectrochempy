# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
Tests for general issues

"""
from spectrochempy import PCA, read_omnic


def _test_issue_15():
    x = read_omnic('irdata/nh4y-activation.spg')
    my_pca = PCA(x)
    my_pca.reconstruct(n_pc=3)
