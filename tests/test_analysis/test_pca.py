# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================




""" Tests for the PCA module

"""
from spectrochempy.api import PCA, masked

# test pca
#---------

def test_pca(IR_source_2D):

    source = IR_source_2D.copy()

    # with masks
    source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing

    pca = PCA(source)
    print(pca)
    pca.printev(npc=5)

    assert str(pca)[:3] == '\nPC'

    pca.screeplot(npc=5)

    pca.scoreplot((1,2))

    pca.scoreplot(1,2, 3)