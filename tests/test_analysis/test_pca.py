# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================




""" Tests for the PCA module

"""
from spectrochempy.api import Pca

# test pca
#---------

def test_pca(IR_source_2D):

    source = IR_source_2D.copy()
    pca = Pca(source)

    print(pca)
    pca.printev(npc=5)

    assert str(pca)[:3] == '\nPC'

    #TODO: check those : it seems that colors are not correct!
    #pca.screeplot(npc=5)

    #pca.scoreplot(pcs=(0,1))
