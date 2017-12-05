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




""" Tests for the svd module

"""
from spectrochempy.api import *
from tests.utils import show_do_not_block

# test svd
#-----------

@show_do_not_block
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

    svd.Vt.plot()
    show()

#    svd.Vt[:10].plot()
#    show()


