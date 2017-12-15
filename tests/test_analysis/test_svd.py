# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version B)
# See full LICENSE agreement in the root directory
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


