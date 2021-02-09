# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

""" Tests integration

"""

from spectrochempy.core import info_


def test_integrate(IR_dataset_2D):

    dataset = IR_dataset_2D[:,1250.:1800.]
    info_(dataset)

    # default dim='x', compare trapz and simps
    area_trap = dataset.trapz()
    area_simp = dataset.simps()

    diff = area_trap - area_simp
    assert(diff.shape == (55,))
    assert((diff/area_trap).max() < 1e-4)

    area_trap_x = dataset.trapz(dim='x')
    diff = area_trap - area_trap_x
    assert(diff.max() == 0.0)

    area_trap_y = dataset.trapz(dim='y')
    assert (area_trap_y.shape== (572,))