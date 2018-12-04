# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# =============================================================================

from spectrochempy.utils import show
from spectrochempy.application import set_loglevel, INFO

set_loglevel(INFO)

def test_plot_generic_1D(IR_dataset_1D):

    for method in ['scatter', 'lines']:
        dataset = IR_dataset_1D.copy()
        dataset.plot(method=method)

    show()


def test_plot_generic_2D(IR_dataset_2D):
    for method in ['stack', 'map', 'image']:
        dataset = IR_dataset_2D.copy()
        dataset.plot(method=method)

    show()

