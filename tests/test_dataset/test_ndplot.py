# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# =============================================================================

from spectrochempy.api import show, preferences, INFO

preferences.log_level = INFO

def test_plot_generic_1D(IR_source_1D):

    for method in ['scatter', 'lines']:
        source = IR_source_1D.copy()
        source.plot(method=method)

    show()


def test_plot_generic_2D(IR_source_2D):
    for method in ['stack', 'map', 'image']:
        source = IR_source_2D.copy()
        source.plot(method=method)

    show()

