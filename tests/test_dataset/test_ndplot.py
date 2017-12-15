# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# =============================================================================



import matplotlib.pyplot as mpl

from tests.utils import show_do_not_block, image_comparison

from spectrochempy.api import *

options.log_level = INFO

# To regenerate the reference figures, set FORCE to True
FORCE = False
# for this regeneration it is advised to set non parallel testing.
# (remove option -nauto in pytest.ini)

#@image_comparison(reference=['IR_source_2D_stack', 'IR_source_2D_map',
#                             'IR_source_2D_image'], force_creation=FORCE)

@show_do_not_block
def test_plot_generic_2D(IR_source_2D):
    for method in ['stack', 'map', 'image']:
        source = IR_source_2D.copy()
        source.plot(method=method)