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