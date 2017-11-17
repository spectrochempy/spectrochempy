# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

import matplotlib.pyplot as mpl

from tests.utils import show_do_not_block, image_comparison

from spectrochempy.api import *

options.log_level = INFO

# To regenerate the reference figures, set FORCE to True
FORCE = True

@image_comparison(reference=['IR_source_2D_stack', 'IR_source_2D_map',
                             'IR_source_2D_image'], force_creation=FORCE)
def test_plot_generic_2D(IR_source_2D):
    for kind in ['stack', 'map', 'image']:
        source = IR_source_2D.copy()
        source.plot(kind=kind)


@image_comparison(reference=['IR_source_1D', 'IR_source_1D_sans'],
                  force_creation=FORCE)
def test_plot_generic_1D(IR_source_1D):
    source = IR_source_1D.copy()
    source.plot()
    assert mpl.rcParams['figure.figsize'] == [6.8, 4.4]
    source.plot(style='sans')
    assert mpl.rcParams['font.family'] == ['sans-serif']


@image_comparison(reference=['IR_source_2D_stack'])
def test_plot_stack(IR_source_2D):
    source = IR_source_2D.copy()
    source.plot_stack()  # plot_stack is an alias of plot(kind='stack')


@image_comparison(reference=['IR_source_2D_map'])
def test_plot_map(IR_source_2D):
    source = IR_source_2D.copy()
    source.plot_map()  # plot_map is an alias of plot(kind='map')


@image_comparison(reference=['IR_source_2D_image'])
def test_plot_image(IR_source_2D):
    source = IR_source_2D.copy()
    source.plot_image()  # plot_image is an alias of plot(kind='image')


@image_comparison(reference=['IR_source_2D_image'], min_similarity=98.0)
def test_plot_image_offset(IR_source_2D):
    source = IR_source_2D.copy() + .05
    source.plot_image()  # plot_image with offset


@image_comparison(reference=['IR_source_2D_map'])
def test_plot_stack_generic(IR_source_2D):
    source = IR_source_2D.copy()
    source.plot()  # generic plot default to map


@show_do_not_block
def test_plot_stack_masked(IR_source_2D):
    # just to see if masked area do not apppear on the figure
    source = IR_source_2D.copy() * 2.
    source[:, 1300.:900.] = masked
    source.plot_stack(colorbar=False)
    source.plot_map(colorbar=False)
    show()


@show_do_not_block
def test_plot_stack_multiple(IR_source_2D):
    source = IR_source_2D.copy()
    s1 = source[-10:]
    s2 = source[:10]
    row = s1[-1]
    row.plot()
    # two on the same plot
    s1.plot_stack()
    s2.plot_stack(data_only=True, hold=True)
    show()


# BUG FIXES IN PLOTS

@show_do_not_block
def test_successive_plot_bug_1a3_28(IR_source_2D):
    source = IR_source_2D.copy() * 2.
    source[:, 1300.:900.] = masked
    source.plot_stack(colorbar=False)
    source.plot()  # in 0.1a3.28 bug because key colorbar is missing.
    show()


@show_do_not_block
def test_successive_plot_bug_with_colorbars(IR_source_2D):
    source = IR_source_2D.copy() * 2.
    source[:, 1300.:900.] = masked
    source.plot_stack()
    source.plot()
    source.plot()  # bug colorbars stacked on the first plot
    source.plot(kind='map')  # bug : no colorbar
    show()
