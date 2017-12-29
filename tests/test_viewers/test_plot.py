# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================



import matplotlib.pyplot as mpl

from tests.utils import  image_comparison

from spectrochempy import *

preferences.log_level = INFO

# To regenerate the reference figures, set FORCE to True
FORCE = True

# for this regeneration it is advised to set non parallel testing.
# (remove option -nauto in pytest.ini)

@image_comparison(reference=['IR_source_2D_stack', 'IR_source_2D_map',
                             'IR_source_2D_image'], force_creation=FORCE)
def test_plot_generic_2D(IR_source_2D):
    for method in ['stack', 'map', 'image']:
        source = IR_source_2D.copy()
        source.plot(method=method)


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
    source.plot_stack()  # plot_stack is an alias of plot(method='stack')


@image_comparison(reference=['IR_source_2D_map'])
def test_plot_map(IR_source_2D):
    source = IR_source_2D.copy()
    source.plot_map()  # plot_map is an alias of plot(method='map')


@image_comparison(reference=['IR_source_2D_image',
                             'IR_source_2D_image_sanspaper'],
                             force_creation=FORCE)
def test_plot_image(IR_source_2D):
    source = IR_source_2D.copy()
    source.plot_image()  # plot_image is an alias of plot(method='image')
    source.plot_image(style=['sans', 'paper'], fontsize=9)

@image_comparison(reference=['IR_source_2D_image',
                             'IR_source_2D_image_sanspaper'],
                  min_similarity=85.0)
def test_plot_image_offset(IR_source_2D):
    source = IR_source_2D.copy() + .0001
    source.plot_image()  # plot_image with offset
    source.plot_image(style=['sans','paper'])


@image_comparison(reference=['IR_source_2D_map'])
def test_plot_stack_generic(IR_source_2D):
    source = IR_source_2D.copy()
    source.plot()  # generic plot default to map



def test_plot_stack_masked(IR_source_2D):
    # just to see if masked area do not apppear on the figure
    source = IR_source_2D.copy() * 2.
    source[:, 1300.:900.] = masked
    source.plot_stack(colorbar=False)
    source.plot_map(colorbar=False)


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


def test_successive_plot_bug_1a3_28(IR_source_2D):
    source = IR_source_2D.copy() * 2.
    source[:, 1300.:900.] = masked
    source.plot_stack(colorbar=False)
    source.plot()  # in 0.1a3.28 bug because key colorbar is missing.
    show()


def test_successive_plot_bug_with_colorbars(IR_source_2D):
    source = IR_source_2D.copy() * 2.
    source[:, 1300.:900.] = masked
    source.plot_stack()
    source.plot()
    source.plot()  # bug colorbars stacked on the first plot
    source.plot(method='map')  # bug : no colorbar
    show()

@image_comparison(reference=['multiplot1','multiplot2'], force_creation=FORCE)
def test_multiplot(IR_source_2D):

    source = IR_source_2D.copy()

    sources=[source, source*1.1, source*1.2, source*1.3]
    labels = ['sample {}'.format(label) for label in
              ["1", "2", "3", "4"]]

    multiplot(sources=sources, method='stack', labels=labels, nrow=2, ncol=2,
                    figsize=(9, 5), sharex=True, sharey=True)

    multiplot(sources=sources, method='map', labels=labels, nrow=2, ncol=2,
                    figsize=(9, 5), sharex=True, sharey=True)

@image_comparison(reference=['IR_source_1D',
                             'IR_source_1D_sans',
                             'IR_source_1D',
                             'multiple_IR_source_1D_scatter',
                             'multiple_IR_source_1D_scatter_sans',
                             'multiple_IR_source_1D_scatter',
                             ], force_creation=FORCE)
def tests_multipleplots_and_styles():
    source = NDDataset.read_omnic(
            os.path.join('irdata', 'NH4Y-activation.SPG'))


    # plot generic
    ax = source[0].copy().plot()

    # plot generic style
    ax = source[0].copy().plot(style='sans')

    # check that style reinit to default
    ax = source[0].copy().plot()

    source = source[:,::100]

    sources = [source[0], source[10], source[20], source[50], source[53]]
    labels = ['sample {}'.format(label) for label in
              ["S1", "S10", "S20", "S50", "S53"]]

    # plot multiple
    plot_multiple(method = 'scatter',
                  sources=sources, labels=labels, legend='best')

    # plot mupltiple with  style
    plot_multiple(method='scatter', style='sans',
                  sources=sources, labels=labels, legend='best')

    # check that style reinit to default
    plot_multiple(method='scatter',
                  sources=sources, labels=labels, legend='best')

##### debugging ####

#### deprecation #

def test_kind_deprecated(IR_source_2D):

    source = IR_source_2D.copy()

    # should raise a deprecation warning
    source.plot(kind='stack', style='sans', colorbar=False)
