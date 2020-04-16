# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import pytest
from spectrochempy import *
from spectrochempy.utils.testing import figures_dir, same_images

prefs = general_preferences

# @pytest.mark.skip
def test_1D():

    dataset = NDDataset.read_omnic(
        os.path.join(prefs.datadir, 'irdata', 'nh4y-activation.spg'))

    # get first spectrum
    nd0 = dataset[0]

    # plot generic
    ax = nd0.plot(output=os.path.join(figures_dir, 'IR_dataset_1D'),
                         savedpi=150)

    # plot generic style
    ax = nd0.plot(style='poster',
                         output=os.path.join(figures_dir, 'IR_dataset_1D_poster'),
                         savedpi=150)

    # check that style reinit to default
    ax = nd0.plot(output='IR_dataset_1D', savedpi=150)
    try:
        assert same_images('IR_dataset_1D.png',
                           os.path.join(figures_dir, 'IR_dataset_1D.png'))
    except:
        os.remove('IR_dataset_1D.png')
        raise AssertionError('comparison fails')
    os.remove('IR_dataset_1D.png')

    # try other type of plots
    ax = nd0.plot_pen()
    ax = nd0[:,::100].plot_scatter()
    ax = nd0.plot_lines()
    ax = nd0[:,::100].plot_bar()

    show()

    # multiple
    d = dataset[:,::100]
    datasets = [d[0], d[10], d[20], d[50], d[53]]
    labels = ['sample {}'.format(label) for label in
              ["S1", "S10", "S20", "S50", "S53"]]


    # plot multiple
    plot_multiple(method='scatter',
                  datasets=datasets, labels=labels, legend='best',
                  output=os.path.join(figures_dir,
                                      'multiple_IR_dataset_1D_scatter'),
                  savedpi=150)

    # plot mupltiple with style
    plot_multiple(method='scatter', style='sans',
                  datasets=datasets, labels=labels, legend='best',
                  output=os.path.join(figures_dir,
                                      'multiple_IR_dataset_1D_scatter_sans'),
                  savedpi=150)

    # check that style reinit to default
    plot_multiple(method='scatter',
                  datasets=datasets, labels=labels, legend='best',
                  output='multiple_IR_dataset_1D_scatter',
                  savedpi=150)
    try:
        assert same_images('multiple_IR_dataset_1D_scatter',
                           os.path.join(figures_dir,
                                        'multiple_IR_dataset_1D_scatter'))
    except:
        os.remove('multiple_IR_dataset_1D_scatter.png')
        raise AssertionError('comparison fails')
    os.remove('multiple_IR_dataset_1D_scatter.png')

    # plot 1D column
    col = dataset[:, 3500.]  # note the indexing using wavenumber!
    _ = col.plot_scatter()

    show()


# ======================================================================================================================
if __name__ == '__main__':
    pass
