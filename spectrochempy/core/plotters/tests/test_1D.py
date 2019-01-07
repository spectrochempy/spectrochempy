# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import pytest
from spectrochempy import *
from spectrochempy.utils.testing import figures_dir, same_images

#@pytest.mark.skip
def test_1D():

    set_loglevel(DEBUG)

    dataset = NDDataset.read_omnic(
            os.path.join(datadir.path, 'irdata', 'nh4y-activation.spg'))


    # plot generic
    ax = dataset[0].plot(output=os.path.join(figures_dir, 'IR_dataset_1D'),
                         savedpi=150)

    # plot generic style
    ax = dataset[0].plot(style='sans',
                        output=os.path.join(figures_dir, 'IR_dataset_1D_sans'),
                        savedpi=150)

    # check that style reinit to default
    ax = dataset[0].plot(output='IR_dataset_1D', savedpi=150)
    try:
        assert same_images('IR_dataset_1D.png',
                             os.path.join(figures_dir, 'IR_dataset_1D.png'))
    except:
        os.remove('IR_dataset_1D.png')
        raise AssertionError('comparison fails')
    os.remove('IR_dataset_1D.png')

    dataset = dataset[:,::100]

    datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
    labels = ['sample {}'.format(label) for label in
              ["S1", "S10", "S20", "S50", "S53"]]

    # plot multiple
    plot_multiple(method = 'scatter',
                  datasets=datasets, labels=labels, legend='best',
                  output=os.path.join(figures_dir,
                                       'multiple_IR_dataset_1D_scatter'),
                  savedpi=150)

    # plot mupltiple with  style
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

    show()




# =============================================================================
if __name__ == '__main__':
    pass
