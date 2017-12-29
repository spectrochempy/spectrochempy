# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

import pytest
from spectrochempy import *
from tests.utils import figures_dir, same_images

@pytest.mark.skip
def test_1D():

    preferences.log_level=DEBUG


    source = NDDataset.read_omnic(
            os.path.join(datadir.path, 'irdata', 'NH4Y-activation.SPG'))


    # plot generic
    ax = source[0].plot(output=os.path.join(figures_dir, 'IR_source_1D'),
                         savedpi=150)

    # plot generic style
    ax = source[0].plot(style='sans',
                        output=os.path.join(figures_dir, 'IR_source_1D_sans'),
                        savedpi=150)

    # check that style reinit to default
    ax = source[0].plot(output='IR_source_1D', savedpi=150)
    try:
        assert same_images('IR_source_1D.png',
                             os.path.join(figures_dir, 'IR_source_1D.png'))
    except:
        os.remove('IR_source_1D.png')
        raise AssertionError('comparison fails')
    os.remove('IR_source_1D.png')

    source = source[:,::100]

    sources = [source[0], source[10], source[20], source[50], source[53]]
    labels = ['sample {}'.format(label) for label in
              ["S1", "S10", "S20", "S50", "S53"]]

    # plot multiple
    plot_multiple(method = 'scatter',
                  sources=sources, labels=labels, legend='best',
                  output=os.path.join(figures_dir,
                                       'multiple_IR_source_1D_scatter'),
                  savedpi=150)

    # plot mupltiple with  style
    plot_multiple(method='scatter', style='sans',
                  sources=sources, labels=labels, legend='best',
                  output=os.path.join(figures_dir,
                                       'multiple_IR_source_1D_scatter_sans'),
                  savedpi=150)

    # check that style reinit to default
    plot_multiple(method='scatter',
                  sources=sources, labels=labels, legend='best',
                  output='multiple_IR_source_1D_scatter',
                  savedpi=150)
    try:
        assert same_images('multiple_IR_source_1D_scatter',
                             os.path.join(figures_dir,
                                          'multiple_IR_source_1D_scatter'))
    except:
        os.remove('multiple_IR_source_1D_scatter.png')
        raise AssertionError('comparison fails')
    os.remove('multiple_IR_source_1D_scatter.png')

    plt.show()








# =============================================================================
if __name__ == '__main__':
    pass
