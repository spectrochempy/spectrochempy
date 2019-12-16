# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================




from spectrochempy import *
import os
import pytest
from spectrochempy.utils.testing import assert_approx_equal


@pytest.fixture()
def script():

    return """
    
    #-----------------------------------------------------------
    # syntax for parameters definition :
    # name : value, low_bound,  high_bound
    #  * for fixed parameters
    #  $ for variable parameters
    #  > for reference to a parameter in the COMMON block
    #    (> is forbidden in the COMMON block)
    # common block parameters should not have a _ in their names
    #-----------------------------------------------------------
    #
    COMMON:
    # common parameters ex.
    # $ gwidth: 1.0, 0.0, none
    $ gratio: 0.1, 0.0, 1.0

    MODEL: LINE_1
    shape: assymvoigtmodel
        * ampl:  1.0, 0.0, none
        $ pos:   3620, 3400.0, 3700.0
        $ ratio: 0.0147, 0.0, 1.0
        $ assym: 0.1, 0, 1
        $ width: 200, 0, 1000

    MODEL: LINE_2
    shape: assymvoigtmodel
        $ ampl:  0.2, 0.0, none
        $ pos:   3520, 3400.0, 3700.0
        > ratio: gratio
        $ assym: 0.1, 0, 1
        $ width: 200, 0, 1000
    """


def test_fit_single_dataset(IR_dataset_2D, script):

    dataset = IR_dataset_2D[54, 3700.:3400.]

    f1 = Fit(dataset, script, silent=True)
    f1.run(maxiter=10, every=1)
#    assert_approx_equal(dataset.model_A, -116.40475, significant=4)
#    assert_approx_equal(f1.fp['width_line_2'], 195.7273, significant=4)
    dataset.plot(plot_model=True)

    dataset2 = dataset.copy() * 2.34
    f2 = Fit(dataset2, script, silent=True)
    f2.run(maxiter=10, every=1)

    dataset2.plot(plot_model=True)

    assert_approx_equal(dataset2.model_A, 116.40475 * 2.34, significant=4)
    assert_approx_equal(f2.fp['width_line_2'], 195.7273, significant=4)

    f2 = Fit(dataset2, script, silent=False)
    f2.run(maxiter=1000, every=1)

    dataset2.plot(plot_model=True)
    show()

def test_fit_multiple_dataset(IR_dataset_2D, script):
    dataset = IR_dataset_2D[54, 3700.:3400.]
    datasets = [dataset.copy(), dataset.copy() * 2.23456]
    f = Fit(datasets, script, silent=True)
    f.run(maxiter=10, every=1)
    assert_approx_equal(datasets[0].model_A, 116.404751, significant=4)
    assert_approx_equal(datasets[1].model_A, 116.404751 * 2.23456, significant=4)
    assert_approx_equal(f.fp['width_line_2'], 195.7273, significant=4)


    #TODO: plotting of multiple datasets
    #plotr(*datasets, showmodel=True, test=True)