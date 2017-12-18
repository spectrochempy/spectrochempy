# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================




from spectrochempy.api import *
import os
import pytest
from tests.utils import assert_approx_equal


@pytest.fixture()
def script():

    return """
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


def test_fit_single_source(IR_source_2D, script):

    source = IR_source_2D[54, 3700.:3400.]

    f1 = Fit(source, script, silent=True)
    f1.run(maxiter=10, every=1)
#    assert_approx_equal(source.model_A, -116.40475, significant=4)
#    assert_approx_equal(f1.fp['width_line_2'], 195.7273, significant=4)
    source.plot(plot_model=True)

    source2 = source.copy() * 2.34
    f2 = Fit(source2, script, silent=True)
    f2.run(maxiter=10, every=1)

    source2.plot(plot_model=True)

    assert_approx_equal(source2.model_A, -116.40475 * 2.34, significant=4)
    assert_approx_equal(f2.fp['width_line_2'], 195.7273, significant=4)

    f2 = Fit(source2, script, silent=False)
    f2.run(maxiter=1000, every=1)

    source2.plot(plot_model=True)
    show()

def test_fit_multiple_source(IR_source_2D, script):
    source = IR_source_2D[54, 3700.:3400.]
    sources = [source.copy(), source.copy() * 2.23456]
    f = Fit(sources, script, silent=True)
    f.run(maxiter=10, every=1)
    assert_approx_equal(sources[0].model_A, -116.404751, significant=4)
    assert_approx_equal(sources[1].model_A, -116.404751 * 2.23456, significant=4)
    assert_approx_equal(f.fp['width_line_2'], 195.7273, significant=4)


    #TODO: plotting of multiple sources
    #plotr(*sources, showmodel=True, test=True)