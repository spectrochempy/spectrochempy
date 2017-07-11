# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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


from spectrochempy.fitting import Fit

from spectrochempy.api import NDDataset
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

def test_fit_single_source(IR_source_1, script):

    source = IR_source_1[54, 3700.:3400.]

    f1 = Fit(source, script, silent=True)
    f1.run(maxiter=10, every=1)
    assert_approx_equal(source.model_A, -50.6219, significant=4)
    assert_approx_equal(f1.fp['width_line_2'], 192.6362, significant=4)

    source2 = source.copy() * 2.34
    f2 = Fit(source2, script, silent=True)
    f2.run(maxiter=10, every=1)
    assert_approx_equal(source2.model_A, -50.6219 * 2.34, significant=4)
    assert_approx_equal(f2.fp['width_line_2'], 192.6362, significant=4)

    source2.plot(showmodel=True)

def test_fit_multiple_source(IR_source_1, script):
    source = IR_source_1[54, 3700.:3400.]
    sources = [source.copy(), source.copy() * 2.23456]
    f = Fit(sources, script, silent=True)
    f.run(maxiter=10, every=1)
    assert_approx_equal(sources[0].model_A, -50.6219, significant=4)
    assert_approx_equal(sources[1].model_A, -50.6219 * 2.23456, significant=4)
    assert_approx_equal(f.fp['width_line_2'], 192.6362, significant=4)


    #TODO: plotting of multiple sources
    #plotr(*sources, showmodel=True, test=True)