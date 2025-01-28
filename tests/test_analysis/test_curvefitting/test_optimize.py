# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest

from spectrochempy import Optimize
from spectrochempy.utils.plots import show
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
    shape: asymmetricvoigtmodel
        * ampl:  1.0, 0.0, none
        $ pos:   3620, 3400.0, 3700.0
        $ ratio: 0.0147, 0.0, 1.0
        $ asym: 0.1, 0, 1
        $ width: 200, 0, 1000

    MODEL: LINE_2
    shape: asymmetricvoigtmodel
        $ ampl:  0.2, 0.0, none
        $ pos:   3520, 3400.0, 3700.0
        > ratio: gratio
        $ asym: 0.1, 0, 1
        $ width: 200, 0, 1000
    """


def test_fit_single_dataset(IR_dataset_2D, script):
    dataset = IR_dataset_2D[54, 3700.0:3400.0]

    f1 = Optimize()
    f1.script = script
    f1.autobase = True
    f1.max_iter = 10
    f1.fit(dataset)
