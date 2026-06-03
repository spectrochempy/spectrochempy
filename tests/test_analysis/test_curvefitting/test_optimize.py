# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._models import asymmetricvoigtmodel


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


@pytest.fixture()
def synthetic_two_peak_dataset():
    x = scp.Coord(np.linspace(3700.0, 3400.0, 301), title="wavenumber", units="cm^-1")
    model = asymmetricvoigtmodel()
    y = (
        model.f(x.data, ampl=1.0, pos=3620.0, width=200.0, ratio=0.0147, asym=0.1)
        + model.f(x.data, ampl=0.2, pos=3520.0, width=200.0, ratio=0.1, asym=0.1)
        + 0.0002 * x.data
        - 0.5
    )
    return scp.NDDataset(
        y,
        coordset=[x],
        units="absorbance",
        title="synthetic optimize spectrum",
    )


def test_fit_single_dataset(synthetic_two_peak_dataset, script):
    dataset = synthetic_two_peak_dataset

    f1 = scp.Optimize()
    f1.script = script
    f1.autobase = True
    f1.max_iter = 10
    result = f1.fit(dataset)

    assert result is f1
    assert f1.n_components == 2
    assert f1.components.shape == (3, dataset.size)
    assert f1.predict().shape == (1, dataset.size)
    assert f1.transform().shape == (1, 2)

    residual = f1.predict().squeeze() - dataset
    assert abs(residual.data).max() < 1e-6
    assert_allclose(
        [
            f1.fp["pos_line_1"],
            f1.fp["pos_line_2"],
            f1.fp["width_line_1"],
            f1.fp["width_line_2"],
        ],
        [3620.0, 3520.0, 200.0, 200.0],
        rtol=0.02,
        atol=3.0,
    )
