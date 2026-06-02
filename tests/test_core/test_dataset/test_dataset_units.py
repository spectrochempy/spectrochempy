# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest
from pint.errors import UndefinedUnitError

import spectrochempy as scp
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import assert_array_almost_equal


def test_nddataset_invalid_units():
    with pytest.raises(UndefinedUnitError):
        scp.NDDataset(np.ones((5, 5)), units="NotAValidUnit")


def test_nddataset_units(nd1d):
    nd = nd1d.copy()
    nd = np.fabs(nd)
    nd.units = "m"
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m**0.5
    nd.units = "cm"
    nd2 = np.sqrt(nd)
    nd.ito("m")
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m**0.5


def test_nddataset_bugs_units_change():
    # check for bug on transmittance conversion
    X = scp.NDDataset([0.0, 0.3, 1.3, 5.0], units="absorbance")

    # A to T
    X1 = X.to("transmittance")
    assert_array_almost_equal(X1.data, 10 ** -np.array([0.0, 0.3, 1.3, 5.0]) * 100)
    assert X1.title == "transmittance"
    # T to abs T
    X2 = X1.to("absolute_transmittance")
    assert_array_almost_equal(X2.data, 10 ** -np.array([0.0, 0.3, 1.3, 5.0]))
    assert X2.title == "transmittance"
    # A to abs T
    X2b = X.to("absolute_transmittance")
    assert_array_almost_equal(X2b, X2)
    assert X2b.title == "transmittance"
    # abs T to T
    X3 = X2.to("transmittance")
    assert_array_almost_equal(X3, X1)
    assert X3.title == "transmittance"
    # T to A
    X4 = X3.to("absorbance")
    assert_array_almost_equal(X4, X)
    assert X4.title == "absorbance"
    # abs T to A
    X5 = X2.to("absorbance")
    assert_array_almost_equal(X5, X)
    assert X5.title == "absorbance"


if __name__ == "__main__":
    pytest.main([__file__])
