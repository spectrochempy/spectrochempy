# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

import pytest

from spectrochempy.core import preferences as prefs
from spectrochempy.core.units import ur
from spectrochempy.utils.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_raises,
)

DATADIR = prefs.datadir
NMRDATA = DATADIR / "nmrdata"

nmrdir = NMRDATA / "bruker" / "tests" / "nmr"


@pytest.mark.skipif(
    not NMRDATA.exists(),
    reason="Experimental data not available for testing",
)
def test_nmr_apodization_em(NMR_dataset_1D, NMR_dataset_2D):
    Hz = ur.Hz
    us = ur.us

    nd = NMR_dataset_1D.copy()
    nd /= nd.real.data.max()  # normalize

    lb = 0.0
    arr, apod = nd.em(lb=lb, retapod=True)
    # arr and dataset should be equal as no em was applied
    assert_array_equal(nd.data, arr.data)

    lb = 0.0
    gb = 0.0
    arr, apod = nd.gm(lb=lb, gb=gb, retapod=True)
    # arr and dataset should be equal as no em was applied
    assert_array_equal(nd.data, arr.data)

    lb = 100
    arr, apod = nd.em(lb=lb, retapod=True)

    # arr and dataset should not be equal as inplace=False
    assert_raises(AssertionError, assert_array_equal, nd.data, arr.data)
    assert_array_almost_equal(apod[1], 0.9987, decimal=4)

    # inplace=True
    arr = nd.em(lb=100.0 * Hz, inplace=True)
    assert_array_equal(nd.data, arr.data)

    nd = NMR_dataset_2D.copy()
    assert nd.shape == (96, 948)

    nd.em(lb=50.0 * Hz, dim=0)
    assert nd.shape == (96, 948)

    nd.em(lb=50.0 * Hz, dim="y")
    assert nd.shape == (96, 948)

    nd = NMR_dataset_2D.copy()
    nd.em(lb=50.0 * Hz, axis=-1)
    assert nd.shape == (96, 948)

    nd = NMR_dataset_2D.copy()
    nd.em(lb=50.0 * Hz, dim="x")
    assert nd.shape == (96, 948)

    nd = NMR_dataset_2D.copy()
    nd.em(lb=100 * Hz, shifted=10000 * us, retapod=True, inplace=False)
    assert nd.shape == (96, 948)

    # sp

    nd = NMR_dataset_1D.copy()

    new1, curve1 = nd.sinm(ssb=2, retapod=True, inplace=False)
    new2, curve2 = nd.sinm(ssb=1, retapod=True, inplace=False)
    new3, curve3 = nd.qsin(ssb=2, retapod=True, inplace=False)
    new4, curve4 = nd.qsin(ssb=1, retapod=True, inplace=False)
    new5, curve5 = nd.sinm(ssb=8, retapod=True, inplace=False)
