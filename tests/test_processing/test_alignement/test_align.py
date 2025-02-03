# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the interpolate module"""

from spectrochempy.processing.alignement.align import align

# align
# -------


def test_align(ds1, ds2):
    ds1c = ds1.copy()
    dss = ds1c.align(ds2, dim="x")  # first syntax
    # TODO: flag copy=False raise an error
    assert dss is not None
    ds3, ds4 = dss  # a tuple is returned

    assert ds3.shape == (10, 100, 6)

    # TODO: labels are not aligned

    dss2 = align(ds1, ds2, dim="x")  # second syntax
    assert dss2 == dss
    assert dss2[0] == dss[0]
    ds5, ds6 = dss2

    # align another dim
    dss3 = align(ds1, ds2, dim="z")  # by default it would be the 'x' dim
    ds7, ds8 = dss3
    assert ds7.shape == (17, 100, 3)

    # align two dims
    dss4 = align(ds1, ds2, dims=["x", "z"])
    ds9, ds10 = dss4

    # align inner
    a, b = align(ds1, ds2, method="inner")
