# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the interpolate module"""

import importlib
import sys

import pytest

import spectrochempy as scp
from spectrochempy.processing.alignment.align import align
from spectrochempy.utils.exceptions import UnitsCompatibilityError

# align
# -------


def test_legacy_align_import_is_deprecated():
    sys.modules.pop("spectrochempy.processing.alignement.align", None)
    with pytest.deprecated_call(match="processing.alignement.align"):
        legacy_module = importlib.import_module(
            "spectrochempy.processing.alignement.align"
        )
    assert legacy_module.align is align


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

    dss_root = scp.align(ds1, ds2, dim="x")
    assert dss_root == dss

    # align another dim
    dss3 = align(ds1, ds2, dim="z")  # by default it would be the 'x' dim
    ds7, ds8 = dss3
    assert ds7.shape == (17, 100, 3)

    # align two dims
    dss4 = align(ds1, ds2, dims=["x", "z"])
    ds9, ds10 = dss4

    # align inner
    a, b = align(ds1, ds2, method="inner")


def test_align_rejects_incompatible_coord_units_with_clear_message():
    ds1 = scp.NDDataset([1.0, 2.0], coordset=[scp.Coord([100.0, 200.0], units="cm^-1")])
    ds2 = scp.NDDataset([3.0, 4.0], coordset=[scp.Coord([1.0, 2.0], units="s")])

    with pytest.raises(UnitsCompatibilityError) as exc:
        align(ds1, ds2, dim="x")
    message = str(exc.value)
    assert "Cannot align datasets" in message
    assert "dimension 'x'" in message
    assert "s" in message
    assert "cm" in message
    assert "Convert the coordinates to compatible units before retrying." in message
