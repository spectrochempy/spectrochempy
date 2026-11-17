# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Tests for the interpolate module"""

import importlib
import sys

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.processing.alignment.align import align
from spectrochempy.utils.exceptions import UnitsCompatibilityError

# align
# -------


def test_legacy_align_import_is_deprecated():
    sys.modules.pop("spectrochempy.processing.alignement.align", None)
    with pytest.deprecated_call(match="processing.alignement.align") as record:
        legacy_module = importlib.import_module(
            "spectrochempy.processing.alignement.align"
        )
    assert legacy_module.align is align
    messages = [str(item.message) for item in record]
    assert any(
        "spectrochempy.processing.alignment.align" in message for message in messages
    )
    assert all("0.13.0" not in message for message in messages)
    assert any("deprecation policy is satisfied" in message for message in messages)


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


@pytest.mark.parametrize("keep_row", [False, True])
@pytest.mark.parametrize("second_labels", [None, ["q", "r", "s"]])
def test_align_inner_spectra(keep_row, second_labels):
    first_x = scp.Coord([1.0, 2.0, 3.0], labels=["a", "b", "c"])
    second_x = scp.Coord([2.0, 3.0, 4.0], labels=second_labels)
    first_values = np.array([10.0, 20.0, 30.0])
    second_values = np.array([2.0, 3.0, 4.0])
    if keep_row:
        first_values = first_values[None, :]
        second_values = second_values[None, :]
    first = scp.NDDataset(first_values)
    second = scp.NDDataset(second_values)
    first.x = first_x
    second.x = second_x

    aligned_first, aligned_second = scp.align(first, second, method="inner")

    np.testing.assert_array_equal(aligned_first.x.data, [2.0, 3.0])
    np.testing.assert_array_equal(aligned_second.x.data, [2.0, 3.0])
    np.testing.assert_array_equal(aligned_first.x.labels, ["b", "c"])
    if second_labels is None:
        assert aligned_second.x.labels is None
    else:
        np.testing.assert_array_equal(aligned_second.x.labels, ["q", "r"])
    np.testing.assert_array_equal(
        (aligned_first - aligned_second).data.ravel(), [18.0, 27.0]
    )


def test_align_inner_rejects_disjoint_coordinates():
    first = scp.NDDataset([10.0, 20.0], coordset=[scp.Coord([1.0, 2.0])])
    second = scp.NDDataset([30.0, 40.0], coordset=[scp.Coord([3.0, 4.0])])

    with pytest.raises(ValueError, match="No shared coordinates along dimension 'x'"):
        scp.align(first, second, method="inner")


def test_align_inner_preserves_each_spectrum_mask():
    first = scp.NDDataset(
        [10.0, 20.0, 30.0],
        mask=[True, False, True],
        coordset=[scp.Coord([1.0, 2.0, 3.0])],
    )
    second = scp.NDDataset(
        [2.0, 3.0, 4.0],
        mask=[True, False, False],
        coordset=[scp.Coord([2.0, 3.0, 4.0])],
    )

    aligned_first, aligned_second = scp.align(first, second, method="inner")

    np.testing.assert_array_equal(aligned_first.data, [20.0, 30.0])
    np.testing.assert_array_equal(aligned_second.data, [2.0, 3.0])
    np.testing.assert_array_equal(aligned_first.mask, [False, True])
    np.testing.assert_array_equal(aligned_second.mask, [True, False])


def test_align_inner_preserves_complex_values():
    first = scp.NDDataset(
        [10.0 + 1.0j, 20.0 + 2.0j, 30.0 + 3.0j],
        coordset=[scp.Coord([1.0, 2.0, 3.0])],
    )
    second = scp.NDDataset([2.0, 3.0, 4.0], coordset=[scp.Coord([2.0, 3.0, 4.0])])

    aligned_first, aligned_second = scp.align(first, second, method="inner")

    np.testing.assert_array_equal(aligned_first.data, [20.0 + 2.0j, 30.0 + 3.0j])
    np.testing.assert_array_equal(aligned_second.data, [2.0, 3.0])


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
