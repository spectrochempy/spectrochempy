# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Integration contracts documented by the public units and masks guide."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pint import DimensionalityError as PintDimensionalityError
from pint import Unit as PintUnit

import spectrochempy as scp


def test_public_unit_objects_keep_identity_and_registry():
    assert scp.Unit is PintUnit
    assert scp.Quantity is scp.ur.Quantity
    assert scp.DimensionalityError is PintDimensionalityError
    assert scp.MaskedArray is np.ma.MaskedArray
    assert scp.MaskedConstant is type(np.ma.masked)

    quantity = scp.Quantity(2.5, "cm")
    assert quantity == 2.5 * scp.ur.cm
    assert quantity._REGISTRY is scp.ur
    assert scp.Unit("cm")._REGISTRY is scp.ur


def test_quantity_and_dataset_conversion_copy_and_inplace_contracts():
    quantity = 2500.0 * scp.ur.nm
    converted_quantity = quantity.to("um")
    assert converted_quantity == 2.5 * scp.ur.um
    assert converted_quantity is not quantity
    assert quantity == 2500.0 * scp.ur.nm

    mutable_quantity = quantity.copy()
    assert mutable_quantity.ito("um") is None
    assert mutable_quantity == 2.5 * scp.ur.um

    dataset = scp.NDDataset([1000.0, 2000.0], units="nm")
    converted_dataset = dataset.to("um")
    assert converted_dataset is not dataset
    assert_allclose(converted_dataset.data, [1.0, 2.0])
    assert_allclose(dataset.data, [1000.0, 2000.0])

    mutable_dataset = dataset.copy()
    assert mutable_dataset.ito("um") is None
    assert_allclose(mutable_dataset.data, [1.0, 2.0])


def test_public_dimensionality_error_and_spectroscopy_context():
    with pytest.raises(scp.DimensionalityError):
        (1.0 * scp.ur.meter).to("second")

    assert (1000.0 * scp.ur("cm^-1")).to("um") == 10.0 * scp.ur.um


def test_dataset_mask_and_nan_remain_distinct():
    coordinate = scp.Coord([1000.0, 1100.0, 1200.0], units="cm^-1")
    dataset = scp.NDDataset(
        [0.25, 2.00, 0.75],
        coordset=[coordinate],
        dims=["x"],
        units="absorbance",
    )
    dataset.meta.source = "synthetic"
    original_shape = dataset.shape

    assert scp.mean(dataset) == 1.0 * scp.ur.absorbance

    dataset[1] = scp.MASKED

    assert dataset.data.tolist() == [0.25, 2.00, 0.75]
    assert dataset.mask.tolist() == [False, True, False]
    assert dataset.x == coordinate
    assert dataset.meta.source == "synthetic"
    assert scp.mean(dataset) == 0.5 * scp.ur.absorbance

    assert dataset.remove_masks() is None
    assert dataset.mask is scp.NOMASK
    assert dataset.data.tolist() == [0.25, 2.00, 0.75]
    assert dataset.shape == original_shape
    assert dataset.x == coordinate
    assert dataset.meta.source == "synthetic"

    nan_dataset = scp.NDDataset([0.25, np.nan, 0.75], units="absorbance")
    assert nan_dataset.mask is scp.NOMASK
    assert np.isnan(scp.mean(nan_dataset).magnitude)


def test_masked_plot_keeps_a_real_gap():
    time = scp.Coord.linspace(0.0, 10.0, 101, title="time", units="s")
    original = scp.NDDataset(
        scp.sin(2.0 * time.data) + 0.25 * scp.cos(5.0 * time.data),
        coordset=[time],
        dims=["x"],
    )
    masked = original.copy()
    masked[40:61] = scp.MASKED

    ax = original.plot(color="0.65", show=False)
    _ = masked.plot(ax=ax, clear=False, color="tab:blue", show=False)

    original_line = ax.lines[0].get_ydata()
    masked_line = ax.lines[1].get_ydata()
    assert not np.ma.getmaskarray(original_line).any()
    assert np.ma.getmaskarray(masked_line)[40:61].all()
    assert not np.ma.getmaskarray(masked_line)[:40].any()
    assert not np.ma.getmaskarray(masked_line)[61:].any()


def test_raw_accessors_preserve_only_their_documented_information():
    dataset = scp.NDDataset([1.0, 2.0, 3.0], units="absorbance")
    dataset[1] = scp.MASKED

    assert type(dataset.data) is np.ndarray
    assert isinstance(dataset.masked_data, scp.MaskedArray)
    assert isinstance(dataset.values, scp.Quantity)
    assert isinstance(dataset.values.magnitude, scp.MaskedArray)
    assert isinstance(dataset.to_array(), scp.MaskedArray)
    assert isinstance(scp.MASKED, scp.MaskedConstant)
    assert scp.MASKED is np.ma.masked
    assert scp.NOMASK is np.ma.nomask
