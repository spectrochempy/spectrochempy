# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset


def _quadratic_dataset():
    x = np.linspace(-2.0, 2.0, 9)
    residuals = np.vstack((np.sin(2.0 * x), np.cos(1.5 * x)))
    trends = np.vstack((1.0 + 2.0 * x + 0.5 * x**2, -3.0 + x - x**2))
    dataset = NDDataset(
        trends + residuals,
        coordset=[
            Coord([0.0, 1.0], title="sample"),
            Coord(x, title="wavenumber", units="cm^-1"),
        ],
        units="absorbance",
        title="synthetic signal",
    )
    mask = np.zeros(dataset.shape, dtype=bool)
    mask[0, 4] = True
    dataset.mask = mask
    dataset.annotate("Synthetic detrend source")
    return dataset


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "constant"},
        {"dim": "y"},
        {"inplace": True},
        {"model": "rubberband"},
        {"title": "detrended data"},
        {"unexpected_option": True},
    ],
)
def test_detrend_rejects_unsupported_options_before_mutating_source(kwargs):
    source = _quadratic_dataset()
    original_data = source.data.copy()
    original_mask = source.mask.copy()
    original_x = source.x.copy()
    original_history = source.history_entries

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        source.detrend(**kwargs)

    np.testing.assert_array_equal(source.data, original_data)
    np.testing.assert_array_equal(source.mask, original_mask)
    assert source.x == original_x
    assert source.history_entries == original_history


def test_detrend_polynomial_order_matches_independent_numpy_fit():
    source = _quadratic_dataset()
    source.mask = False
    original = source.copy()
    x = source.x.data
    expected = np.vstack(
        [row - np.polyval(np.polyfit(x, row, 2), x) for row in source.data]
    )

    result = source.detrend(order=2)

    np.testing.assert_allclose(result.data, expected, atol=2.0e-15)
    assert result.shape == source.shape
    assert result.x == source.x
    assert result.units == source.units
    np.testing.assert_array_equal(source.data, original.data)
    assert source.history_entries == original.history_entries


def test_detrend_breakpoint_fits_piecewise_linear_trends():
    x = np.arange(10.0)
    data = np.where(x <= 4.0, 1.0 + 2.0 * x, 20.0 - 3.0 * x)
    source = NDDataset(data, coordset=[Coord(x, units="s")])
    original = source.copy()

    result = source.detrend(order="linear", breakpoints=[4.0])

    np.testing.assert_allclose(result.data, 0.0, atol=1.0e-12)
    np.testing.assert_array_equal(source.data, original.data)
    assert source.x == original.x
