"""Despiking diagnostics, valid runs and independent spectrum processing."""

import numpy as np
import pytest

import spectrochempy as scp

METHODS = ["katsumoto", "whitaker", "neighbors"]


def signal():
    x = np.arange(81)
    return 10 + 0.003 * x + 4 * np.exp(-0.5 * ((x - 40) / 2) ** 2)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [(81,), (2, 81), (2, 3, 81)])
@pytest.mark.parametrize("axis", [0, -1])
def test_diagnostic_matches_actual_changes_and_independent_spectra(method, shape, axis):
    data = np.broadcast_to(signal(), shape).copy()
    for i, row in enumerate(data.reshape(-1, 81)):
        row += 0.1 * i
        row[43 + i] += 20 + 2 * i
    data = np.moveaxis(data, -1, axis)
    source = scp.NDDataset(data, units="count", name="source", title="Raman intensity")
    source.set_coordset(
        **{
            source.dims[i]: scp.Coord(
                np.arange(n), units="cm^-1" if i == axis % len(shape) else "s"
            )
            for i, n in enumerate(data.shape)
        }
    )
    source.history = "acquired"
    source.meta["sample"] = "reference"
    before = source.history_entries
    result, changed = source.despike(
        method=method, dim=source.dims[axis], delta=6, return_mask=True
    )
    default = source.despike(method=method, dim=axis, delta=6)
    np.testing.assert_array_equal(result.data, default.data)
    np.testing.assert_array_equal(changed.data, result.data != source.data)
    assert changed.dtype == np.dtype(bool)
    assert changed.units is None
    assert method in changed.history_entries[-1]["message"]
    assert "delta=6" in changed.history_entries[-1]["message"]
    assert changed.dims == source.dims
    assert changed.coordset == source.coordset
    assert not np.shares_memory(changed.data, source.data)
    assert result.units == source.units
    assert result.name == source.name
    assert result.title == source.title
    assert result.meta["sample"] == "reference"
    assert result.history_entries[:-1] == before
    assert source.history_entries == before
    np.testing.assert_array_equal(source.data, data)
    rows = np.moveaxis(result.data, axis, -1).reshape(-1, 81)
    inputs = np.moveaxis(data, axis, -1).reshape(-1, 81)
    for row, original in zip(rows, inputs, strict=True):
        individual = scp.NDDataset(original)
        np.testing.assert_array_equal(
            row, individual.despike(method=method, delta=6).data
        )


@pytest.mark.parametrize("method", METHODS)
def test_masks_and_nonfinite_values_are_preserved_and_not_diagnosed(method):
    data = signal()
    data[12] = np.nan
    data[24] = np.inf
    data[35] += 20
    data[60] += 100
    mask = np.zeros(81, dtype=bool)
    mask[60] = True
    source = scp.NDDataset(data, mask=mask)
    result, changed = source.despike(method=method, delta=6, return_mask=True)
    np.testing.assert_array_equal(result.mask, source.mask)
    np.testing.assert_array_equal(changed.mask, source.mask)
    assert np.isnan(result.data[12])
    assert result.data[24] == np.inf
    assert result.data[60] == data[60]
    assert not changed.data[[12, 24, 60]].any()
    valid = ~mask & np.isfinite(data)
    np.testing.assert_array_equal(changed.data, valid & (data != result.data))
    if method != "neighbors":
        for start, stop in [(0, 12), (13, 24), (25, 60), (61, 81)]:
            expected = scp.NDDataset(data[start:stop]).despike(method=method, delta=6)
            np.testing.assert_array_equal(result.data[start:stop], expected.data)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("length", [1, 2, 5, 81])
def test_constant_and_short_traces(method, length):
    source = scp.NDDataset(np.full(length, 10.0))
    with np.errstate(all="raise"):
        result, changed = source.despike(method=method, return_mask=True)
    np.testing.assert_array_equal(result.data, source.data)
    assert not changed.data.any()


@pytest.mark.parametrize("method", METHODS)
def test_all_masked_trace(method):
    source = scp.NDDataset(signal(), mask=np.ones(81, dtype=bool))
    result, changed = source.despike(method=method, return_mask=True)
    np.testing.assert_array_equal(result.data, source.data)
    assert not changed.data.any()


@pytest.mark.parametrize("method", METHODS)
def test_descending_coordinates_and_unit_conversion(method):
    source = scp.NDDataset(
        signal(), coordset=[scp.Coord.linspace(1800, 200, 81, units="cm^-1")]
    )
    source.data[43] += 20
    result, changed = source.despike(method=method, delta=6, return_mask=True)
    converted = source.copy()
    converted.x = source.x.to("m^-1")
    result2, changed2 = converted.despike(method=method, delta=6, return_mask=True)
    np.testing.assert_array_equal(result.data, result2.data)
    np.testing.assert_array_equal(changed.data, changed2.data)
    assert changed.x == source.x
    assert changed2.x == converted.x


@pytest.mark.parametrize("method", METHODS)
def test_integer_data_allows_fractional_replacement(method):
    source = scp.NDDataset([0, 1, 0, 1, 0, 100, 1, 0, 1, 0, 1], dtype=np.int64)
    result, changed = source.despike(method=method, delta=6, size=5, return_mask=True)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_array_equal(changed.data, source.data != result.data)
    assert source.dtype == np.dtype(np.int64)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("delta", [0, -1, np.nan, np.inf])
def test_invalid_threshold(method, delta):
    with pytest.raises(ValueError, match="delta"):
        scp.NDDataset(signal()).despike(method=method, delta=delta)


@pytest.mark.parametrize("method", METHODS)
def test_complex_data_is_rejected(method):
    with pytest.raises(ValueError, match="real"):
        scp.NDDataset(signal().astype(complex)).despike(method=method)


@pytest.mark.parametrize("method", METHODS)
def test_masked_values_do_not_influence_valid_corrections(method):
    source = scp.NDDataset(signal())
    source.data[43] += 20
    source.mask = np.arange(81) == 25
    altered = source.copy()
    altered.data[25] = -1e12
    result = source.despike(method=method, delta=6)
    result2 = altered.despike(method=method, delta=6)
    np.testing.assert_array_equal(result.data[:25], result2.data[:25])
    np.testing.assert_array_equal(result.data[26:], result2.data[26:])


def test_neighbors_leaves_endpoints_and_masked_triplets_unchanged():
    source = scp.NDDataset([100, 0, 0, 100, 0, 0, 100])
    source.mask = [False, False, True, False, False, False, False]
    result, changed = source.despike(method="neighbors", return_mask=True)
    np.testing.assert_array_equal(result.data, source.data)
    assert not changed.data.any()


def test_short_valid_runs_are_not_joined_for_katsumoto():
    source = scp.NDDataset([0, 1, 30, 1, 0, 100, 0, 1, 30, 1, 0])
    source.mask = np.arange(11) == 5
    result, changed = source.despike(size=9, return_mask=True)
    np.testing.assert_array_equal(result.data, source.data)
    assert not changed.data.any()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("position", [40, 43, 65])
def test_injected_spike_qualification_on_narrow_band_reference(method, position):
    reference = signal() + np.random.default_rng(42).normal(0, 0.02, 81)
    observed = reference.copy()
    observed[position] += 20
    result, changed = scp.despike(
        scp.NDDataset(observed), method=method, delta=6, return_mask=True
    )
    if method == "whitaker" and position == 40:
        assert not changed.data[position]
        assert result.data[position] == observed[position]
    else:
        assert changed.data[position]
        tolerance = 3.0 if position == 40 else 1.5
        assert abs(result.data[position] - reference[position]) < tolerance
    np.testing.assert_array_equal(changed.data, result.data != observed)


@pytest.mark.parametrize("method", ["katsumoto", "whitaker"])
def test_float32_dtype_is_preserved(method):
    source = scp.NDDataset(signal(), dtype=np.float32)
    source.data[43] += 20
    result = source.despike(method=method, delta=6)
    assert result.dtype == source.dtype
