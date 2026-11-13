# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

import spectrochempy as scp


@pytest.mark.parametrize(
    "settings",
    [
        {"method": "avg"},
        {"method": "han"},
        {"method": "median"},
        {"method": "savgol"},
        {"method": "whittaker"},
        {"method": "gaussian", "sigma": 2, "sigma_units": "coordinate"},
        {"method": "butterworth", "cutoff": 0.15},
        {"method": "notch", "period": 5},
        {"method": "wavelet"},
    ],
)
def test_split_filters_each_spectrum_along_selected_dimension(settings):
    if settings["method"] == "wavelet":
        pytest.importorskip("pywt")
    rng = np.random.default_rng(1852)
    data = rng.normal(size=(48, 3))
    data[24:, :] += 10
    mask = np.zeros(data.shape, dtype=bool)
    mask[20:24, 0] = True
    mask[25:29, 1] = True
    mask[:, 2] = True
    data[mask] = np.nan
    coord = scp.Coord(np.arange(48.0), units="cm")
    source = scp.NDDataset(data, mask=mask, name="gapped", units="absorbance")
    source.set_coordset(y=coord)
    source.history = "Input history"

    result = scp.Filter(mask_policy="split", **settings).transform(source, dim="y")

    np.testing.assert_array_equal(result.mask, mask)
    np.testing.assert_array_equal(result.data[mask], data[mask])
    np.testing.assert_array_equal(source.data, data)
    np.testing.assert_array_equal(result.y.data, coord.data)
    assert result.name == source.name
    assert result.units == source.units
    assert result.history_entries[:-1] == source.history_entries
    history = result.history_entries[-1]["parameters"]
    assert history["mask_policy"] == "split"
    for column, start, stop in [(0, 0, 20), (0, 24, 48), (1, 0, 25), (1, 29, 48)]:
        piece = scp.NDDataset(data[start:stop, column], coordset=[coord[start:stop]])
        expected = scp.Filter(**settings).transform(piece)
        np.testing.assert_allclose(result.data[start:stop, column], expected.data)
    if settings["method"] == "wavelet":
        assert len(history["segments"]) == 4
        assert history["segments"][0]["start"] == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"method": "savgol", "size": 5},
        {"method": "whittaker", "order": 2},
        {"method": "butterworth", "cutoff": 0.15},
        {"method": "notch", "period": 5},
        {"method": "wavelet"},
        {"method": "wavelet", "wavelet": "haar", "wavelet_level": 3},
    ],
)
def test_split_rejects_short_segment(settings):
    if settings["method"] == "wavelet":
        pytest.importorskip("pywt")
    source = scp.NDDataset(np.arange(25.0), coordset=[scp.Coord(np.arange(25.0))])
    source[20:23] = scp.MASKED

    with pytest.raises(ValueError, match=r"segment \[23:25\)"):
        scp.Filter(mask_policy="split", **settings).transform(source)
    assert not source.history_entries


@pytest.mark.parametrize("units", ["absorbance", "V"])
def test_split_derivative_uses_signed_spacing_and_preserves_mask(units):
    coord = scp.Coord(np.arange(20.0, 0, -0.5), units="cm")
    source = scp.NDDataset(
        3 * coord.data, coordset=[coord], units=units, title="signal"
    )
    source[12:20] = scp.MASKED
    result = source.savgol(deriv=1, mask_policy="split")
    np.testing.assert_allclose(result.data[~source.mask], 3)
    np.testing.assert_array_equal(result.mask, source.mask)
    assert result.units == source.units / coord.units
    assert result.title == "signal (1st derivative)"
    assert result.history_entries[-1]["parameters"]["delta"] == -0.5


def test_split_fully_masked_dataset_is_copied():
    source = scp.NDDataset([2**53 + 1, 2], mask=[True, True], dtype=np.int64)
    result = scp.Filter(method="wavelet", mask_policy="split").transform(source)
    assert result is not source
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result.data, source.data)
    np.testing.assert_array_equal(result.mask, source.mask)


@pytest.mark.parametrize("mask", [[True, False, False], [False, False, False]])
def test_split_rejects_unmasked_nonfinite_sample(mask):
    source = scp.NDDataset([0.0, np.nan, 2.0], mask=mask)
    with pytest.raises(ValueError, match="requires finite data"):
        scp.Filter(method="avg", mask_policy="split").transform(source)


def test_split_derivative_warns_once_on_irregular_spacing():
    source = scp.NDDataset(
        np.arange(48.0).reshape(2, 24),
        coordset=[scp.Coord([0, 1]), scp.Coord(np.arange(24.0) ** 2)],
    )
    source[:, 10:14] = scp.MASKED

    with pytest.warns(UserWarning, match="index-based delta") as records:
        result = source.savgol(deriv=1, mask_policy="split")

    assert len(records) == 1
    np.testing.assert_allclose(result.data[~source.mask], 1.0)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [np.complex128, np.clongdouble])
def test_split_rejects_complex_window_filter_input(masked, dtype):
    source = scp.NDDataset(np.arange(10.0) * (1 + 2j), dtype=dtype)
    if masked:
        source[5] = scp.MASKED

    with pytest.raises(ValueError, match="requires real data"):
        scp.Filter(method="median", mask_policy="split").transform(source)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"), [(np.int64, np.float64), (np.float32, np.float32)]
)
def test_split_keeps_filter_output_dtype(dtype, expected_dtype):
    source = scp.NDDataset([0, 1, 0, 1, 0, 1, 0], dtype=dtype)
    source[3] = scp.MASKED
    filtered = scp.Filter(method="gaussian", sigma=1, mask_policy="split")(source)
    expected = scp.Filter(method="gaussian", sigma=1)(source[:3])

    assert filtered.dtype == expected_dtype
    np.testing.assert_allclose(filtered.data[:3], expected.data)
    assert filtered.data[3] == source.data[3]
    np.testing.assert_array_equal(filtered.mask, source.mask)


@pytest.mark.parametrize("delta", [None, 2.0])
def test_split_derivative_uses_active_coordinate(delta):
    x = np.arange(20.0)
    source = scp.NDDataset(3 * x, units="V")
    source.set_coordset(x=[scp.Coord(2 * x, units="s"), scp.Coord(x, units="cm")])
    source.x.select(2)
    source[8:12] = scp.MASKED

    result = source.savgol(deriv=1, delta=delta, mask_policy="split")

    np.testing.assert_allclose(result.data[~source.mask], 1.5)
    assert result.units == source.units / source.x.default.units


def test_split_derivative_resolves_coordinate_reference():
    x = np.arange(20.0) * -0.25
    coord = scp.Coord(x, units="cm")
    source = scp.NDDataset(
        np.broadcast_to(3 * x, (20, 20)).T.copy(),
        units="V",
        coordset=scp.CoordSet(x=coord, y="x"),
    )
    source[8:12, :] = scp.MASKED

    result = source.savgol(deriv=1, dim="y", mask_policy="split")

    np.testing.assert_allclose(result.data[~source.mask], 3.0)
    assert result.units == source.units / coord.units
