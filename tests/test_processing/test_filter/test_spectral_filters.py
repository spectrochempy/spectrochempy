# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

import spectrochempy as scp

FILTERS = [
    {"method": "butterworth", "cutoff": 0.08},
    {"method": "notch", "period": 12.0},
    {"method": "gaussian", "sigma": 2.0, "sigma_units": "coordinate"},
]


@pytest.mark.parametrize("options", FILTERS)
def test_spectral_filters_batch_and_history(options):
    axis = np.arange(127.0, -1.0, -1.0)
    data = np.stack([np.sin(axis), np.cos(axis)], axis=1)
    dataset = scp.NDDataset(data, units="absorbance", name="spectra")
    dataset.set_coordset(y=scp.Coord(axis, units="cm^-1"), x=scp.Coord([0, 1]))
    dataset.history = "Original acquisition"
    model = scp.Filter(**options)
    result = model.transform(dataset, dim="y")
    expected = np.column_stack(
        [model.transform(dataset[:, i].squeeze()).data for i in range(2)]
    )
    np.testing.assert_allclose(result.data, expected)
    np.testing.assert_array_equal(result.y.data, dataset.y.data)
    np.testing.assert_array_equal(result.x.data, dataset.x.data)
    np.testing.assert_array_equal(dataset.data, data)
    assert result.units == dataset.units
    assert result.y.units == dataset.y.units
    assert result.history[:-1] == dataset.history
    entry = result.history_entries[-1]
    assert entry["operation"] == options["method"]
    assert entry["parameters"]["dimension"] == "y"
    for name, value in options.items():
        if name != "method":
            assert entry["parameters"][name] == value


@pytest.mark.parametrize("options", FILTERS)
@pytest.mark.parametrize(
    ("invalid", "message"),
    [(np.nan, "finite data"), (np.inf, "finite data"), (scp.MASKED, "unmasked data")],
)
def test_spectral_filters_reject_invalid_data(options, invalid, message):
    dataset = scp.NDDataset(np.ones(32))
    dataset[10] = invalid
    with pytest.raises(ValueError, match=message):
        scp.Filter(**options).transform(dataset)


@pytest.mark.parametrize("options", FILTERS)
def test_spectral_filters_reject_irregular_axis(options):
    axis = np.arange(32.0)
    axis[10] += 0.25
    dataset = scp.NDDataset(np.ones(32), coordset=[scp.Coord(axis)])
    with pytest.raises(ValueError, match="uniformly spaced coordinate"):
        scp.Filter(**options).transform(dataset)


@pytest.mark.parametrize(
    ("options", "minimum"),
    [
        ({"method": "butterworth", "cutoff": 0.08, "order": 3}, 13),
        ({"method": "butterworth", "cutoff": 0.08, "order": 4}, 16),
        ({"method": "notch", "period": 12.0}, 10),
    ],
)
def test_iir_padding_limit(options, minimum):
    dataset = scp.NDDataset(np.ones(minimum), coordset=[scp.Coord(np.arange(minimum))])
    model = scp.Filter(**options)
    with pytest.raises(ValueError, match=f"at least {minimum} samples"):
        model.transform(dataset[:-1])
    np.testing.assert_allclose(model.transform(dataset).data, 1.0)


@pytest.mark.parametrize("order", [1, 3, 4])
def test_butterworth_combined_cutoff_and_phase(order):
    axis = np.arange(4096) * 0.5
    cutoff = 0.125
    wave = np.sin(2 * np.pi * cutoff * axis)
    dataset = scp.NDDataset(wave, coordset=[scp.Coord(axis)])
    result = scp.Filter(method="butterworth", cutoff=cutoff, order=order)(dataset)
    np.testing.assert_allclose(
        result.data[256:-256], wave[256:-256] / np.sqrt(2), atol=1e-9
    )


def test_notch_reduces_ftir_fringe_and_preserves_band_area():
    axis = np.arange(2048.0)
    band = np.exp(-0.5 * ((axis - 1000) / 45) ** 2)
    fringe = 0.04 * np.sin(2 * np.pi * axis / 16)
    dataset = scp.NDDataset(
        np.stack([band, band + fringe]),
        coordset=[scp.Coord([0, 1]), scp.Coord(4000 - axis, units="cm^-1")],
    )
    result = scp.Filter(method="notch", period=16.0, quality_factor=12.0)(dataset)
    interior = slice(512, -512)
    residual = (result.data[1] - result.data[0])[interior]
    assert np.linalg.norm(residual) < 0.02 * np.linalg.norm(fringe[interior])
    band_window = slice(800, 1201)
    area = np.trapezoid(band[band_window])
    np.testing.assert_allclose(
        np.trapezoid(result.data[:, band_window], axis=-1), area, rtol=1e-3
    )


@pytest.mark.parametrize("mode", ["interp", "wrap", "constant"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_gaussian_scipy_parity(mode, complex_input):
    data = np.zeros(32, dtype=complex if complex_input else float)
    data[0] = 2 + 3j if complex_input else 2
    dataset = scp.NDDataset(data)
    result = scp.Filter(method="gaussian", sigma=2.0, mode=mode, cval=0.5)(dataset)
    expected = gaussian_filter1d(
        data, 2.0, mode="reflect" if mode == "interp" else mode, cval=0.5
    )
    np.testing.assert_allclose(result.data, expected)


def test_gaussian_coordinate_sigma_matches_point_sigma():
    data = np.zeros(32)
    data[15] = 1.0
    coordinate = scp.Coord(np.arange(32) * 0.5, units="Hz")
    dataset = scp.NDDataset(data, coordset=[coordinate])
    model = scp.Filter(method="gaussian", sigma=1.5, sigma_units="coordinate")
    result = model(dataset)
    np.testing.assert_allclose(result.data, gaussian_filter1d(data, 3.0))
    assert result.history_entries[-1]["parameters"]["sigma_points"] == 3.0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "butterworth", "cutoff": 0.5}, "below Nyquist"),
        ({"method": "butterworth", "cutoff": 0.08, "order": 0}, "positive"),
        ({"method": "notch", "period": 2.0}, "below Nyquist"),
        (
            {"method": "notch", "period": 12.0, "quality_factor": np.inf},
            "quality_factor",
        ),
        (
            {"method": "notch", "period": 2.5, "quality_factor": 0.6},
            "bandwidth below Nyquist",
        ),
        ({"method": "gaussian", "sigma": 0.0}, "positive"),
    ],
)
def test_spectral_filters_reject_invalid_settings(options, message):
    dataset = scp.NDDataset(np.ones(32), coordset=[scp.Coord(np.arange(32))])
    with pytest.raises(ValueError, match=message):
        scp.Filter(**options)(dataset)


@pytest.mark.parametrize("options", FILTERS)
def test_spectral_filters_require_coordinate_for_physical_parameters(options):
    dataset = scp.NDDataset(np.ones(32))
    with pytest.raises(ValueError, match="uniformly spaced coordinate"):
        scp.Filter(**options)(dataset)


@pytest.mark.parametrize("options", FILTERS)
def test_spectral_filters_use_active_coordinate_and_keep_group(options):
    samples = np.arange(64)
    data = np.sin(samples / 2)
    axis = scp.Coord(1000 - samples * 0.5, title="wavenumber", units="cm^-1")
    other = scp.Coord(samples, title="sample number")
    dataset = scp.NDDataset(data)
    dataset.set_coordset(x=[axis, other])
    dataset.x.select(2)
    assert dataset.x.default.title == axis.title

    result = scp.Filter(**options)(dataset)
    expected = scp.Filter(**options)(scp.NDDataset(data, coordset=[axis]))
    np.testing.assert_allclose(result.data, expected.data)
    assert result.x.default_index == dataset.x.default_index
    assert result.x.default.title == axis.title
    assert result.x.default.units == axis.units
    assert result.x.names == dataset.x.names
    for name in dataset.x.names:
        np.testing.assert_array_equal(result.x[name].data, dataset.x[name].data)


def test_spectral_filters_handle_partial_coordset():
    data = np.tile(np.sin(np.arange(64)), (4, 1))
    dataset = scp.NDDataset(data)
    dataset.set_coordset(y=scp.Coord(np.arange(4)))

    result = scp.Filter(method="gaussian", sigma=2.0)(dataset)
    np.testing.assert_allclose(result.data, gaussian_filter1d(data, 2.0))
    np.testing.assert_array_equal(result.y.data, dataset.y.data)
    assert result.x is None

    with pytest.raises(ValueError, match="uniformly spaced coordinate"):
        scp.Filter(method="butterworth", cutoff=0.08)(dataset)


def test_gaussian_preserves_single_sample_geometry():
    dataset = scp.NDDataset([5.0], coordset=[scp.Coord([2.0], units="Hz")])
    result = scp.Filter(method="gaussian", sigma=2.0)(dataset)
    assert result.shape == (1,)
    assert result.dims == dataset.dims
    np.testing.assert_array_equal(result.x.data, dataset.x.data)
    assert result.x.units == dataset.x.units
    np.testing.assert_allclose(result.data, dataset.data)


@pytest.mark.parametrize("options", FILTERS)
def test_spectral_filters_preserve_3d_geometry(options):
    axis = np.arange(64.0)
    data = np.sin(axis)[None, :, None] * np.arange(1, 7).reshape(2, 1, 3)
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord(np.arange(2)), scp.Coord(axis), scp.Coord(np.arange(3))],
    )
    model = scp.Filter(**options)
    result = model(dataset, dim="y")
    assert result.shape == dataset.shape
    assert result.dims == dataset.dims
    np.testing.assert_array_equal(result.y.data, axis)
    for i, j in np.ndindex(2, 3):
        spectrum = scp.NDDataset(data[i, :, j], coordset=[scp.Coord(axis)])
        np.testing.assert_allclose(result.data[i, :, j], model(spectrum).data)


def test_gaussian_tiny_positive_sigma_is_identity():
    data = np.arange(8.0)
    dataset = scp.NDDataset(data)
    result = scp.Filter(method="gaussian", sigma=1e-300)(dataset)
    np.testing.assert_array_equal(result.data, data)
    assert not np.shares_memory(result.data, dataset.data)


@pytest.mark.parametrize(
    "options",
    [
        {"method": "butterworth", "cutoff": 1e-8},
        {"method": "butterworth", "cutoff": 1e-9},
        {"method": "notch", "period": 1e9},
        {"method": "notch", "period": 16.0, "quality_factor": 1e20},
    ],
)
def test_iir_rejects_unresolvable_settings(options):
    dataset = scp.NDDataset(np.ones(64), coordset=[scp.Coord(np.arange(64))])
    with pytest.raises(ValueError, match="outside the numerically reliable range"):
        scp.Filter(**options)(dataset)


def test_coordinate_based_filter_preserves_axis_reference():
    axis = scp.Coord(np.arange(32) * 0.5, units="Hz")
    data = np.broadcast_to(np.sin(axis.data)[:, None], (32, 32))
    dataset = scp.NDDataset(data)
    dataset.set_coordset(y="x", x=axis)
    model = scp.Filter(method="gaussian", sigma=1.0, sigma_units="coordinate")

    result = model(dataset, dim="y")
    np.testing.assert_allclose(result.data, gaussian_filter1d(data, 2.0, axis=0))
    assert result.coordset.references == {"y": "x"}


@pytest.mark.parametrize("dtype", [np.int16, np.float16, np.longdouble])
def test_gaussian_filters_unsupported_scipy_dtypes_as_double(dtype):
    impulse = [0, 1, 0, 0, 0, 0, 0, 0]
    dataset = scp.NDDataset(impulse, dtype=dtype, units="V")
    result = scp.Filter(method="gaussian", sigma=1.0)(dataset)
    np.testing.assert_allclose(
        result.data, gaussian_filter1d(np.array(impulse, float), 1)
    )
    assert result.dtype == np.dtype(np.float64)
    assert result.units == dataset.units
    assert dataset.dtype == np.dtype(dtype)
