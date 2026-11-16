import sys

import numpy as np
import pytest

import spectrochempy as scp


def test_wavelet_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "pywt", None)
    spectrum = scp.NDDataset(np.ones(32))
    np.testing.assert_allclose(scp.Filter(method="gaussian", sigma=1)(spectrum).data, 1)
    with pytest.raises(ImportError, match=r"spectrochempy\[wavelets\]"):
        scp.Filter(method="wavelet")(spectrum)


@pytest.fixture
def pywt():
    return pytest.importorskip("pywt")


@pytest.mark.parametrize("size", [63, 64])
@pytest.mark.parametrize("threshold", [0.0, 0.2])
def test_wavelet_matches_pywavelets(pywt, size, threshold):
    data = np.random.default_rng(1852).normal(size=size)
    spectrum = scp.NDDataset(data)
    model = scp.Filter(
        method="wavelet", wavelet="db2", wavelet_level=2, wavelet_threshold=threshold
    )
    actual = model(spectrum)
    approx, *details = pywt.wavedec(data, "db2", mode="symmetric", level=2)
    details = [pywt.threshold(detail, threshold, mode="soft") for detail in details]
    expected = pywt.waverec([approx, *details], "db2", mode="symmetric")[:size]
    assert actual.shape == spectrum.shape
    np.testing.assert_allclose(actual.data, expected, atol=1e-12)


def test_wavelet_batches_match_separate_spectra(pywt):
    rng = np.random.default_rng(1852)
    data = rng.normal(size=(129, 2)) * [0.1, 0.8]
    spectrum = scp.NDDataset(
        data,
        units="absorbance",
        coordset=[scp.Coord(np.arange(129.0)[::-1], units="cm^-1"), scp.Coord([0, 1])],
    )
    spectrum.history = "Synthetic acquisition"
    model = scp.Filter(method="wavelet")
    filtered = model(spectrum, dim="y")
    expected = [model(scp.NDDataset(column)) for column in data.T]
    np.testing.assert_allclose(
        filtered.data, np.column_stack([item.data for item in expected])
    )
    np.testing.assert_array_equal(filtered.y.data, spectrum.y.data)
    np.testing.assert_array_equal(filtered.x.data, spectrum.x.data)
    np.testing.assert_array_equal(spectrum.data, data)
    assert filtered.units == spectrum.units
    assert filtered.y.units == spectrum.y.units
    assert filtered.history_entries[:-1] == spectrum.history_entries
    entry = filtered.history_entries[-1]
    assert entry["operation"] == "wavelet"
    assert entry["parameters"]["dimension"] == "y"
    assert entry["parameters"]["level"] == 3
    finest = pywt.wavedec(data, "sym4", level=3, mode="symmetric", axis=0)[-1]
    center = np.median(finest, axis=0)
    mad = np.median(np.abs(finest - center), axis=0)
    threshold = mad / 0.6744897501960817 * np.sqrt(2 * np.log(data.shape[0]))
    assert entry["parameters"]["threshold"] == pytest.approx(threshold)
    assert entry["parameters"]["threshold"] == pytest.approx(
        [item.history_entries[-1]["parameters"]["threshold"] for item in expected]
    )


def test_wavelet_constant_has_finite_zero_threshold(pywt):
    dataset = scp.NDDataset(np.zeros(32))
    result = scp.Filter(method="wavelet")(dataset)
    np.testing.assert_array_equal(result.data, dataset.data)
    assert result.history_entries[-1]["parameters"]["threshold"] == 0


@pytest.mark.parametrize(
    ("data", "settings", "message"),
    [
        (np.ones(8), {}, "at least 14 samples"),
        (np.ones(32), {"wavelet_level": 0}, "wavelet_level"),
        (np.ones(32), {"wavelet_level": 10}, "wavelet_level"),
        (np.ones(32), {"wavelet_threshold": -1}, "wavelet_threshold"),
        (np.ones(32), {"wavelet_threshold": np.inf}, "wavelet_threshold"),
        (np.ones(32, dtype=complex), {}, "real data"),
    ],
)
def test_wavelet_invalid_settings(pywt, data, settings, message):
    dataset = scp.NDDataset(data)
    with pytest.raises(ValueError, match=message):
        scp.Filter(method="wavelet", **settings)(dataset)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [(np.nan, "finite data"), (np.inf, "finite data"), (scp.MASKED, "unmasked data")],
)
def test_wavelet_invalid_samples(invalid, message):
    dataset = scp.NDDataset(np.ones(32))
    dataset[10] = invalid
    with pytest.raises(ValueError, match=message):
        scp.Filter(method="wavelet")(dataset)


@pytest.mark.parametrize("threshold", [None, 1e300])
def test_wavelet_uses_double_precision_for_float32(pywt, threshold):
    data = np.full(33, 3e38, dtype=np.float32)
    dataset = scp.NDDataset(data, dtype=np.float32)
    model = scp.Filter(method="wavelet", wavelet_threshold=threshold)
    with np.errstate(over="raise"):
        result = model(dataset)
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(result.data, data, rtol=1e-10)
