"""Core operations on explicitly associated ordinary complex channels."""

import numpy as np
import pytest

import spectrochempy as scp


def _channels():
    values = np.arange(2 * 5 * 8, dtype=float).reshape(2, 5, 8)
    data = values + 1j * (values[::-1] + 0.5)
    source = scp.NDDataset(
        data,
        dims=["c", "y", "x"],
        coordset=[
            scp.Coord.arange(2, labels=["A", "B"]),
            scp.Coord(np.arange(5) * 0.25, units="s", title="time"),
            scp.Coord(np.arange(8) * 0.125, units="s", title="time"),
        ],
    )
    source.meta.td = list(source.shape)
    source.meta.isfreq = [False, False, False]
    return source


@pytest.mark.parametrize("dim,axis", [("x", 2), ("y", 1)])
def test_complex_channel_fft_matches_numpy_and_preserves_source(dim, axis):
    source = _channels()
    before = source.data.copy()

    spectrum = scp.fft(source, dim=dim)
    expected = np.fft.fftshift(np.fft.fft(before, axis=axis), axes=axis)

    np.testing.assert_allclose(spectrum.data, expected, atol=1e-10)
    np.testing.assert_array_equal(source.data, before)
    assert spectrum.shape == source.shape
    assert spectrum.dtype.kind == "c"
    assert spectrum.dims == ["c", "y", "x"]
    assert spectrum.coordset["c"].labels.tolist() == ["A", "B"]


def test_complex_channel_components_and_native_roundtrip(tmp_path):
    source = _channels()
    mask = np.zeros(source.shape, dtype=bool)
    mask[1, 2, 3] = True
    source.mask = mask

    np.testing.assert_array_equal(source.real.data, source.data.real)
    np.testing.assert_array_equal(source.imag.data, source.data.imag)
    np.testing.assert_array_equal(source.real.mask, source.mask)
    np.testing.assert_array_equal(source.imag.mask, source.mask)

    path = source.save_as(tmp_path / "complex_channels", confirm=False)
    loaded = scp.NDDataset.load(path)

    np.testing.assert_array_equal(loaded.data, source.data)
    np.testing.assert_array_equal(loaded.mask, source.mask)
    assert loaded.dims == source.dims
    assert loaded.coordset["c"].labels.tolist() == ["A", "B"]
