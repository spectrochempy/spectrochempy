# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import NDDataset


def _frequency_dataset(shape=(2, 8), transformed_axis=-1):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    data = data + 1.0j * (data[::-1] + 0.5)
    spectrum = np.fft.fftshift(
        np.fft.fft(data, axis=transformed_axis), axes=transformed_axis
    )

    coords = [Coord.arange(length) for length in shape]
    size = shape[transformed_axis]
    frequencies = np.fft.fftshift(np.fft.fftfreq(size, d=0.25))
    coords[transformed_axis] = Coord(frequencies, units="Hz", title="frequency")
    return NDDataset(spectrum, coordset=coords, title="synthetic spectrum")


def _expected_ifft(dataset, size, axis=-1):
    unshifted = np.fft.ifftshift(dataset.data, axes=axis)
    return np.fft.ifft(unshifted, n=size, axis=axis)


def _assert_time_coordinate(result, spectrum, size, dim="x"):
    frequency = spectrum.coordset[dim]
    expected_step = 1.0 / (size * abs(frequency.spacing))
    expected = np.arange(size) * expected_step.to("us").magnitude

    coordinate = result.coordset[dim]
    assert coordinate.title == "time"
    assert coordinate.units == expected_step.to("us").units
    np.testing.assert_allclose(coordinate.data, expected, rtol=2.0e-4)


def test_ifft_default_and_equal_size_preserve_existing_result_and_source():
    spectrum = _frequency_dataset()
    original_data = spectrum.data.copy()
    original_coordinate = spectrum.x.copy()

    default = spectrum.ifft()
    equal = spectrum.ifft(size=spectrum.x.size)

    assert default.shape == spectrum.shape
    np.testing.assert_allclose(default.data, _expected_ifft(spectrum, 8))
    np.testing.assert_allclose(equal.data, default.data)
    _assert_time_coordinate(default, spectrum, 8)
    _assert_time_coordinate(equal, spectrum, 8)
    np.testing.assert_array_equal(spectrum.data, original_data)
    assert spectrum.x == original_coordinate


def test_fft_ifft_roundtrip_preserves_values_coordinate_and_domain_metadata():
    data = np.arange(8, dtype=float) + 1.0j * np.arange(8, 16, dtype=float)
    time = Coord.arange(8) * 0.25
    time.units = "s"
    time.title = "time"
    source = NDDataset(data, coordset=[time])
    source.meta.td = [source.size]
    source.meta.isfreq = [False]

    spectrum = source.fft()
    result = spectrum.ifft()

    assert spectrum.meta.isfreq == [True]
    assert result.meta.isfreq == [False]
    np.testing.assert_allclose(result.data, source.data)
    np.testing.assert_allclose(result.x.to("s").data, source.x.data)


@pytest.mark.parametrize("size", [12, 6])
def test_ifft_size_controls_values_shape_and_coordinate(size):
    spectrum = _frequency_dataset()
    original_data = spectrum.data.copy()
    original_coordinate = spectrum.x.copy()

    result = spectrum.ifft(size=size)

    assert result.shape == (2, size)
    np.testing.assert_allclose(result.data, _expected_ifft(spectrum, size))
    _assert_time_coordinate(result, spectrum, size)
    np.testing.assert_array_equal(spectrum.data, original_data)
    assert spectrum.x == original_coordinate


def test_ifft_si_alias_controls_size():
    spectrum = _frequency_dataset()

    result = spectrum.ifft(si=12)

    assert result.shape == (2, 12)
    np.testing.assert_allclose(result.data, _expected_ifft(spectrum, 12))
    _assert_time_coordinate(result, spectrum, 12)


@pytest.mark.parametrize("size", [12, 6])
def test_ifft_resized_time_axis_preserves_known_frequency(size):
    frequencies = np.fft.fftshift(np.fft.fftfreq(8, d=0.25))
    spectrum_data = np.zeros(8, dtype=complex)
    spectrum_data[np.flatnonzero(np.isclose(frequencies, 0.5))[0]] = 1.0
    spectrum = NDDataset(
        spectrum_data,
        coordset=[Coord(frequencies, units="Hz", title="frequency")],
    )

    result = spectrum.ifft(size=size)

    time_step = result.x.spacing.to("s").magnitude
    phase_step = np.angle(result.data[1] / result.data[0])
    recovered_frequency = phase_step / (2.0 * np.pi * time_step)
    assert recovered_frequency == pytest.approx(0.5, rel=2.0e-4)


@pytest.mark.parametrize("inplace", [False, True])
def test_ifft_size_on_nonfinal_dimension_and_inplace(inplace):
    spectrum = _frequency_dataset(shape=(8, 3), transformed_axis=0)
    original = spectrum.copy()

    result = spectrum.ifft(dim="y", size=6, inplace=inplace)

    assert (result is spectrum) is inplace
    assert result.shape == (6, 3)
    np.testing.assert_allclose(result.data, _expected_ifft(original, 6, axis=0))
    _assert_time_coordinate(result, original, 6, dim="y")
    assert result.x == original.x
    if not inplace:
        np.testing.assert_array_equal(spectrum.data, original.data)
        assert spectrum.y == original.y
