# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import warnings

import numpy as np
import pytest

from spectrochempy import Coord
from spectrochempy import Correlation2DResult as PublicCorrelation2DResult
from spectrochempy import NDDataset
from spectrochempy import correlation_spectroscopy as public_correlation_spectroscopy
from spectrochempy.analysis.correlation_spectroscopy import Correlation2DResult
from spectrochempy.analysis.correlation_spectroscopy import correlation_spectroscopy


def _series(data, perturbation=(0.0, 1.0, 2.0)):
    return NDDataset(
        data,
        coordset=[
            Coord(perturbation, title="time", units="s"),
            Coord([1000.0, 1100.0], title="wavenumber", units="cm^-1"),
        ],
        units="absorbance",
    )


def test_correlation_spectroscopy_is_exposed_at_the_package_root():
    assert public_correlation_spectroscopy is correlation_spectroscopy
    assert PublicCorrelation2DResult is Correlation2DResult


def test_correlation_spectroscopy_returns_symmetric_and_antisymmetric_maps():
    dataset = _series([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])

    result = correlation_spectroscopy(dataset)

    centered = dataset.data - dataset.data.mean(axis=0, keepdims=True)
    expected_synchronous = centered.T @ centered / (len(dataset.y) - 1)
    indices = np.arange(len(dataset.y))
    hilbert_noda = np.zeros((indices.size, indices.size))
    for i in range(indices.size):
        for j in range(indices.size):
            if i != j:
                hilbert_noda[i, j] = 1.0 / (np.pi * (j - i))
    expected_asynchronous = centered.T @ hilbert_noda @ centered / (indices.size - 1)

    assert isinstance(result, Correlation2DResult)
    assert result.perturbation_dim == "y"
    np.testing.assert_allclose(result.synchronous.data, expected_synchronous)
    np.testing.assert_allclose(result.asynchronous.data, expected_asynchronous)
    np.testing.assert_allclose(result.synchronous.data, result.synchronous.data.T)
    np.testing.assert_allclose(
        result.asynchronous.data, -result.asynchronous.data.T, atol=1e-14
    )
    np.testing.assert_array_equal(result.synchronous.x.data, dataset.x.data)
    np.testing.assert_array_equal(result.synchronous.y.data, dataset.x.data)
    assert result.synchronous.units == dataset.units**2
    assert result.asynchronous.units == dataset.units**2


def test_correlation_spectroscopy_accepts_x_as_perturbation_dimension():
    dataset = NDDataset(
        [[1.0, 4.0, 3.0], [2.0, 1.0, 2.0]],
        coordset=[
            Coord([1000.0, 1100.0], title="wavenumber", units="cm^-1"),
            Coord([0.0, 1.0, 2.0], title="time", units="s"),
        ],
        units="absorbance",
    )

    result = correlation_spectroscopy(dataset, perturbation_dim="x")

    assert result.perturbation_dim == "x"
    assert result.synchronous.shape == (2, 2)
    np.testing.assert_array_equal(result.synchronous.x.data, dataset.y.data)
    np.testing.assert_array_equal(result.synchronous.y.data, dataset.y.data)


@pytest.mark.parametrize(
    "perturbation, message",
    [
        ((0.0, 2.0, 1.0), "strictly monotonic"),
        ((0.0, 1.0, 3.0), "uniformly spaced"),
    ],
)
def test_correlation_spectroscopy_rejects_invalid_perturbation_coordinates(
    perturbation, message
):
    with pytest.raises(ValueError, match=message):
        correlation_spectroscopy(
            _series([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], perturbation)
        )


def test_correlation_spectroscopy_spacing_validation_respects_coord_precision():
    tiny_irregular = NDDataset(
        [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]],
        coordset=[
            Coord([0.0, 1e-15, 3e-15], title="time", units="s", rounding=False),
            Coord([1000.0, 1100.0], title="wavenumber", units="cm^-1"),
        ],
    )
    with pytest.raises(ValueError, match="uniformly spaced"):
        correlation_spectroscopy(tiny_irregular)

    epoch_seconds = 1.7e9 + np.arange(10) * 0.001
    regular_epoch = NDDataset(
        np.arange(20, dtype=float).reshape(10, 2),
        coordset=[
            Coord(epoch_seconds, title="time", units="s", rounding=False),
            Coord([1000.0, 1100.0], title="wavenumber", units="cm^-1"),
        ],
    )
    assert regular_epoch.y.linear
    assert correlation_spectroscopy(regular_epoch).synchronous.shape == (2, 2)


def test_correlation_spectroscopy_follows_coord_linearization_settings():
    data = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]
    spectral = Coord([1000.0, 1100.0], title="wavenumber", units="cm^-1")

    normalized_coord = Coord(
        [0.0, 1.0, 2.01], title="time", units="s", linearize_below=1.0
    )
    assert normalized_coord.linear
    np.testing.assert_allclose(normalized_coord.data, [0.0, 1.005, 2.01])
    normalized_dataset = NDDataset(
        data, coordset=[normalized_coord, spectral], units="absorbance"
    )
    assert correlation_spectroscopy(normalized_dataset).synchronous.shape == (2, 2)

    strict_coord = Coord(
        [0.0, 1.0, 2.01],
        title="time",
        units="s",
        rounding=False,
        linearize_below=0.0,
    )
    assert not strict_coord.linear
    strict_dataset = NDDataset(data, units="absorbance")
    strict_dataset.set_coordset(y=strict_coord, x=spectral)
    assert not strict_dataset.y.linear
    with pytest.raises(ValueError, match="uniformly spaced"):
        correlation_spectroscopy(strict_dataset)


def test_correlation_spectroscopy_rejects_missing_or_nonfinite_signal_values():
    with pytest.raises(ValueError, match="finite values"):
        correlation_spectroscopy(_series([[1.0, 4.0], [2.0, np.nan], [4.0, 1.0]]))

    masked = _series([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    masked.mask = np.array([[False, False], [False, True], [False, False]], dtype=bool)
    with pytest.raises(ValueError, match="masked values"):
        correlation_spectroscopy(masked)

    complex_dataset = _series([[1.0 + 1.0j, 4.0], [2.0, 2.0], [4.0, 1.0]])
    with pytest.raises(TypeError, match="real numeric data"):
        correlation_spectroscopy(complex_dataset)


def test_correlation_spectroscopy_validates_input_and_dimension():
    with pytest.raises(TypeError, match="NDDataset"):
        correlation_spectroscopy([[1.0, 2.0]])

    with pytest.raises(ValueError, match="two-dimensional"):
        correlation_spectroscopy(NDDataset([1.0, 2.0]))

    with pytest.raises(ValueError, match="perturbation_dim"):
        correlation_spectroscopy(_series([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]), "z")


def test_correlation_spectroscopy_matches_analytic_three_point_series():
    dataset = _series([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]])

    result = correlation_spectroscopy(dataset)

    np.testing.assert_allclose(result.synchronous.data, [[1.0, 0.0], [0.0, 3.0]])
    cross_peak = 3.0 / (2.0 * np.pi)
    np.testing.assert_allclose(
        result.asynchronous.data,
        [[0.0, cross_peak], [-cross_peak, 0.0]],
        atol=1e-14,
    )


def test_correlation_spectroscopy_reversing_time_changes_only_asynchronous_sign():
    dataset = _series([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]])

    forward = correlation_spectroscopy(dataset)
    backward = correlation_spectroscopy(dataset[::-1])

    np.testing.assert_allclose(backward.synchronous.data, forward.synchronous.data)
    np.testing.assert_allclose(
        backward.asynchronous.data, -forward.asynchronous.data, atol=1e-14
    )
    transposed = correlation_spectroscopy(dataset.T, perturbation_dim="y")
    np.testing.assert_allclose(transposed.synchronous.data, forward.synchronous.data)
    np.testing.assert_allclose(transposed.asynchronous.data, forward.asynchronous.data)


def test_correlation_spectroscopy_mean_centering_removes_static_spectrum():
    dataset = _series([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]])
    shifted = dataset.copy()
    shifted.data = dataset.data + [100.0, -50.0]

    original = correlation_spectroscopy(dataset)
    result = correlation_spectroscopy(shifted)

    np.testing.assert_allclose(result.synchronous.data, original.synchronous.data)
    np.testing.assert_allclose(result.asynchronous.data, original.asynchronous.data)


def test_correlation_spectroscopy_proportional_changes_have_no_asynchronous_map():
    dataset = _series([[1.0, 6.0], [2.0, 4.0], [3.0, 2.0]])

    result = correlation_spectroscopy(dataset)

    np.testing.assert_allclose(result.synchronous.data, [[1.0, -2.0], [-2.0, 4.0]])
    np.testing.assert_allclose(result.asynchronous.data, 0.0, atol=1e-14)


def test_correlation_spectroscopy_rejects_irregular_times_with_default_coord():
    dataset = _series([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]], (0.0, 1.0, 2.01))

    with pytest.raises(ValueError, match="uniformly spaced"):
        correlation_spectroscopy(dataset)


@pytest.mark.parametrize(
    "perturbation",
    [
        [0.0 + 1.0j, 1.0 + 2.0j, 2.0 + 3.0j],
        [complex(0.0, np.nan), complex(1.0, np.nan)],
    ],
    ids=["finite-complex", "imaginary-nan"],
)
def test_correlation_spectroscopy_rejects_complex_perturbation(perturbation):
    dataset = NDDataset(np.arange(2 * len(perturbation)).reshape(-1, 2))
    dataset.set_coordset(
        y=Coord(perturbation, dtype="complex128", rounding=False),
        x=Coord([1000.0, 1100.0]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="perturbation coordinate must be real"):
            correlation_spectroscopy(dataset)
