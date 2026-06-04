import numpy as np
import pytest

import spectrochempy as scp


@pytest.fixture()
def synthetic_1d_baseline_dataset():
    x = np.linspace(4000.0, 1000.0, 150)

    true_baseline = 0.001 * (x - 2500.0) ** 2 + 0.5

    peak1 = 1.0 * np.exp(-((x - 3500.0) ** 2) / 8000.0)
    peak2 = 0.8 * np.exp(-((x - 1600.0) ** 2) / 5000.0)
    peaks = peak1 + peak2

    rng = np.random.default_rng(42)
    noise = 0.01 * rng.normal(size=x.shape)

    data = true_baseline + peaks + noise

    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord(x, title="wavenumber", units="cm^-1")],
        title="synthetic 1D baseline spectrum",
        units="absorbance",
    )
    dataset.name = "synthetic_1d"

    return dataset, true_baseline, peaks


@pytest.fixture()
def synthetic_2d_baseline_dataset(synthetic_1d_baseline_dataset):
    _, true_baseline_1d, peaks_1d = synthetic_1d_baseline_dataset

    n_spectra = 6
    n_wavelengths = len(true_baseline_1d)

    row_coord = np.arange(n_spectra, dtype=float)
    x = np.linspace(4000.0, 1000.0, n_wavelengths)

    data = np.zeros((n_spectra, n_wavelengths))
    true_baseline_2d = np.zeros((n_spectra, n_wavelengths))
    true_peaks_2d = np.zeros((n_spectra, n_wavelengths))

    for i in range(n_spectra):
        offset = 0.2 * i
        scale = 1.0 + 0.1 * i
        bl = true_baseline_1d + offset
        pk = peaks_1d * scale
        true_baseline_2d[i, :] = bl
        true_peaks_2d[i, :] = pk
        data[i, :] = bl + pk

    dataset = scp.NDDataset(
        data,
        coordset=[
            scp.Coord(row_coord, title="spectrum index", units=None),
            scp.Coord(x, title="wavenumber", units="cm^-1"),
        ],
        title="synthetic 2D baseline dataset",
        units="absorbance",
    )
    dataset.name = "synthetic_2d"

    return dataset, true_baseline_2d, true_peaks_2d


@pytest.fixture()
def synthetic_ms_like_dataset():
    t = np.linspace(0.0, 50.0, 200)

    true_baseline = 0.5 + 0.02 * t + 0.001 * t**2

    peak1 = 0.8 * np.exp(-((t - 10.0) ** 2) / 1.0)
    peak2 = 1.2 * np.exp(-((t - 25.0) ** 2) / 0.8)
    peak3 = 0.6 * np.exp(-((t - 40.0) ** 2) / 1.5)
    peaks = peak1 + peak2 + peak3

    rng = np.random.default_rng(42)
    noise = 0.02 * rng.normal(size=t.shape)

    data = true_baseline + peaks + noise

    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord(t, title="time", units="minutes")],
        title="synthetic MS ion current",
        units="a.u.",
    )
    dataset.name = "synthetic_ms"

    return dataset, true_baseline, peaks
