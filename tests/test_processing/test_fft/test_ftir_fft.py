# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Regression tests for FFT coordinate generation."""

import numpy as np

import spectrochempy as scp
from spectrochempy.core.units import ur


def _ftir_chain(use_hamming=False):
    ir = scp.read_spa("irdata/interferogram/interfero.SPA")
    source_data = ir.data.copy()
    source_coord = ir.x.data.copy()

    if use_hamming:
        work = ir.dc().hamming()
        work.zf(inplace=True, size=2 * ir.size)
    else:
        work = ir.dc()
        work = work.zf(size=2 * work.size)

    transformed = work.fft()
    return ir, work, transformed, source_data, source_coord


def _assert_ftir_coordinate(dataset):
    x = np.asarray(dataset.x.data, dtype=float)

    assert dataset.x.units == ur("cm^-1")
    assert dataset.x.title in {"wavenumber", "wavenumbers"}
    assert dataset.shape[-1] == dataset.x.size
    assert np.isfinite(x).all()
    assert np.isfinite(dataset.data).all()
    assert np.all(np.diff(x) < 0)
    assert x[0] > 4000.0
    assert abs(x[-1]) < 1.0e-12
    assert x.min() >= -1.0e-12
    assert ((x >= 400.0) & (x <= 4000.0)).sum() > 1000


def _assert_ftir_coordinate_matches_rfft_bins(source, transformed):
    x = np.asarray(transformed.x.data, dtype=float)
    spacing = abs(source.x.spacing)
    expected = np.fft.rfftfreq(source.size)[: transformed.x.size][::-1] / spacing
    expected = scp.Coord(expected).to("cm^-1").data

    assert np.array_equal(x, expected)
    assert x[0] == expected[0]
    assert np.isclose(x[0], 7898.731)
    assert np.isclose(x[0] - x[1], expected[0] - expected[1])
    assert np.isclose(x[0] - x[1], 1.899)


def _compare_to_omnic(dataset):
    omnic = scp.read_spa("irdata/interferogram/spectre.SPA")
    x = np.asarray(dataset.x.data, dtype=float)
    y = np.asarray(dataset.data[0], dtype=float)
    xo = np.asarray(omnic.x.data, dtype=float)
    yo = np.asarray(omnic.data[0], dtype=float)

    lo = max(x.min(), xo.min(), 400.0)
    hi = min(x.max(), xo.max(), 4000.0)
    assert lo < hi

    grid = np.linspace(lo, hi, 2000)
    yi = np.interp(grid, x[np.argsort(x)], y[np.argsort(x)])
    yoi = np.interp(grid, xo[np.argsort(xo)], yo[np.argsort(xo)])
    offset = np.median(yi - yoi)
    residual = yi - offset - yoi
    correlation = np.corrcoef(yi - yi.mean(), yoi - yoi.mean())[0, 1]

    calc_band = grid[np.argmax(yi)]
    omnic_band = grid[np.argmax(yoi)]

    assert correlation > 0.999
    assert np.sqrt(np.mean(residual**2)) < 0.03
    assert abs(calc_band - omnic_band) < 10.0


def test_ftir_interferogram_fft_coordinate_matches_spectrum_window():
    ir, work, transformed, source_data, source_coord = _ftir_chain()

    assert ir.shape == (1, 4160)
    assert work.shape == (1, 8320)
    assert transformed.shape == (1, 4160)
    _assert_ftir_coordinate(transformed)
    _assert_ftir_coordinate_matches_rfft_bins(work, transformed)
    _compare_to_omnic(transformed)
    assert np.array_equal(ir.data, source_data)
    assert np.array_equal(ir.x.data, source_coord)


def test_ftir_interferogram_hamming_fft_coordinate_matches_spectrum_window():
    ir, work, transformed, source_data, source_coord = _ftir_chain(use_hamming=True)

    assert ir.shape == (1, 4160)
    assert work.shape == (1, 8320)
    assert transformed.shape == (1, 4160)
    _assert_ftir_coordinate(transformed)
    _assert_ftir_coordinate_matches_rfft_bins(work, transformed)
    _compare_to_omnic(transformed)
    assert np.array_equal(ir.data, source_data)
    assert np.array_equal(ir.x.data, source_coord)


def test_generic_complex_fft_coordinate_matches_shifted_frequency_order():
    size = 1024
    spacing = 0.001
    bin_index = 64
    frequency = bin_index / (size * spacing)
    time = np.arange(size) * spacing
    data = np.exp(2j * np.pi * frequency * time)
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord(time, units="s")],
        meta={"td": [size], "isfreq": [False]},
    )

    transformed = dataset.fft()
    x = np.asarray(transformed.x.data, dtype=float)
    peak_frequency = x[np.argmax(np.abs(transformed.data))]

    assert transformed.x.units == ur.Hz
    assert transformed.shape[-1] == transformed.x.size
    assert np.isclose(x[0], -1.0 / (2.0 * spacing))
    assert np.isclose(x[-1], 1.0 / (2.0 * spacing) - 1.0 / (size * spacing))
    assert np.all(np.diff(x) > 0)
    assert abs(peak_frequency - frequency) < 1.0 / (size * spacing)


def test_generic_real_fft_coordinate_matches_shifted_frequency_order():
    size = 1024
    spacing = 0.001
    bin_index = 64
    frequency = bin_index / (size * spacing)
    time = np.arange(size) * spacing
    data = np.cos(2 * np.pi * frequency * time)
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord(time, units="s")],
        meta={"td": [size], "isfreq": [False]},
    )

    transformed = dataset.fft()
    x = np.asarray(transformed.x.data, dtype=float)
    peak_frequencies = x[np.argsort(np.abs(transformed.data))[-2:]]

    assert transformed.x.units == ur.Hz
    assert transformed.shape[-1] == transformed.x.size
    assert np.all(np.diff(x) > 0)
    assert np.allclose(np.sort(peak_frequencies), [-frequency, frequency])
