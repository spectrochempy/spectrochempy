"""Regression tests for separable multiplet fitting."""

from copy import deepcopy
from types import MappingProxyType

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import FitResult


def _reference():
    y = np.linspace(3, -3, 41)[:, None]
    t = np.linspace(-1, 1, 51)[None, :]
    x = 4 * t + 0.4 * t**3

    def model(p):
        a, b, c, d, e, f, g, h, i, j = p
        first = (
            a
            * np.exp(-4 * np.log(2) * ((y - b) / c) ** 2)
            / (1 + 4 * ((x - d) / e) ** 2)
        )
        second = (
            f
            * np.exp(-4 * np.log(2) * ((x - i) / j) ** 2)
            / (1 + 4 * ((y - g) / h) ** 2)
        )
        return first + second

    truth = [2, -0.42, 1.1, -0.55, 1.4, 1.3, 0.62, 1.35, 0.75, 1.65]
    data = model(truth) + np.random.default_rng(1849).normal(0, 0.01, (41, 51))
    mask = np.zeros(data.shape, bool)
    mask[15:18, 19:23] = True
    data[2, 3], data[30, 37] = np.nan, np.inf
    valid = ~mask & np.isfinite(data)
    spectrum = scp.NDDataset(
        data,
        coordset=[scp.Coord(y[:, 0], units="ppm"), scp.Coord(x[0], units="Hz")],
        mask=mask,
        units="V",
    )
    initial = [1.8, -0.3, 1.25, -0.4, 1.6, 1.1, 0.5, 1.5, 0.6, 1.8]
    low = [0, -1.2, 0.4, -1.4, 0.4, 0, 0.1, 0.4, 0.1, 0.4]
    high = [4, 0.05, 2.5, 0.1, 2.5, 4, 1.4, 2.5, 1.6, 2.5]

    def param(i):
        return {"value": initial[i], "bounds": (low[i], high[i])}

    peaks = [
        {
            "height": param(i),
            "axes": {
                "y": {"model": y_model, "center": param(i + 1), "width": param(i + 2)},
                "x": {"model": x_model, "center": param(i + 3), "width": param(i + 4)},
            },
        }
        for i, y_model, x_model in [
            (0, "gaussian", "lorentzian"),
            (5, "lorentzian", "gaussian"),
        ]
    ]
    oracle = least_squares(
        lambda p: (data - model(p))[valid],
        initial,
        bounds=(low, high),
        jac="3-point",
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )
    return spectrum, valid, peaks, oracle


def test_fit_nd_reference_and_permuted_axes():
    spectrum, valid, peaks, oracle = _reference()
    initial = deepcopy(peaks)
    opt = scp.Optimize(peaks=peaks)
    assert opt.fit(spectrum) is opt
    result = opt.result
    assert isinstance(result, FitResult)
    assert result.diagnostics["success"]
    assert result.diagnostics["n_observations"] == 2077
    assert result.diagnostics["degrees_of_freedom"] == 2067
    rss = oracle.fun @ oracle.fun
    assert_allclose(result.diagnostics["rss"], rss, rtol=1e-5)
    assert opt.jacobian.shape == (2077, 10)
    assert not opt.jacobian.flags.writeable
    order = [0, 3, 4, 1, 2, 5, 8, 9, 6, 7]
    actual = [
        value
        for peak in result.peaks
        for value in [
            peak["height"],
            peak["axes"]["x"]["center"],
            peak["axes"]["x"]["width"],
            peak["axes"]["y"]["center"],
            peak["axes"]["y"]["width"],
        ]
    ]
    assert_allclose(actual, oracle.x[order], atol=2e-5, rtol=0)
    j = oracle.jac[:, order]
    assert_allclose(opt.jacobian, j, atol=3e-5, rtol=1e-4)
    expected = (rss / 2067) * np.linalg.inv(j.T @ j)
    stderr = np.sqrt(np.diag(expected))
    assert_allclose(result.stderr, stderr, rtol=0.01)
    assert_allclose(
        (result.covariance - expected) / np.outer(stderr, stderr), 0, atol=0.02
    )
    assert result.confidence_intervals.shape == (10, 2)
    assert result.parameter_names[1] == "peaks[0].axes['x'].center"
    assert result.parameter_units[:5] == (
        scp.ur.V,
        scp.ur.Hz,
        scp.ur.Hz,
        scp.ur.ppm,
        scp.ur.ppm,
    )
    for part in (result.fitted, result.residuals, *result.components):
        assert part.dims == spectrum.dims
        assert part.units == spectrum.units
        assert_array_equal(part.mask, ~valid)
        assert_array_equal(part.x._data, spectrum.x._data)
        assert part.history_entries[-1]["operation"] == "optimize"
    assert_allclose(sum(part.data for part in result.components), result.fitted.data)
    assert_allclose(scp.simulate_peaks(spectrum, result.peaks).data, result.fitted.data)
    assert_allclose(result.residuals.data[valid], oracle.fun, atol=2e-5, rtol=0)
    assert peaks == initial
    assert np.count_nonzero(spectrum.mask) == 12

    permuted_spectrum = scp.NDDataset(
        spectrum.data[::-1, ::-1].T,
        dims=list(reversed(spectrum.dims)),
        coordset=[
            scp.Coord(spectrum.x._data[::-1], units=spectrum.x.units),
            scp.Coord(spectrum.y._data[::-1], units=spectrum.y.units),
        ],
        units=spectrum.units,
        mask=spectrum.mask[::-1, ::-1].T,
    )
    permuted = scp.Optimize(peaks=peaks).fit(permuted_spectrum).result
    assert_allclose(permuted.fitted.data.T[::-1, ::-1], result.fitted.data, atol=2e-7)
    assert_allclose(permuted.covariance, result.covariance, rtol=1e-4, atol=1e-11)


def _fixed_peak(dims):
    return {
        "height": {"value": -2, "fixed": True},
        "axes": {
            dim: {
                "model": "gaussian",
                "center": {"value": 0, "fixed": True},
                "width": {"value": 1, "fixed": True},
            }
            for dim in dims
        },
    }


def test_fit_nd_fixed_3d_units_and_failed_refit():
    coords = [scp.Coord.linspace(-3, 3, 20, units="Hz") for _ in range(3)]
    grid = scp.NDDataset(np.zeros((20, 20, 20)), coordset=coords, units="V")
    peak = _fixed_peak(grid.dims)
    peak["height"]["value"] = -2000 * scp.ur.mV
    peak["axes"]["x"]["width"]["value"] = 0.001 * scp.ur.kHz
    truth = {
        "height": -2,
        "axes": {
            dim: {"model": "gaussian", "center": 0, "width": 1} for dim in grid.dims
        },
    }
    spectrum = scp.simulate_peaks(grid, [truth])
    opt = scp.Optimize(peaks=[peak]).fit(spectrum)
    assert_array_equal(opt.predict().data, spectrum.data)
    assert opt.result.diagnostics["covariance_reason"] == "no_free_parameters"
    assert opt.result.covariance is None
    assert opt.jacobian is None
    peak["height"] = -1000 * scp.ur.mV
    free_height = scp.Optimize(peaks=[peak]).fit(spectrum)
    assert_allclose(free_height.result.peaks[0]["height"], -2)
    assert free_height.jacobian.shape == (spectrum.size, 1)
    with pytest.raises(NotImplementedError, match="projections"):
        opt.transform()
    with pytest.raises(ValueError, match="at least two"):
        opt.fit(spectrum[0, 0].squeeze())
    with pytest.raises(NotFittedError):
        _ = opt.result


@pytest.mark.parametrize("mode", ["active_bounds", "rank_deficient", "not_converged"])
def test_fit_nd_covariance_unavailable(mode):
    coords = [scp.Coord.linspace(-2, 2, 9) for _ in range(2)]
    grid = scp.NDDataset(np.zeros((9, 9)), coordset=coords)
    peak = _fixed_peak(grid.dims)
    truth = {
        "height": -2,
        "axes": {
            dim: {"model": "gaussian", "center": 0, "width": 1} for dim in grid.dims
        },
    }
    spectrum = scp.simulate_peaks(grid, [truth])
    peak["height"] = {"value": -0.5, "bounds": (-1, 0)}
    peaks, kwargs = [peak], {}
    if mode == "rank_deficient":
        peak["height"] = -0.6
        peaks.append(peak)
    elif mode == "not_converged":
        kwargs["max_fun_calls"] = 1
    result = scp.Optimize(peaks=peaks, **kwargs).fit(spectrum).result
    assert result.diagnostics["covariance_reason"] == mode
    assert result.covariance is None
    assert result.confidence_intervals is None


def test_legacy_optimize_singleton_matches_1d():
    x = np.linspace(-4, 4, 101)
    signal = scp.gaussian(x, pos=0.23, width=1.5, normalized=False)
    signal += np.random.default_rng(8).normal(0, 0.001, x.size)
    coord = scp.Coord(x, units="Hz")
    spectra = [
        scp.NDDataset(signal, coordset=[coord]),
        scp.NDDataset(signal[None, :], coordset=[scp.Coord([0]), coord.copy()]),
    ]
    script = """
MODEL: line
shape: gaussianmodel
$ ampl: 0.9, 0, 2
$ pos: 0.1, -2, 2
$ width: 1.0, .2, 4
"""
    fit, singleton = (scp.Optimize(script=script).fit(spectrum) for spectrum in spectra)
    assert fit.script == singleton.script
    assert dict(fit.fp) == dict(singleton.fp)
    assert_allclose(
        fit.result.fitted.data.squeeze(), singleton.result.fitted.data.squeeze()
    )
    assert_allclose(
        fit.result.residuals.data.squeeze(), singleton.result.residuals.data.squeeze()
    )
    assert_array_equal(
        np.asarray(fit.result.fitted.mask).squeeze(),
        np.asarray(singleton.result.fitted.mask).squeeze(),
    )
    assert fit.result.diagnostics == singleton.result.diagnostics
    assert_allclose(fit.result.covariance, singleton.result.covariance)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_axis", "every dataset dimension"),
        ("bad_width", "lower bound must be positive"),
        ("out_of_bounds", "initial value must lie"),
        ("no_observations", "requires observations"),
        ("bad_grid", "strictly monotone"),
        ("complex", "real intensities"),
    ],
)
def test_fit_nd_rejects_invalid_inputs(case, message):
    coords = [scp.Coord.linspace(-2, 2, 9) for _ in range(2)]
    data = np.zeros((9, 9))
    mask = case == "no_observations"
    if case == "bad_grid":
        coords[-1] = scp.Coord(np.zeros(9))
    if case == "complex":
        data = data.astype(complex) + 1j
    spectrum = scp.NDDataset(data, coordset=coords, mask=np.full(data.shape, mask))
    peak = _fixed_peak(spectrum.dims)
    if case == "missing_axis":
        del peak["axes"]["x"]
    elif case == "bad_width":
        peak["axes"]["x"]["width"] = 1
    elif case == "out_of_bounds":
        peak["height"] = {"value": 2, "bounds": (-3, -1)}
    with pytest.raises(ValueError, match=message):
        scp.Optimize(peaks=[peak]).fit(spectrum)


@pytest.mark.parametrize(
    ("height", "bounds", "reason"),
    [
        (2e-9, (0, 4e-9), None),
        (2e-9, (0, 4), None),
        (0.999, (1, 4), "active_bounds"),
        (1.0, (1, 4), "active_bounds"),
        (1.000001, (1, 4), None),
        (1.000000001, (1, 4), None),
    ],
)
def test_fit_nd_height_scale_and_active_bounds(height, bounds, reason):
    coords = [scp.Coord.linspace(-1, 1, 11) for _ in range(2)]
    grid = scp.NDDataset(np.zeros((11, 11)), coordset=coords)
    peak = _fixed_peak(grid.dims)
    truth = {
        "height": height,
        "axes": {
            dim: {"model": "gaussian", "center": 0, "width": 1} for dim in grid.dims
        },
    }
    start = height / 2 if height < 1e-6 else 1.5
    peak["height"] = {"value": start, "bounds": bounds}
    result = scp.Optimize(peaks=[peak]).fit(scp.simulate_peaks(grid, [truth])).result
    assert result.diagnostics["success"]
    assert result.diagnostics["covariance_reason"] == reason
    assert_allclose(
        result.peaks[0]["height"], max(height, bounds[0]), rtol=1e-5, atol=0
    )
    if reason is None:
        assert_allclose(result.diagnostics["r_squared"], 1)
    else:
        assert result.covariance is None


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e9])
def test_fit_nd_recentered_coordinate(offset):
    x = scp.Coord(offset + np.linspace(-2, 2, 21), units="Hz")
    y = scp.Coord.linspace(-2, 2, 21, units="Hz")
    grid = scp.NDDataset(np.zeros((21, 21)), coordset=[y, x])
    truth = {
        "height": -2,
        "axes": {
            "x": {"model": "gaussian", "center": offset + 0.24, "width": 1},
            "y": {"model": "gaussian", "center": 0, "width": 1},
        },
    }
    peak = _fixed_peak(grid.dims)
    peak["axes"]["x"]["center"] = offset + 0.1
    opt = scp.Optimize(peaks=[peak]).fit(scp.simulate_peaks(grid, [truth]))
    assert opt.result.diagnostics["success"]
    assert_allclose(
        opt.result.peaks[0]["axes"]["x"]["center"] - offset, 0.24, atol=2e-7
    )
    assert opt.result.diagnostics["rss"] < 1e-12


@pytest.mark.parametrize("height", [1e155, 1e-170])
def test_fit_nd_diagnostics_scale_squared_residuals(height):
    coords = [scp.Coord.linspace(-2, 2, 5) for _ in range(2)]
    spectrum = scp.NDDataset(np.full((5, 5), height), coordset=coords)
    peak = _fixed_peak(spectrum.dims)
    peak["height"]["value"] = 0
    diagnostics = scp.Optimize(peaks=[peak]).fit(spectrum).result.diagnostics
    assert_allclose(diagnostics["rmse"], height, rtol=1e-14, atol=0)
    assert_allclose(diagnostics["aic"], 50 * np.log(height))
    assert diagnostics["rss_status"] == "numeric_range"
    if height > 1:
        assert np.isposinf(diagnostics["rss"])


def test_fit_nd_preserves_large_fixed_integer_center():
    offset = 2**53
    x = scp.Coord(offset + np.arange(-4, 8, 2), dtype=np.int64)
    y = scp.Coord(np.arange(-2, 3))
    grid = scp.NDDataset(np.zeros((5, 6)), coordset=[y, x])
    truth = {
        "height": 1,
        "axes": {
            "x": {"model": "gaussian", "center": offset + 1, "width": 1},
            "y": {"model": "gaussian", "center": 0, "width": 1},
        },
    }
    peak = _fixed_peak(grid.dims)
    peak["height"]["value"] = 1
    peak["axes"]["x"]["center"]["value"] = offset + 1
    spectrum = scp.simulate_peaks(grid, [truth])
    opt = scp.Optimize(peaks=[peak]).fit(spectrum)
    assert opt.result.peaks[0]["axes"]["x"]["center"] == offset + 1
    assert_array_equal(opt.predict().data, spectrum.data)
    assert_allclose(opt.predict().data[2, 2:4], [0.0625, 0.0625], atol=0)


def test_fit_nd_large_integer_origin_preserves_center_derivative():
    offset = 2**53
    x = scp.Coord(offset + np.arange(-4, 8, 2), dtype=np.int64)
    y = scp.Coord(np.arange(-2, 3))
    grid = scp.NDDataset(np.zeros((5, 6)), coordset=[y, x])
    truth = {
        "height": 1,
        "axes": {
            "x": {"model": "gaussian", "center": offset + 1, "width": 2},
            "y": {"model": "gaussian", "center": 0, "width": 2},
        },
    }
    spectrum = scp.simulate_peaks(grid, [truth])
    peak = _fixed_peak(grid.dims)
    peak["height"]["value"] = 1
    peak["axes"]["x"]["center"] = offset
    for dim in grid.dims:
        peak["axes"][dim]["width"]["value"] = 2
    opt = scp.Optimize(peaks=[peak]).fit(spectrum)
    assert opt.result.diagnostics["success"]
    assert opt.result.peaks[0]["axes"]["x"]["center"] == offset + 1
    assert_allclose(np.linalg.norm(opt.jacobian), 1.2037712, rtol=1e-6)
    assert_array_equal(opt.predict().data, spectrum.data)


def test_fit_nd_resolves_weak_peak_and_accepts_read_only_mappings():
    grid = scp.NDDataset(
        np.zeros((5, 21)),
        coordset=[scp.Coord(np.arange(-2, 3)), scp.Coord(np.arange(-10, 11))],
    )

    def peak(height, center):
        return {
            "height": height,
            "axes": {
                "x": {"model": "gaussian", "center": center, "width": 0.25},
                "y": {"model": "gaussian", "center": 0, "width": 0.25},
            },
        }

    spectrum = scp.simulate_peaks(grid, [peak(1, -5), peak(2e-11, 5)])
    specs = [peak(1, -5), peak(1e-11, 5)]
    specs[0]["height"] = {"value": 1, "fixed": True}
    for spec in specs:
        for dim, axis in spec["axes"].items():
            for key in ("center", "width"):
                axis[key] = MappingProxyType({"value": axis[key], "fixed": True})
            spec["axes"][dim] = MappingProxyType(axis)
        spec["axes"] = MappingProxyType(spec["axes"])
    opt = scp.Optimize(peaks=tuple(map(MappingProxyType, specs))).fit(spectrum)
    assert opt.result.diagnostics["success"]
    assert_allclose(opt.result.peaks[1]["height"], 2e-11, rtol=1e-8, atol=0)


def test_fit_nd_confidence_interval_contains_extended_precision_center():
    offset = 10**12
    center = np.longdouble(offset) + np.longdouble("0.333333333")
    x = scp.Coord(offset + np.arange(-4, 8, 2), dtype=np.int64)
    y = scp.Coord(np.arange(-2, 3))
    grid = scp.NDDataset(np.zeros((5, 6)), coordset=[y, x])
    truth = {
        "height": 1,
        "axes": {
            "x": {"model": "gaussian", "center": center, "width": 2},
            "y": {"model": "gaussian", "center": 0, "width": 2},
        },
    }
    spectrum = scp.simulate_peaks(grid, [truth])
    spectrum.data[0, -1] += 1e-7
    peak = _fixed_peak(grid.dims)
    peak["height"]["value"] = 1
    peak["axes"]["x"]["center"] = center
    for dim in grid.dims:
        peak["axes"][dim]["width"]["value"] = 2
    result = scp.Optimize(peaks=[peak]).fit(spectrum).result
    assert result.diagnostics["success"]
    assert result.stderr[0] > 0
    low, high = result.confidence_intervals[0]
    assert low <= result.peaks[0]["axes"]["x"]["center"] <= high
