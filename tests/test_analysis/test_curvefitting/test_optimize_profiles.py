"""Global spectral/kinetic fits with user profiles and shared parameters."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.optimize import least_squares
from scipy.special import voigt_profile

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError


def _stretched_recovery(t, tau, beta):
    return -np.expm1(-((t / tau) ** beta))


def _grid():
    t = np.array([0, 0.3, 0.8, 1.6, 3, 6, 12, 30.0])
    x = np.linspace(-8, 6, 81)[::-1]
    return scp.NDDataset(
        np.zeros((t.size, x.size)),
        coordset=[scp.Coord(t, units="s"), scp.Coord(x, units="Hz")],
        units="V",
    )


def _gaussian(x, center, width):
    return np.exp(-4 * np.log(2) * ((x - center) / width) ** 2)


def test_global_four_bands_with_shared_stretched_recovery():
    grid = _grid()
    truth = [(1.2, -5, 1.4, 1), (2, -2, 1.8, 2), (1.5, 1, 1.2, 4), (0.8, 4, 1.5, 8)]
    expected = sum(
        height
        * _stretched_recovery(grid.y.data[:, None], tau, 0.85)
        * _gaussian(grid.x.data[None, :], center, width)
        for height, center, width, tau in truth
    )
    noise = np.random.default_rng(1945).normal(0, 0.002, grid.shape)
    observed = grid.copy()
    observed.data = expected + noise
    mask = np.zeros(grid.shape, dtype=bool)
    mask[2, 6:10] = True
    observed.mask = mask
    observed.data[3, 2] = np.nan
    peaks = [
        {
            "height": {"value": height * 0.9, "bounds": (0, 4)},
            "axes": {
                "x": {
                    "model": "gaussian",
                    "parameters": {
                        "center": center + 0.1,
                        "width": {"value": width * 1.1, "bounds": (0.3, 4)},
                    },
                },
                "y": {
                    "model": _stretched_recovery,
                    "parameters": {
                        "tau": {"value": tau * 0.9, "bounds": (0.1, 20), "unit": "s"},
                        "beta": {"shared": "beta"},
                    },
                },
            },
        }
        for height, center, width, tau in truth
    ]
    shared = {"beta": {"value": 1, "bounds": (0.3, 1.5)}}
    initial = deepcopy(peaks)
    opt = scp.Optimize(peaks=peaks, shared_parameters=shared).fit(observed)
    result = opt.result
    assert result.diagnostics["success"]
    assert result.diagnostics["n_varying_parameters"] == 17
    assert result.diagnostics["n_observations"] == grid.size - 5
    beta_index = result.parameter_names.index("shared_parameters['beta']")
    assert_allclose(
        result.peaks[0]["axes"]["y"]["parameters"]["beta"].magnitude, 0.85, atol=0.004
    )
    for part, (_, center, width, tau) in zip(result.peaks, truth, strict=True):
        assert_allclose(part["axes"]["x"]["parameters"]["center"], center, atol=0.01)
        assert_allclose(part["axes"]["x"]["parameters"]["width"], width, atol=0.015)
        assert_allclose(
            part["axes"]["y"]["parameters"]["tau"].to("s").magnitude, tau, rtol=0.015
        )
    assert result.parameter_units[beta_index].dimensionless
    assert result.covariance is not None
    assert opt.jacobian.shape == (grid.size - 5, 17)
    assert not opt.jacobian.flags.writeable
    assert_array_equal(result.fitted.mask, mask | ~np.isfinite(observed.data))
    assert_allclose(sum(part.data for part in result.components), result.fitted.data)
    assert_allclose(scp.simulate_peaks(grid, result.peaks).data, result.fitted.data)
    assert peaks == initial
    permuted = (
        scp.Optimize(peaks=peaks, shared_parameters=shared).fit(observed.T).result
    )
    assert_allclose(permuted.fitted.data.T, result.fitted.data, atol=2e-7)
    beta_columns = []
    for part in result.peaks:
        tau = part["axes"]["y"]["parameters"]["tau"].magnitude
        beta = part["axes"]["y"]["parameters"]["beta"].magnitude
        xparams = part["axes"]["x"]["parameters"]
        z = (grid.y.data[:, None] / tau) ** beta
        logt = np.zeros_like(z)
        np.log(grid.y.data[:, None] / tau, where=grid.y.data[:, None] > 0, out=logt)
        beta_columns.append(
            -part["height"]
            * np.exp(-z)
            * z
            * logt
            * _gaussian(grid.x.data[None, :], **xparams)
        )
    valid = ~result.fitted.mask
    assert_allclose(opt.jacobian[:, beta_index], sum(beta_columns)[valid], atol=2e-7)


@pytest.mark.parametrize("ratio_scale", [1, 100])
def test_voigt_and_shared_time_match_independent_global_solver(ratio_scale):
    grid = _grid()
    x, t = grid.x.data[None, :], grid.y.data[:, None]

    def line(center, width, ratio):
        sigma = ratio * width / (2 * np.sqrt(2 * np.log(2)))
        gamma = (1 - ratio) * width / 2
        return voigt_profile(x - center, sigma, gamma) / voigt_profile(0, sigma, gamma)

    def model(p):
        a, b, tau, ratio = p
        return (-np.expm1(-t / tau)) * (
            a * line(-2, 2, ratio) + b * line(2, 1.5, ratio)
        )

    data = model([2, 1, 2.5, 0.6]) + np.random.default_rng(42).normal(
        0, 0.003, grid.shape
    )
    observed = grid.copy()
    observed.data = data
    peaks = [
        {
            "height": {"value": amp, "bounds": (0, 4)},
            "axes": {
                "x": {
                    "model": "voigt",
                    "center": {"value": center, "fixed": True},
                    "width": {"value": width, "fixed": True},
                    "ratio": {"shared": "ratio"},
                },
                "y": {
                    "model": "saturation_recovery",
                    "parameters": {"tau": {"shared": "tau"}},
                },
            },
        }
        for amp, center, width in [(1.8, -2, 2), (0.8, 2, 1.5)]
    ]
    shared = {
        "ratio": {
            "value": 0.5 * ratio_scale,
            "bounds": (0, ratio_scale),
            "unit": "percent" if ratio_scale == 100 else "dimensionless",
        },
        "tau": {"value": 2000 * scp.ur.ms, "unit": "ms", "bounds": (100, 20000)},
    }
    opt = scp.Optimize(peaks=peaks, shared_parameters=shared).fit(observed)
    result = opt.result
    oracle = least_squares(
        lambda p: (data - model(p)).ravel(),
        [1.8, 0.8, 2, 0.5],
        bounds=([0, 0, 0.1, 0], [4, 4, 20, 1]),
        jac="3-point",
    )
    assert_allclose(result.fitted.data, model(oracle.x), atol=3e-6)
    assert result.diagnostics["n_varying_parameters"] == 4
    assert (
        result.parameter_units[result.parameter_names.index("shared_parameters['tau']")]
        == scp.ur.ms
    )
    assert_allclose(
        result.peaks[0]["axes"]["y"]["parameters"]["tau"], oracle.x[2], atol=3e-5
    )
    assert_allclose(scp.simulate_peaks(grid, result.peaks).data, result.fitted.data)
    names = result.parameter_names
    order = [0, 3, 2, 1]
    factors = np.array([1, 1 / ratio_scale, 0.001, 1])
    assert_allclose(opt.jacobian, oracle.jac[:, order] * factors, atol=2e-6)
    assert "shared_parameters['ratio']" in names


@pytest.mark.parametrize(
    "model",
    [
        "gaussian",
        "lorentzian",
        "voigt",
        "pseudovoigt",
        "saturation_recovery",
        "exponential_decay",
    ],
)
def test_builtin_profiles_simulation_and_all_fixed_fit(model):
    grid = _grid()
    xparameters = {"center": -1, "width": 2}
    if model in ("saturation_recovery", "exponential_decay"):
        axis = {"model": model, "parameters": {"tau": 3}}
        xmodel = "gaussian"
    else:
        xmodel = model
        axis = {"model": "saturation_recovery", "parameters": {"tau": 3}}
    if xmodel in ("voigt", "pseudovoigt"):
        xparameters["ratio"] = 0.5
    truth = [
        {
            "height": 2,
            "axes": {"x": {"model": xmodel, "parameters": xparameters}, "y": axis},
        }
    ]
    clean = scp.simulate_peaks(grid, truth)
    fit_peaks = deepcopy(truth)
    fit_peaks[0]["height"] = {"value": 2, "fixed": True}
    for item in fit_peaks[0]["axes"].values():
        item["parameters"] = {
            key: {"value": value, "fixed": True}
            for key, value in item["parameters"].items()
        }
    opt = scp.Optimize(peaks=fit_peaks).fit(clean)
    assert_allclose(opt.predict().data, clean.data)
    assert opt.result.diagnostics["covariance_reason"] == "no_free_parameters"
    assert opt.result.parameter_names == ()


def test_custom_profile_units_and_history_roundtrip(tmp_path):
    def custom(x, center, width):
        return scp.NDDataset(_gaussian(x, center, width), units="dimensionless")

    grid = _grid()
    peak = {
        "height": 2,
        "axes": {
            "x": {
                "model": custom,
                "parameters": {"center": -2 * scp.ur.Hz, "width": 2 * scp.ur.Hz},
            },
            "y": {"model": "saturation_recovery", "parameters": {"tau": 3 * scp.ur.s}},
        },
    }
    clean = scp.simulate_peaks(grid, [peak])
    path = clean.write(tmp_path / "callback.scp", overwrite=True)
    restored = scp.NDDataset.load(path)
    assert_allclose(restored.data, clean.data)
    history = restored.history_entries[-1]["parameters"]["peaks"]
    assert "custom" in history[0]["axes"]["x"]["model"]


@pytest.mark.parametrize(
    "callback",
    [
        lambda x: np.zeros((x.size, 1)),
        lambda x: np.ones(x.size, dtype=complex),
        lambda x: np.full(x.size, np.nan),
        lambda x: scp.NDDataset(np.ones(x.size), units="V"),
    ],
)
def test_invalid_callback_output_rejected(callback):
    grid = _grid()
    peaks = [
        {
            "height": {"value": 1, "fixed": True},
            "axes": {
                "x": {"model": callback, "parameters": {}},
                "y": {
                    "model": "saturation_recovery",
                    "parameters": {"tau": {"value": 2, "fixed": True}},
                },
            },
        }
    ]
    with pytest.raises(ValueError):
        scp.Optimize(peaks=peaks).fit(grid)


@pytest.mark.parametrize(
    "shared,reference",
    [
        ({}, "missing"),
        ({"unused": 1}, "missing"),
        ({"tau": {"value": 2, "unit": "Hz"}}, "tau"),
    ],
)
def test_invalid_shared_references_and_units(shared, reference):
    peak = {
        "height": 1,
        "axes": {
            "x": {
                "model": "gaussian",
                "parameters": {"center": 0, "width": {"value": 2, "bounds": (0.1, 4)}},
            },
            "y": {
                "model": "saturation_recovery",
                "parameters": {"tau": {"shared": reference}},
            },
        },
    }
    with pytest.raises(ValueError):
        scp.Optimize(peaks=[peak], shared_parameters=shared).fit(_grid())


def test_failed_refit_invalidates_general_fit():
    peak = {
        "height": {"value": 1, "fixed": True},
        "axes": {
            "x": {"model": lambda x: np.ones_like(x), "parameters": {}},
            "y": {"model": lambda x: np.ones_like(x), "parameters": {}},
        },
    }
    opt = scp.Optimize(peaks=[peak]).fit(_grid())
    grid = _grid()
    grid.mask = np.ones(grid.shape, bool)
    with pytest.raises(ValueError):
        opt.fit(grid)
    with pytest.raises(NotFittedError):
        _ = opt.result


@pytest.mark.parametrize("quantity", [False, True])
def test_shared_parameter_across_axes_with_different_units(quantity):
    def decay(t, tau):
        return np.exp(-t / tau)

    z = np.array([0, 500, 2500, 7000.0])
    y = np.array([0, 1, 3, 9.0])
    x = np.linspace(-3, 3, 9)
    values = (
        np.exp(-z[:, None, None] / 3000)
        * np.exp(-y[None, :, None] / 3)
        * _gaussian(x[None, None, :], 0, 2)
    )
    data = scp.NDDataset(
        values,
        coordset=[
            scp.Coord(z, units="ms"),
            scp.Coord(y, units="s"),
            scp.Coord(x, units="Hz"),
        ],
    )
    peak = {
        "height": {"value": 1, "fixed": True},
        "axes": {
            "z": {
                "model": decay,
                "parameters": {"tau": {"shared": "time", "unit": "ms"}},
            },
            "y": {
                "model": decay,
                "parameters": {"tau": {"shared": "time", "unit": "s"}},
            },
            "x": {
                "model": "gaussian",
                "parameters": {
                    "center": {"value": 0, "fixed": True},
                    "width": {"value": 2, "fixed": True},
                },
            },
        },
    }
    time = (
        {"value": 2 * scp.ur.s, "bounds": (0.1 * scp.ur.s, 10 * scp.ur.s)}
        if quantity
        else {"value": 2, "bounds": (0.1, 10), "unit": "s"}
    )
    fit = scp.Optimize(
        peaks=[peak],
        shared_parameters={
            "time": time,
        },
    ).fit(data)
    assert fit.result.diagnostics["n_varying_parameters"] == 1
    assert_allclose(fit.predict().data, values, atol=1e-8)
    params = fit.result.peaks[0]["axes"]
    assert_allclose(params["z"]["parameters"]["tau"].magnitude, 3000, rtol=1e-7)
    assert_allclose(params["y"]["parameters"]["tau"].magnitude, 3, rtol=1e-7)
    assert_allclose(scp.simulate_peaks(data, fit.result.peaks).data, values, atol=1e-8)


def test_reject_unused_shared_parameter():
    peak = {
        "height": 1,
        "axes": {
            "x": {"model": lambda x: np.ones_like(x), "parameters": {}},
            "y": {"model": lambda x: np.ones_like(x), "parameters": {}},
        },
    }
    with pytest.raises(ValueError, match="Every shared parameter"):
        scp.Optimize(peaks=[peak], shared_parameters={"unused": 1}).fit(_grid())


def test_voigt_normalization_does_not_depend_on_sampling():
    grid = _grid()
    peak = {
        "height": 2,
        "axes": {
            "x": {
                "model": "voigt",
                "parameters": {"center": 0.123, "width": 2, "ratio": 0.7},
            },
            "y": {"model": lambda y: np.ones_like(y), "parameters": {}},
        },
    }
    sampled = scp.simulate_peaks(grid, [peak])
    sigma = 0.7 * 2 / np.sqrt(8 * np.log(2))
    gamma = 0.3
    oracle = (
        2
        * voigt_profile(grid.x.data - 0.123, sigma, gamma)
        / voigt_profile(0, sigma, gamma)
    )
    assert_allclose(sampled.data, np.broadcast_to(oracle, grid.shape))
    assert sampled.data.max() < 2


def test_shared_dimensionful_coefficient_requires_intensity_units():
    grid = _grid()
    grid.units = None
    peak = {
        "height": {"shared": "amplitude"},
        "axes": {
            "x": {"model": lambda x: np.ones_like(x), "parameters": {}},
            "y": {"model": lambda x: np.ones_like(x), "parameters": {}},
        },
    }
    with pytest.raises(ValueError, match="units"):
        scp.Optimize(
            peaks=[peak],
            shared_parameters={
                "amplitude": {"value": 1, "unit": "Hz"},
            },
        ).fit(grid)


def test_shared_callback_quantity_infers_unit_without_declaration():
    grid = _grid()
    truth = 2 * _stretched_recovery(grid.y.data[:, None], 3, 1)
    truth = truth * _gaussian(grid.x.data[None, :], 0, 2)
    grid.data = truth
    peak = {
        "height": {"value": 2, "fixed": True},
        "axes": {
            "x": {
                "model": "gaussian",
                "center": {"value": 0, "fixed": True},
                "width": {"value": 2, "fixed": True},
            },
            "y": {
                "model": _stretched_recovery,
                "parameters": {
                    "tau": {"shared": "time"},
                    "beta": {"value": 1, "fixed": True},
                },
            },
        },
    }
    fit = scp.Optimize(
        peaks=[peak],
        shared_parameters={
            "time": {"value": 2 * scp.ur.s, "bounds": (0.1 * scp.ur.s, 10 * scp.ur.s)},
        },
    ).fit(grid)
    assert fit.result.parameter_units == (scp.ur.s,)
    assert_allclose(fit.predict().data, truth, atol=1e-8)
    assert_allclose(
        fit.result.peaks[0]["axes"]["y"]["parameters"]["tau"].magnitude, 3, rtol=1e-7
    )


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("shared_center", [False, True])
def test_general_gaussian_center_is_independent_of_coordinate_origin(
    bounded, shared_center
):
    origin = 1e9
    x = origin + np.linspace(-2, 2, 21)
    y = np.linspace(-2, 2, 21)
    center = origin + 0.24
    values = 2 * _gaussian(x[None, :], center, 0.9) * _gaussian(y[:, None], 0, 1.1)
    grid = scp.NDDataset(
        values,
        coordset=[
            scp.Coord(y, units="Hz", rounding=False),
            scp.Coord(x, units="Hz", rounding=False),
        ],
    )
    parameter = {"value": origin + 0.1}
    if bounded:
        parameter["bounds"] = (origin - 1, origin + 1)
    peak = {
        "height": {"value": 2, "fixed": True},
        "axes": {
            "x": {
                "model": "gaussian",
                "parameters": {
                    "center": {"shared": "position"} if shared_center else parameter,
                    "width": {"value": 0.9, "fixed": True},
                },
            },
            "y": {
                "model": "gaussian",
                "center": {"value": 0, "fixed": True},
                "width": {"value": 1.1, "fixed": True},
            },
        },
    }
    fit = scp.Optimize(
        peaks=[peak], shared_parameters={"position": parameter} if shared_center else {}
    ).fit(grid)
    assert_allclose(fit.predict().data, values, atol=1e-8)
    fitted = fit.result.peaks[0]["axes"]["x"]["parameters"]["center"]
    assert abs(fitted - center) < 1e-8
    jacobian = -values * 8 * np.log(2) * (x[None, :] - center) / 0.9**2
    assert_allclose(fit.jacobian[:, 0], jacobian.ravel(), atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-9])
def test_kinetic_jacobian_and_covariance_are_invariant_to_time_scale(scale):
    t = np.linspace(0, 10, 41)
    noise = np.random.default_rng(42).normal(0, 0.001, (41, 3))
    values = np.exp(-t[:, None] / 3) * np.ones((1, 3)) + noise
    grid = scp.NDDataset(
        values,
        coordset=[
            scp.Coord(t * scale, units="s", rounding=False),
            scp.Coord([0, 1, 2]),
        ],
    )
    peak = {
        "height": {"value": 1, "fixed": True},
        "axes": {
            "x": {"model": lambda x: np.ones_like(x), "parameters": {}},
            "y": {
                "model": "exponential_decay",
                "parameters": {
                    "tau": {"value": 2 * scale, "bounds": (0.1 * scale, 20 * scale)}
                },
            },
        },
    }
    oracle = least_squares(
        lambda p: (values - np.exp(-t[:, None] / p[0])).ravel(),
        [2.0],
        bounds=([0.1], [20]),
        jac="3-point",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    fit = scp.Optimize(peaks=[peak]).fit(grid)
    tau = fit.result.peaks[0]["axes"]["y"]["parameters"]["tau"]
    assert_allclose(tau, oracle.x[0] * scale, rtol=2e-7)
    expected_jacobian = (-fit.predict().data * (t * scale)[:, None] / tau**2).ravel()
    assert_allclose(fit.jacobian[:, 0], expected_jacobian, atol=3e-9 / scale, rtol=2e-6)
    variance = (oracle.fun @ oracle.fun) / (values.size - 1)
    covariance = variance / (oracle.jac[:, 0] @ oracle.jac[:, 0]) * scale**2
    assert_allclose(fit.result.covariance[0, 0], covariance, rtol=2e-6)


@pytest.mark.parametrize("mixed", [False, True])
def test_general_profiles_preserve_representable_gaussian_tails(mixed):
    grid = scp.NDDataset(
        np.zeros((2, 3)),
        coordset=[scp.Coord([0.0, 1.0]), scp.Coord([16.0, 17.0, 18.0])],
    )
    peak = {
        "height": 1e300,
        "axes": {
            "x": {"model": "gaussian", "center": 0, "width": 1},
            "y": {"model": "gaussian", "center": 0, "width": 100},
        },
    }
    expected = scp.simulate_peaks(grid, [peak])
    peak["axes"]["x"] = {"model": "gaussian", "parameters": {"center": 0, "width": 1}}
    if mixed:
        peak["axes"]["y"] = {"model": lambda y: _gaussian(y, 0, 100), "parameters": {}}
    actual = scp.simulate_peaks(grid, [peak])
    assert (actual.data > 0).all()
    assert_allclose(actual.data, expected.data, rtol=3e-13, atol=0)
    fixed = {
        "height": {"value": 1e300, "fixed": True},
        "axes": {
            "x": {
                "model": "gaussian",
                "parameters": {
                    "center": {"value": 0, "fixed": True},
                    "width": {"value": 1, "fixed": True},
                },
            },
            "y": {"model": lambda y: _gaussian(y, 0, 100), "parameters": {}},
        },
    }
    fit = scp.Optimize(peaks=[fixed]).fit(expected)
    assert_allclose(fit.predict().data, expected.data, rtol=3e-13, atol=0)


@pytest.mark.parametrize("noise", [0, 0.01])
def test_general_solver_scales_grid_before_center_subtraction(noise):
    x = scp.Coord([-1e308, 1, 2, 1e308], linearize_below=-1)
    y = scp.Coord(np.arange(-2, 3))
    grid = scp.NDDataset(np.zeros((5, 4)), coordset=[y, x])
    truth = {
        "height": 1,
        "axes": {
            "x": {"model": "lorentzian", "center": 1e308, "width": 1e308},
            "y": {"model": "gaussian", "center": 0, "width": 2},
        },
    }
    spectrum = scp.simulate_peaks(grid, [truth])
    spectrum.data[:, 0] += noise
    peak = {
        "height": {"value": 1, "fixed": True},
        "axes": {
            "x": {
                "model": "lorentzian",
                "center": 1e308,
                "width": {"value": 1e308, "fixed": True},
            },
            "y": {
                "model": "gaussian",
                "center": {"value": 0, "fixed": True},
                "width": {"value": 2, "fixed": True},
            },
        },
    }
    legacy = scp.Optimize(peaks=[peak]).fit(spectrum)
    fit = scp.Optimize(peaks=[peak], shared_parameters={}).fit(spectrum)
    assert fit.result.diagnostics["success"]
    assert_allclose(fit.predict().data, legacy.predict().data, rtol=1e-6, atol=1e-8)
    assert_allclose(
        fit.jacobian, legacy.jacobian, rtol=1e-4, atol=np.nextafter(0.0, 1.0)
    )
    if not noise:
        assert_allclose(fit.jacobian[8, 0] * 1e308, 16 / 289, rtol=1e-6)


def test_general_physical_jacobian_survives_normalized_tail_underflow():
    from spectrochempy.analysis.curvefitting import _optimize_nd

    grid = scp.NDDataset(
        np.zeros((3, 3)),
        coordset=[
            scp.Coord([-1, 0, 1]),
            scp.Coord([-1e170, 0, 1e170], linearize_below=-1),
        ],
    )
    height = 1e300
    tail = 0.25 * (height / 1e170) / 1e170
    grid.data = np.array([0.0625, 1, 0.0625])[:, None] * [tail, height, tail]
    peak = {
        "height": {"value": height, "fixed": True},
        "axes": {
            "x": {
                "model": "lorentzian",
                "center": {"value": 0, "fixed": True},
                "width": {"value": 1, "bounds": (0.5, 2)},
            },
            "y": {
                "model": "gaussian",
                "center": {"value": 0, "fixed": True},
                "width": {"value": 1, "fixed": True},
            },
        },
    }
    result, solution = _optimize_nd.fit_separable_peaks(
        grid, [peak], max_calls=1, shared_parameters={}
    )
    assert_array_equal(solution.jac[[3, 5], 0], 0)
    assert_allclose(
        result["jacobian"][[3, 5], 0],
        -0.5 * (height / 1e170) / 1e170,
        rtol=1e-6,
        atol=0,
    )


def test_callback_result_quantities_are_detached_without_copying_callable_state():
    class Spectral:
        def __deepcopy__(self, memo):
            raise AssertionError("Callback identity must be preserved")

        def __call__(self, x):
            return np.ones_like(x)

    def decay(t, tau):
        return np.exp(-t / tau)

    t = np.array([0.0, 1.0, 2.0])
    grid = scp.NDDataset(
        np.exp(-t[:, None] / 2) * np.ones((1, 3)),
        coordset=[scp.Coord(t, units="s"), scp.Coord([0, 1, 2])],
    )
    spectral = Spectral()
    peak = {
        "height": {"value": 1, "fixed": True},
        "axes": {
            "x": {"model": spectral, "parameters": {}},
            "y": {
                "model": decay,
                "parameters": {"tau": {"value": 2, "unit": "s", "fixed": True}},
            },
        },
    }
    fit = scp.Optimize(peaks=[peak]).fit(grid)
    first, second = fit.result, fit.result
    first.peaks[0]["axes"]["y"]["parameters"]["tau"].ito("ms")
    for result in (second, fit.result):
        tau = result.peaks[0]["axes"]["y"]["parameters"]["tau"]
        assert tau.units == scp.ur.s
        assert tau.magnitude == 2
        assert result.peaks[0]["axes"]["x"]["model"] is spectral
        assert_allclose(scp.simulate_peaks(grid, result.peaks).data, fit.predict().data)
