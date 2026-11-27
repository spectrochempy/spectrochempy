"""Qualify complete native profile context without wiring a log fitting engine."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._optimize_nd import _grid
from spectrochempy.analysis.curvefitting._optimize_profiles import (
    _prepare_profile_layout,
)
from spectrochempy.analysis.curvefitting._optimize_profiles import _prepare_profiles
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting._profile_models import evaluate_profiles


def flat(axis):
    return np.ones_like(axis)


def profile_problem(parameter_unit="s", intensity_unit="V"):
    x = scp.Coord(
        [6000, 4500, 3100, 2000, 1300, 850, 520, 300, 160, 80, 30, 0], units="ms"
    )
    y = scp.Coord([0, 0.13, 0.4, 0.9, 1.8, 3.5], units="s")
    tx, ty = np.meshgrid(x.to("s").data, y.data)

    def signal(tau, background):
        return 0.7 * np.exp(-tx / tau) + 0.6 * np.exp(-(tx + ty) / tau) + background

    def slopes(tau):
        return (
            0.7 * tx * np.exp(-tx / tau) + 0.6 * (tx + ty) * np.exp(-(tx + ty) / tau)
        ) / tau**2

    source = scp.NDDataset(
        signal(1.4, 0.15) + 0.001 * np.cos(2 * tx + 3 * ty), coordset=[y, x], units="V"
    ).to(intensity_unit)
    source[1:3, 2:6] = scp.MASKED
    source.data[0, 0] = np.nan
    peaks = [
        {
            "height": {"value": height * scp.ur.V, "fixed": True},
            "axes": {
                "x": {
                    "model": "exponential_decay",
                    "parameters": {"tau": {"shared": "tau"}},
                },
                "y": {"model": model, "parameters": {"tau": {"shared": "tau"}}},
            },
        }
        for height, model in ((1.3, "exponential_decay"), (0.7, "saturation_recovery"))
    ]
    peaks.append(
        {
            "height": {
                "value": 0.1 * scp.ur.V,
                "bounds": (-0.5 * scp.ur.V, 1 * scp.ur.V),
            },
            "axes": {dim: {"model": flat, "parameters": {}} for dim in ("x", "y")},
        }
    )
    shared = {
        "tau": {
            "value": 0.6 * scp.ur.s,
            "unit": parameter_unit,
            "bounds": (0.1 * scp.ur.s, 4 * scp.ur.s),
        }
    }
    return source, peaks, shared, signal, slopes


@pytest.mark.parametrize("parameter_unit", ["s", "ms"])
@pytest.mark.parametrize("intensity_unit", ["V", "mV"])
@pytest.mark.parametrize("normalization", [0.25, 2])
def test_shared_profile_coordinates_against_independent_fit_jacobian_and_covariance(
    parameter_unit, intensity_unit, normalization
):
    source, specs, shared, signal, slopes = profile_problem(
        parameter_unit, intensity_unit
    )
    original, initial, shared_initial = source.copy(), deepcopy(specs), deepcopy(shared)
    coordinates, units = _grid(source)
    peaks, groups, _, layout = _prepare_profile_layout(
        specs, shared, source.units, units
    )
    time_factor = (1 * scp.ur.s).to(parameter_unit).magnitude
    intensity_factor = (1 * scp.ur.V).to(intensity_unit).magnitude
    assert len(layout.parameters) == 4
    assert tuple(p.fixed for p in layout.parameters) == (True, False, True, False)
    assert layout.free_names == ("shared_parameters['tau']", "peaks[2].height")
    assert layout.parameters[1].identifier == ("shared_parameters", "tau")
    assert len(groups[1]["bindings"]) == 4
    assert all(
        p is g["parameter"] for p, g in zip(layout.parameters, groups, strict=True)
    )
    space = _ParameterCoordinates(
        _ParameterLayout(layout.parameters, (normalization, 0.4 * intensity_factor)),
        ("log", "lin"),
        ((0.5 * scp.ur.s).to(parameter_unit).magnitude, None),
    )
    valid = ~source.mask & np.isfinite(source.data)
    observed = np.asarray(source.data[valid], dtype=float)
    intensity_scale = np.max(np.abs(observed))
    free = [g for g in groups if not g["parameter"].fixed]

    def evaluate(vector):
        for group, value in zip(free, space.to_physical(vector), strict=True):
            for target, key, convert in group["bindings"]:
                target[key] = convert(value)
        return sum(evaluate_profiles(source.shape, coordinates, peak) for peak in peaks)

    limits = np.asarray(space.bounds)
    solution = least_squares(
        lambda z: (observed - evaluate(z)[valid]) / intensity_scale,
        space.initial,
        bounds=(limits[:, 0], limits[:, 1]),
        jac="3-point",
        x_scale=1,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )
    physical = space.to_physical(solution.x)
    observed_v = observed / intensity_factor
    oracle = least_squares(
        lambda p: observed_v - signal(*p)[valid],
        [0.6, 0.1],
        bounds=([0.1, -0.5], [4, 1]),
        jac=lambda p: np.column_stack((-slopes(p[0])[valid], -np.ones(valid.sum()))),
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )
    assert solution.success and oracle.success
    conversion = np.array([time_factor, intensity_factor])
    np.testing.assert_allclose(physical / conversion, oracle.x, rtol=1e-7)
    expected_jac = np.column_stack(
        (
            -slopes(oracle.x[0])[valid] * intensity_factor / time_factor,
            -np.ones(valid.sum()),
        )
    )
    physical_jac = space.physical_jacobian(solution.jac * intensity_scale, solution.x)
    column_scales = np.max(np.abs(expected_jac), axis=0)
    np.testing.assert_allclose(
        physical_jac / column_scales, expected_jac / column_scales, rtol=2e-6, atol=2e-8
    )
    covariance = (
        np.linalg.inv(oracle.jac.T @ oracle.jac)
        * (oracle.fun @ oracle.fun)
        / (valid.sum() - 2)
    )
    covariance = conversion[:, None] * covariance * conversion
    solver_covariance = (
        np.linalg.inv(solution.jac.T @ solution.jac)
        * (solution.fun @ solution.fun)
        / (valid.sum() - 2)
    )
    np.testing.assert_allclose(
        space.physical_covariance(solver_covariance, solution.x), covariance, rtol=3e-6
    )
    assert abs(covariance[0, 1]) > 0.1 * np.sqrt(np.prod(np.diag(covariance)))
    evaluate(solution.x)
    tau_seconds = physical[0] / time_factor
    for peak in peaks[:2]:
        np.testing.assert_allclose(
            peak["axes"]["x"]["parameters"]["tau"], tau_seconds * 1000
        )
        np.testing.assert_allclose(peak["axes"]["y"]["parameters"]["tau"], tau_seconds)
    legacy = scp.Optimize(peaks=specs, shared_parameters=shared).fit(source)
    assert legacy.result.parameter_names == layout.free_names
    assert legacy.result.parameter_units == (scp.ur.Unit(parameter_unit), source.units)
    np.testing.assert_allclose(
        legacy.jacobian / column_scales,
        expected_jac / column_scales,
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(legacy.result.covariance, covariance, rtol=2e-5)
    np.testing.assert_allclose(
        legacy.result.peaks[0]["axes"]["x"]["parameters"]["tau"] / 1000,
        oracle.x[0],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        legacy.result.peaks[2]["height"] / intensity_factor, oracle.x[1], rtol=2e-6
    )
    np.testing.assert_array_equal(legacy.result.fitted.mask, ~valid)
    np.testing.assert_array_equal(source.data, original.data)
    np.testing.assert_array_equal(source.mask, original.mask)
    assert specs == initial and shared == shared_initial


def test_complete_context_keeps_fixed_units_and_legacy_free_order():
    source, specs, shared, *_ = profile_problem("ms", "mV")
    _, units = _grid(source)
    prepared, groups, _, layout = _prepare_profile_layout(
        specs, shared, source.units, units
    )
    legacy_peaks, legacy_free, _ = _prepare_profiles(specs, shared, source.units, units)
    assert tuple(g["parameter"] for g in legacy_free) == layout.free_parameters
    assert tuple(g["parameter"].name for g in groups) == (
        "peaks[0].height",
        "shared_parameters['tau']",
        "peaks[1].height",
        "peaks[2].height",
    )
    assert layout.parameters[0].unit == scp.ur.mV
    assert layout.parameters[0].value == 1300
    assert prepared[1]["height"] == legacy_peaks[1]["height"] == 700
    assert [len(g["bindings"]) for g in groups] == [1, 4, 1, 1]


def test_profile_preparations_and_input_declarations_do_not_share_state():
    source, specs, shared, *_ = profile_problem()
    _, units = _grid(source)
    first, groups, _, layout = _prepare_profile_layout(
        specs, shared, source.units, units
    )
    second, others, _, same = _prepare_profile_layout(
        specs, shared, source.units, units
    )
    assert layout == same
    target, key, convert = groups[1]["bindings"][0]
    target[key] = convert(3)
    assert first[0]["axes"]["x"]["parameters"]["tau"] == 3000
    assert second[0]["axes"]["x"]["parameters"]["tau"] == 600
    assert others[1]["bindings"][0][0] is not target
    assert shared["tau"]["value"] == 0.6 * scp.ur.s


def test_local_definition_object_identity_does_not_create_profile_sharing():
    source, specs, shared, *_ = profile_problem()
    definition = {"value": 1.1, "bounds": (0.1, 4)}
    specs[0]["height"] = definition
    specs[1]["height"] = definition
    _, units = _grid(source)
    _, groups, _, layout = _prepare_profile_layout(specs, shared, source.units, units)
    assert layout.parameters[0].identifier == ("peaks", 0, "height")
    assert layout.parameters[2].identifier == ("peaks", 1, "height")
    assert groups[0]["bindings"][0][0] is not groups[2]["bindings"][0][0]
    assert len(layout.free_parameters) == 4


def test_complete_layout_distinguishes_changes_to_fixed_declarations():
    source, specs, shared, *_ = profile_problem()
    _, units = _grid(source)
    _, _, _, before = _prepare_profile_layout(specs, shared, source.units, units)
    specs[0]["height"]["value"] = 1.4 * scp.ur.V
    _, _, _, after = _prepare_profile_layout(specs, shared, source.units, units)
    assert before.free_parameters == after.free_parameters
    assert before.parameters != after.parameters
    assert before.parameters[0].value == 1.3
    assert after.parameters[0].value == 1.4


@pytest.mark.parametrize("unit", ["s", "ms"])
def test_shared_fixed_parameter_keeps_bindings_but_no_solver_coordinate(unit):
    source, specs, shared, *_ = profile_problem(unit)
    shared["tau"]["fixed"] = True
    _, units = _grid(source)
    _, groups, _, layout = _prepare_profile_layout(specs, shared, source.units, units)
    space = _ParameterCoordinates(layout)
    assert layout.parameters[1].fixed
    assert len(groups[1]["bindings"]) == 4
    assert layout.free_names == ("peaks[2].height",)
    assert len(space.initial) == 1
    assert space.references == (None,)
    _, legacy, _ = _prepare_profiles(specs, shared, source.units, units)
    assert tuple(group["parameter"] for group in legacy) == layout.free_parameters


@pytest.mark.parametrize("case", ["unknown", "unused", "units", "bounds"])
def test_complete_preparation_keeps_legacy_validation(case):
    source, specs, shared, *_ = profile_problem()
    _, units = _grid(source)
    if case == "unknown":
        specs[0]["axes"]["x"]["parameters"]["tau"]["shared"] = "missing"
    elif case == "unused":
        shared["unused"] = {"value": 1}
    elif case == "units":
        shared["tau"]["unit"] = "Hz"
    else:
        shared["tau"]["bounds"] = (4 * scp.ur.s, 0.1 * scp.ur.s)
    for prepare in (_prepare_profiles, _prepare_profile_layout):
        with pytest.raises(ValueError):
            prepare(specs, shared, source.units, units)


def test_integer_center_context_survives_a_float64_coordinate_barrier():
    origin = 2**60
    specs = [
        {
            "height": {"value": 1, "fixed": True},
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {"value": origin + 1, "bounds": (origin - 2, origin + 2)},
                    "width": {"value": 2, "fixed": True},
                },
                "y": {
                    "model": "gaussian",
                    "center": {"value": 0, "fixed": True},
                    "width": {"value": 2, "fixed": True},
                },
            },
        }
    ]
    _, _, _, layout = _prepare_profile_layout(specs, {}, None, {"x": None, "y": None})
    record = layout.free_parameters[0]
    assert isinstance(record.value, int) and record.value == origin + 1
    assert record.bounds == (origin - 2, origin + 2)
    assert layout.free_names == ("peaks[0].axes['x'].parameters['center']",)
    with pytest.raises(ValueError, match="Physical bounds|collapse"):
        _ParameterCoordinates(layout)


def test_fixed_extended_center_is_not_quantized_by_complete_preparation():
    center = np.longdouble("1000000.0000000001")
    specs = [
        {
            "height": {"value": 1, "fixed": True},
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {"value": center, "fixed": True},
                    "width": {"value": 0.002, "bounds": (0.001, 0.01)},
                },
                "y": {
                    "model": "gaussian",
                    "center": {"value": 0, "fixed": True},
                    "width": {"value": 2, "fixed": True},
                },
            },
        }
    ]
    peaks, _, _, layout = _prepare_profile_layout(
        specs, {}, None, {"x": None, "y": None}
    )
    record = next(
        p
        for p in layout.parameters
        if p.identifier[-3:] == ("x", "parameters", "center")
    )
    assert record.fixed and record.value == center
    assert isinstance(record.value, type(center))
    assert peaks[0]["axes"]["x"]["parameters"]["center"] == center
    assert layout.free_names == ("peaks[0].axes['x'].parameters['width']",)
