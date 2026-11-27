"""Private native coordinate wiring with independent physical inference oracles."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis.curvefitting import _optimize_profiles
from spectrochempy.analysis.curvefitting._centered_coordinates import (
    _CenteredParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._optimize_nd import _grid
from spectrochempy.analysis.curvefitting._optimize_profiles import (
    _prepare_profile_layout,
)
from spectrochempy.analysis.curvefitting._optimize_profiles import fit_profiles
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


def flat(axis):
    return np.ones_like(axis)


def fixed(value):
    return {"value": value, "fixed": True}


def kinetic_problem(unit="s", intensity_unit="V", normalization=1):
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
    specs = [
        {
            "height": fixed(h * scp.ur.V),
            "axes": {
                "x": {
                    "model": "exponential_decay",
                    "parameters": {"tau": {"shared": "tau"}},
                },
                "y": {"model": model, "parameters": {"tau": {"shared": "tau"}}},
            },
        }
        for h, model in ((1.3, "exponential_decay"), (0.7, "saturation_recovery"))
    ]
    specs.append(
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
            "unit": unit,
            "bounds": (0.1 * scp.ur.s, 4 * scp.ur.s),
        }
    }
    _, units = _grid(source)
    _, _, _, layout = _prepare_profile_layout(specs, shared, source.units, units)
    time = (1 * scp.ur.s).to(unit).magnitude
    intensity = (1 * scp.ur.V).to(intensity_unit).magnitude
    space = _CenteredParameterCoordinates(
        _ParameterLayout(layout.parameters, (normalization, 0.4 * intensity)),
        (None, None),
        ("log", "lin"),
        (0.5 * time, None),
    )

    def provider(prepared, names):
        tau = prepared[0]["axes"]["x"]["parameters"]["tau"] / 1000
        return np.stack(
            (-slopes(tau) * intensity / time, -np.ones(source.shape)), axis=-1
        )

    return source, specs, shared, space, signal, slopes, provider, time, intensity


@pytest.mark.parametrize("unit", ["s", "ms"])
@pytest.mark.parametrize("intensity_unit", ["V", "mV"])
@pytest.mark.parametrize("normalization", [0.25, 2])
@pytest.mark.parametrize("analytic", [False, True])
def test_log_shared_time_parameter_matches_independent_joint_fit(
    unit, intensity_unit, normalization, analytic
):
    (
        source,
        specs,
        shared,
        space,
        signal,
        slopes,
        provider,
        time,
        intensity,
    ) = kinetic_problem(unit, intensity_unit, normalization)
    before = deepcopy((specs, shared, space))
    original = source.copy()
    fit, solution = fit_profiles(
        source,
        specs,
        400,
        shared,
        parameter_coordinates=space,
        residual_jacobian=provider if analytic else None,
    )
    valid = np.isfinite(source.data) & ~source.mask
    observed = source.data[valid] / intensity
    oracle = least_squares(
        lambda p: observed - signal(*p)[valid],
        [0.6, 0.1],
        bounds=([0.1, -0.5], [4, 1]),
        jac=lambda p: np.column_stack((-slopes(p[0])[valid], -np.ones(valid.sum()))),
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )
    assert solution.success and oracle.success
    conversion = np.array([time, intensity])
    np.testing.assert_allclose(fit["values"] / conversion, oracle.x, rtol=2e-6)
    np.testing.assert_allclose(space.to_physical(solution.x), fit["values"], rtol=2e-14)
    expected_jac = oracle.jac * intensity / conversion
    scale = np.max(np.abs(expected_jac), axis=0)
    np.testing.assert_allclose(
        fit["jacobian"] / scale, expected_jac / scale, rtol=3e-6, atol=3e-8
    )
    covariance = (
        np.linalg.inv(oracle.jac.T @ oracle.jac)
        * (oracle.fun @ oracle.fun)
        / (valid.sum() - 2)
    )
    covariance = conversion[:, None] * covariance * conversion
    np.testing.assert_allclose(fit["covariance"], covariance, rtol=5e-6, atol=0)
    assert abs(covariance[0, 1]) > 0.1 * np.sqrt(np.prod(np.diag(covariance)))
    assert fit["diagnostics"]["covariance_reason"] is None
    np.testing.assert_allclose(
        solution.jac,
        fit["jacobian"]
        * space.derivatives(solution.x)
        / np.max(np.abs(source.data[valid])),
        rtol=3e-6,
        atol=1e-9,
    )
    np.testing.assert_array_equal(fit["fitted"].mask, ~valid)
    np.testing.assert_array_equal(source.data, original.data)
    assert fit["fitted"].units == source.units
    config = fit["config"]["parameter_coordinates"]["shared_parameters['tau']"]
    assert config["initial_value"] == 0.6 * time
    assert config["log_reference"] == 0.5 * time
    assert config["normalization_scale"] == normalization
    assert before == (specs, shared, space)
    assert fit["diagnostics"]["jacobian_coordinates"] == "physical"
    legacy = scp.Optimize(peaks=specs, shared_parameters=shared).fit(source)
    np.testing.assert_allclose(
        legacy.result._parameter_values, fit["values"], rtol=2e-6
    )
    np.testing.assert_allclose(legacy.result.covariance, fit["covariance"], rtol=8e-6)


def center_problem(origin=2**53, shared_center=True, intensity=1, normalization=1):
    rel = np.arange(-6, 8, 2)
    x = scp.Coord(origin + rel, units="Hz", dtype=np.int64, rounding=False)
    y = scp.Coord(
        (np.longdouble(origin) + rel) * np.longdouble(0.001),
        units="kHz",
        dtype=np.longdouble,
        rounding=False,
    )
    position = {"value": origin, "bounds": (origin - 2, origin + 2)}
    specs = [
        {
            "height": {
                "value": 1.5 * intensity,
                "bounds": (0.5 * intensity, 4 * intensity),
            },
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {"shared": "position"} if shared_center else position,
                    "width": {"value": 3.6, "bounds": (1, 7)},
                },
                "y": {
                    "model": "gaussian",
                    "center": {"shared": "position"}
                    if shared_center
                    else fixed(np.longdouble(origin) * np.longdouble(0.001)),
                    "width": fixed(0.004),
                },
            },
        }
    ]
    shared = {"position": {**position, "unit": "Hz"}} if shared_center else {}
    grids, units = _grid(scp.NDDataset(np.zeros((len(rel), len(rel))), coordset=[y, x]))
    prepared, _, _, layout = _prepare_profile_layout(
        specs, shared, scp.ur.V if intensity == 1 else scp.ur.mV, units
    )
    yorigin = prepared[0]["axes"]["y"]["parameters"]["center"]
    tx = np.asarray(x.data.astype(np.longdouble) - origin, dtype=float)[None, :]
    ty = np.asarray((y.data - yorigin) * 1000, dtype=float)[:, None]
    a = 4 * np.log(2)

    def model(h, c, w):
        return h * np.exp(
            -a * ((tx - c) ** 2 / w**2 + (ty - (c if shared_center else 0)) ** 2 / 16)
        )

    source = scp.NDDataset(
        model(2 * intensity, 0.33, 3.2) + 0.0001 * intensity * np.cos(tx + ty),
        coordset=[y, x],
        units="V" if intensity == 1 else "mV",
    )
    source[1:2, 1:3] = scp.MASKED
    source.data[0, -1] = np.nan
    space = _CenteredParameterCoordinates(
        _ParameterLayout(layout.parameters, (intensity, normalization, 0.5)),
        (None, origin - 2, None),
        ("lin", "lin", "log"),
        (None, None, 2),
    )
    return source, specs, shared, space, model


@pytest.mark.parametrize("origin", [0, 2**53])
@pytest.mark.parametrize("shared_center", [False, True])
@pytest.mark.parametrize("intensity", [1, 1000])
@pytest.mark.parametrize("normalization", [0.5, 2])
def test_free_centers_use_stable_offsets_and_each_represented_binding(
    origin, shared_center, intensity, normalization
):
    source, specs, shared, space, model = center_problem(
        origin, shared_center, intensity, normalization
    )
    saved = deepcopy((specs, shared, space))
    original = source.copy()
    fit, solution = fit_profiles(
        source, specs, 500, shared, parameter_coordinates=space
    )
    assert solution.success
    valid = np.isfinite(source.data) & ~source.mask
    h, center, w = fit["values"]
    axes = fit["peaks"][0]["axes"]
    cx = axes["x"]["parameters"]["center"]
    cy = axes["y"]["parameters"]["center"]
    dx = np.asarray(source.x.data.astype(np.longdouble) - cx, dtype=float)[None, :]
    dy = np.asarray((source.y.data.astype(np.longdouble) - cy) * 1000, dtype=float)[
        :, None
    ]
    a = 4 * np.log(2)
    g = np.exp(-a * (dx**2 / float(w) ** 2 + dy**2 / 16))
    expected = np.stack(
        (
            -g,
            -2
            * a
            * float(h)
            * g
            * (dx / float(w) ** 2 + (dy / 16 if shared_center else 0)),
            -2 * a * float(h) * g * dx**2 / float(w) ** 3,
        ),
        axis=-1,
    )[valid]
    np.testing.assert_allclose(
        fit["fitted"].data[valid], float(h) * g[valid], rtol=8e-14
    )
    columns = np.max(np.abs(expected), axis=0)
    np.testing.assert_allclose(
        fit["jacobian"] / columns, expected / columns, rtol=5e-6, atol=2e-8
    )
    rss = np.sum((source.data - float(h) * g)[valid] ** 2)
    covariance = np.linalg.inv(expected.T @ expected) * rss / (valid.sum() - 3)
    np.testing.assert_allclose(fit["covariance"], covariance, rtol=8e-6, atol=0)
    np.testing.assert_allclose(
        space.to_physical(solution.x), fit["values"], rtol=0, atol=0
    )
    assert fit["diagnostics"]["covariance_reason"] is None
    assert space.origins[1] == origin - 2
    assert (
        fit["config"]["parameter_coordinates"][space.layout.free_names[1]][
            "initial_value"
        ]
        == origin
    )
    history = fit["fitted"].history_entries[-1]["parameters"]["parameter_coordinates"][
        space.layout.free_names[1]
    ]
    assert history["initial_value"] == origin
    assert history["origin"] == str(space.origins[1])
    np.testing.assert_array_equal(source.x.data, original.x.data)
    np.testing.assert_array_equal(source.y.data, original.y.data)
    np.testing.assert_array_equal(fit["fitted"].x.data, source.x.data)
    assert saved == (specs, shared, space)
    if not origin:
        oracle = least_squares(
            lambda p: (source.data - model(*p))[valid],
            [1.5 * intensity, 0, 3.6],
            bounds=([0.5 * intensity, -2, 1], [4 * intensity, 2, 7]),
            ftol=1e-12,
            gtol=1e-12,
            xtol=1e-12,
        )
        np.testing.assert_allclose(fit["values"], oracle.x, rtol=3e-6, atol=1e-8)


@pytest.mark.parametrize(
    "field", ["value", "bounds", "unit", "name", "identifier", "fixed"]
)
def test_complete_context_mismatches_are_rejected(field):
    source, specs, shared, space, *_ = kinetic_problem()
    records = list(space.layout.parameters)
    old = records[0]
    value = {
        "value": old.value + 1,
        "bounds": (-2, 3),
        "unit": scp.ur.Hz,
        "name": "other",
        "identifier": ("other",),
        "fixed": False,
    }[field]
    records[0] = replace(old, **{field: value})
    n = sum(not p.fixed for p in records)
    bad = _CenteredParameterCoordinates(_ParameterLayout(tuple(records)), (None,) * n)
    with pytest.raises(ValueError, match="match the complete"):
        fit_profiles(source, specs, 100, shared, parameter_coordinates=bad)


def test_free_centers_require_origins_and_reject_analytic_provider():
    source, specs, shared, space, *_ = center_problem()
    uncentered = _CenteredParameterCoordinates(
        space.layout, (None, None, None), ("lin", "lin", "log"), (None, None, 2)
    )
    with pytest.raises(ValueError, match="explicit linear origins"):
        fit_profiles(source, specs, 100, shared, parameter_coordinates=uncentered)
    with pytest.raises(ValueError, match="Analytic Jacobians with free native centers"):
        fit_profiles(
            source,
            specs,
            100,
            shared,
            parameter_coordinates=space,
            residual_jacobian=lambda *_: None,
        )
    legacy = _ParameterCoordinates(space.layout)
    with pytest.raises(TypeError, match="centered coordinate space"):
        fit_profiles(source, specs, 100, shared, parameter_coordinates=legacy)


def test_all_fixed_context_and_budget_exclusion():
    source = scp.NDDataset(
        np.ones((3, 4)), coordset=[scp.Coord([0, 1, 2]), scp.Coord([0, 1, 2, 3])]
    )
    specs = [
        {
            "height": fixed(1),
            "axes": {d: {"model": flat, "parameters": {}} for d in source.dims},
        }
    ]
    _, units = _grid(source)
    _, _, _, layout = _prepare_profile_layout(specs, {}, source.units, units)
    fit, solution = fit_profiles(
        source,
        specs,
        20,
        {},
        parameter_coordinates=_CenteredParameterCoordinates(layout, ()),
    )
    assert (
        solution is None
        and fit["diagnostics"]["covariance_reason"] == "no_free_parameters"
    )
    source, specs, shared, space, *_ = kinetic_problem()
    fit, solution = fit_profiles(source, specs, 1, shared, parameter_coordinates=space)
    assert (
        not solution.success
        and fit["diagnostics"]["covariance_reason"] == "not_converged"
    )


def test_optimize_uses_existing_result_config_and_physical_outputs(monkeypatch):
    source, specs, shared, space, *_ = kinetic_problem()
    original = _optimize_profiles.fit_profiles

    def private_fit(*args, **kwargs):
        return original(*args, parameter_coordinates=space, **kwargs)

    monkeypatch.setattr(_optimize_profiles, "fit_profiles", private_fit)
    opt = scp.Optimize(peaks=specs, shared_parameters=shared).fit(source)
    result = opt.result
    name = space.layout.free_names[0]
    assert result.parameters["parameter_coordinates"][name]["initial_value"] == 0.6
    assert result.parameter_names == space.layout.free_names
    assert result.covariance is not None and np.isfinite(result.stderr).all()
    assert opt.jacobian.shape[1] == 2
    result.parameters["parameter_coordinates"][name]["initial_value"] = 999
    assert opt.result.parameters["parameter_coordinates"][name]["initial_value"] == 0.6


@pytest.mark.parametrize("small", [False, True])
def test_finite_inference_bound_lost_to_normalization_excludes_covariance(small):
    magnitude = 1e-100 if small else 1e100
    gain = 1e100 if small else 1e-100
    x = scp.Coord.linspace(-1, 1, 21, rounding=False)
    y = scp.Coord([0, 1, 2])

    def linear(axis, k):
        return axis * (k * gain)

    data = np.broadcast_to(2 * x.data, (3, 21)).copy() + 0.001 * np.cos(
        x.data[None, :] + np.arange(3)[:, None]
    )
    source = scp.NDDataset(data, coordset=[y, x])
    specs = [
        {
            "height": fixed(1),
            "axes": {
                "x": {
                    "model": linear,
                    "parameters": {
                        "k": {"value": magnitude, "bounds": (1e-310, 4e300)}
                    },
                },
                "y": {"model": flat, "parameters": {}},
            },
        }
    ]
    _, units = _grid(source)
    _, _, _, layout = _prepare_profile_layout(specs, {}, source.units, units)
    space = _CenteredParameterCoordinates(layout, (None,), ("log",), (magnitude,))
    fit, solution = fit_profiles(source, specs, 300, {}, parameter_coordinates=space)
    assert solution.success
    design = np.broadcast_to(x.data, source.shape)
    expected = np.sum(design * source.data) / np.sum(design**2)
    np.testing.assert_allclose(fit["values"][0] / magnitude, expected, rtol=1e-7)
    assert np.isfinite(fit["jacobian"]).all()
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "numeric_range"


def test_active_bound_and_rank_deficiency_keep_physical_inference_reasons():
    source = scp.NDDataset(
        np.full((3, 4), 1.5), coordset=[scp.Coord([0, 1, 2]), scp.Coord([0, 1, 2, 3])]
    )
    axes = {d: {"model": flat, "parameters": {}} for d in source.dims}
    specs = [{"height": {"value": 0.5, "bounds": (0.1, 0.8)}, "axes": axes}]
    _, units = _grid(source)
    _, _, _, layout = _prepare_profile_layout(specs, {}, source.units, units)
    space = _CenteredParameterCoordinates(layout, (None,), ("log",), (0.2,))
    fit, solution = fit_profiles(source, specs, 200, {}, parameter_coordinates=space)
    assert (
        solution.success and fit["diagnostics"]["covariance_reason"] == "active_bounds"
    )
    source.data += 0.001 * np.arange(12).reshape(3, 4)
    specs = [
        {"height": {"value": v, "bounds": (0.1, 3)}, "axes": deepcopy(axes)}
        for v in (0.5, 0.6)
    ]
    _, _, _, layout = _prepare_profile_layout(specs, {}, source.units, units)
    space = _CenteredParameterCoordinates(layout, (None, None), ("log", "log"), (1, 1))
    fit, solution = fit_profiles(source, specs, 200, {}, parameter_coordinates=space)
    assert (
        solution.success and fit["diagnostics"]["covariance_reason"] == "rank_deficient"
    )


def test_noncenter_origins_and_affine_log_units_are_rejected_before_evaluation():
    source, specs, shared, space, *_ = kinetic_problem()
    wrong = _CenteredParameterCoordinates(
        space.layout, (None, 0), ("log", "lin"), (0.5, None)
    )
    with pytest.raises(ValueError, match="Only free native centers"):
        fit_profiles(source, specs, 200, shared, parameter_coordinates=wrong)
    calls = []

    def linear(axis, k):
        calls.append(k)
        return k * axis

    specs = [
        {
            "height": fixed(1),
            "axes": {
                "x": {
                    "model": linear,
                    "parameters": {"k": {"value": 2, "unit": "degC", "bounds": (1, 5)}},
                },
                "y": {"model": flat, "parameters": {}},
            },
        }
    ]
    _, units = _grid(source)
    _, _, _, layout = _prepare_profile_layout(specs, {}, source.units, units)
    wrong = _CenteredParameterCoordinates(layout, (None,), ("log",), (1,))
    with pytest.raises(ValueError, match="multiplicative"):
        fit_profiles(source, specs, 200, {}, parameter_coordinates=wrong)
    assert calls == []


def test_context_comparison_does_not_round_distinct_integer_declarations():
    source, specs, shared, space, *_ = center_problem()
    records = list(space.layout.parameters)
    records[1] = replace(records[1], value=np.float64(2**53))
    supplied = _CenteredParameterCoordinates(
        _ParameterLayout(tuple(records), space.layout.free_scales),
        space.origins,
        space.transforms,
        space.references,
    )
    shared["position"]["value"] = 2**53 + 1
    with pytest.raises(ValueError, match="match the complete|precision"):
        fit_profiles(source, specs, 100, shared, parameter_coordinates=supplied)


def test_repeated_fits_keep_independent_outputs_and_initial_configuration():
    source, specs, shared, space, *_ = kinetic_problem()
    first, _ = fit_profiles(source, specs, 300, shared, parameter_coordinates=space)
    second, _ = fit_profiles(source, specs, 300, shared, parameter_coordinates=space)
    first["peaks"][0]["height"] = -999
    name = space.layout.free_names[0]
    first["config"]["parameter_coordinates"][name]["initial_value"] = -999
    assert second["peaks"][0]["height"] == 1.3
    assert second["config"]["parameter_coordinates"][name]["initial_value"] == 0.6
    assert space.layout.free_values[0] == 0.6


def test_explicit_linear_coordinates_match_legacy_profile_fit():
    source, specs, shared, space, *_ = kinetic_problem()
    linear = _CenteredParameterCoordinates(space.layout, (None, None))
    first, _ = fit_profiles(source, specs, 300, shared)
    second, _ = fit_profiles(source, specs, 300, shared, parameter_coordinates=linear)
    np.testing.assert_allclose(second["values"], first["values"], rtol=2e-6)
    np.testing.assert_allclose(
        second["jacobian"], first["jacobian"], rtol=3e-6, atol=1e-8
    )
    np.testing.assert_allclose(second["covariance"], first["covariance"], rtol=5e-6)


@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt", "pseudovoigt"])
def test_native_line_shape_conventions_match_the_existing_fitter(model):
    x = scp.Coord.linspace(-4, 4, 81, units="Hz", rounding=False)
    template = scp.NDDataset(np.zeros(81), coordset=[x], units="V")
    truth = {"model": model, "center": 0.2, "width": 1.5}
    if model in ("voigt", "pseudovoigt"):
        truth["ratio"] = 0.4
    source = scp.simulate_peaks(template, [{"height": 2, "axes": {"x": truth}}])
    source.data += 0.0001 * np.cos(x.data)
    axis = {
        "model": model,
        "center": {"value": 0, "bounds": (-1, 1)},
        "width": {"value": 2, "bounds": (0.5, 4)},
    }
    if "ratio" in truth:
        axis["ratio"] = fixed(0.4)
    specs = [{"height": {"value": 1.5, "bounds": (0.5, 3)}, "axes": {"x": axis}}]
    _, units = _grid(source, min_ndim=1)
    _, _, _, layout = _prepare_profile_layout(specs, {}, source.units, units)
    space = _CenteredParameterCoordinates(
        layout, (None, 0, None), ("lin", "lin", "log"), (None, None, 1)
    )
    fit, solution = fit_profiles(
        source, specs, 300, {}, min_ndim=1, parameter_coordinates=space
    )
    legacy, _ = fit_profiles(source, specs, 300, {}, min_ndim=1)
    assert solution.success
    np.testing.assert_allclose(fit["values"], legacy["values"], rtol=3e-6, atol=1e-8)
    np.testing.assert_allclose(
        fit["jacobian"], legacy["jacobian"], rtol=1e-5, atol=1e-7
    )
    deviation = np.sqrt(np.diag(legacy["covariance"]))
    np.testing.assert_allclose(
        fit["covariance"] / deviation[:, None] / deviation,
        legacy["covariance"] / deviation[:, None] / deviation,
        rtol=1e-4,
        atol=1e-7,
    )
