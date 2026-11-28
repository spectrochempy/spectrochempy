"""Exercise the real private historical backend against the M2j references."""

import importlib
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting import _optimize_nd as backend
from spectrochempy.analysis.curvefitting._centered_coordinates import (
    _CenteredParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from tests.test_analysis.test_curvefitting.test_historical_peak_coordinates import (
    assert_covariance,
)
from tests.test_analysis.test_curvefitting.test_historical_peak_coordinates import fixed
from tests.test_analysis.test_curvefitting.test_historical_peak_coordinates import (
    height_problem,
)
from tests.test_analysis.test_curvefitting.test_historical_peak_coordinates import (
    problem,
)
from tests.test_analysis.test_curvefitting.test_historical_peak_coordinates import (
    qualify,
)


def private_fit(source, specs, space, max_calls=300, **kwargs):
    return backend.fit_separable_peaks(
        source, specs, max_calls, parameter_coordinates=space, **kwargs
    )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("model", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("origin", [0, 2**53])
@pytest.mark.parametrize("unit", ["Hz", "kHz"])
@pytest.mark.parametrize("intensity_unit", ["V", "mV"])
@pytest.mark.parametrize("normalization", [0.25, 2])
def test_private_fit_preserves_physical_inference_and_initial_configuration(
    ndim, model, origin, unit, intensity_unit, normalization
):
    source, specs, reference, center0, hz, intensity = problem(
        ndim, model, origin, unit, intensity_unit
    )
    qualification = qualify(source, specs, normalization)
    space = qualification["space"]
    saved = deepcopy((specs, space))
    original = source.copy()
    fit, solution = private_fit(source, specs, space)
    assert solution.success and fit["diagnostics"]["covariance_reason"] is None
    assert fit["names"] == space.layout.free_names
    assert fit["units"] == space.layout.free_units
    h, c, w = fit["values"]
    fitted, expected = reference(float(h), float((c - center0) / hz), float(w / hz))
    valid = np.isfinite(source.data) & ~source.mask
    expected = expected[valid]
    np.testing.assert_allclose(
        fit["fitted"].data[valid], fitted[valid], rtol=2e-13, atol=1e-14 * intensity
    )
    np.testing.assert_allclose(fit["jacobian"], expected, rtol=3e-12, atol=1e-13)
    residual = (source.data - fitted)[valid]
    covariance = (
        np.linalg.inv(expected.T @ expected) * (residual @ residual) / (valid.sum() - 3)
    )
    assert_covariance(fit["covariance"], covariance)
    np.testing.assert_allclose(
        solution.jac,
        expected * space.derivatives(solution.x) / np.max(np.abs(source.data[valid])),
        rtol=3e-12,
        atol=1e-13,
    )
    np.testing.assert_array_equal(space.to_physical(solution.x), fit["values"])
    np.testing.assert_array_equal(
        solution.fun, fit["residuals"].data[valid] / np.max(np.abs(source.data[valid]))
    )
    config = fit["config"]["parameter_coordinates"]
    for record, norm, origin_value, transform, ref in zip(
        space.layout.free_parameters,
        space.layout.free_scales,
        space.origins,
        space.transforms,
        space.references,
        strict=True,
    ):
        entry = config[record.name]
        assert entry["initial_value"] == record.value
        assert entry["bounds"] == list(record.bounds)
        assert entry["normalization_scale"] == norm
        assert entry["origin"] == origin_value
        assert entry["coordinate_transform"] == transform
        assert entry["log_reference"] == ref
    assert saved == (specs, space)
    np.testing.assert_array_equal(source.data, original.data)
    np.testing.assert_array_equal(source.mask, original.mask)
    np.testing.assert_array_equal(fit["fitted"].mask, ~valid)
    assert fit["fitted"].units == source.units
    for dim in source.dims:
        np.testing.assert_array_equal(
            getattr(fit["fitted"], dim).data, getattr(source, dim).data
        )
    if not origin:
        np.testing.assert_allclose(
            fit["values"], qualification["values"], rtol=3e-6, atol=1e-8
        )
    history = fit["fitted"].history_entries[-1]["parameters"]["parameter_coordinates"]
    assert history[space.layout.free_names[1]]["origin"] == str(space.origins[1])
    config[space.layout.free_names[0]]["initial_value"] = 999
    assert space.layout.free_values[0] != 999


@pytest.mark.parametrize("normalization", [1e-5, 1, 1e5])
@pytest.mark.parametrize("intensity", [1, 1000])
def test_log_heights_match_correlated_ols_and_fits_are_independent(
    normalization, intensity
):
    source, specs, columns = height_problem(intensity)
    space = qualify(source, specs, normalization, height_log=True)["space"]
    fit, solution = private_fit(source, specs, space)
    assert solution.success and fit["diagnostics"]["covariance_reason"] is None
    valid = ~source.mask & np.isfinite(source.data)
    design = columns[valid]
    target = source.data[valid]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ expected
    covariance = (
        np.linalg.inv(design.T @ design) * (residual @ residual) / (target.size - 2)
    )
    np.testing.assert_allclose(fit["values"], expected, rtol=1e-7)
    np.testing.assert_allclose(fit["jacobian"], -design, rtol=2e-13, atol=1e-14)
    assert_covariance(fit["covariance"], covariance, rtol=2e-7)
    first = fit["fitted"].copy()
    second, _ = private_fit(source * 0.8, specs, space)
    np.testing.assert_allclose(second["values"], fit["values"] * 0.8, rtol=2e-7)
    np.testing.assert_array_equal(fit["fitted"].data, first.data)
    assert specs[0]["height"] is specs[1]["height"]


@pytest.mark.parametrize(
    "field", ["value", "bounds", "fixed", "unit", "name", "identifier"]
)
def test_complete_fixed_context_must_match_before_evaluation(field, monkeypatch):
    source, specs, *_ = problem()
    space = qualify(source, specs)["space"]
    records = list(space.layout.parameters)
    index = next(i for i, p in enumerate(records) if p.fixed)
    old = records[index]
    change = {
        "value": old.value + 1,
        "bounds": (-3, 3),
        "fixed": False,
        "unit": scp.ur.Hz,
        "name": "different",
        "identifier": ("different",),
    }[field]
    records[index] = replace(old, **{field: change})
    if field == "fixed":
        layout = replace(
            space.layout, parameters=tuple(records), normalization_scales=None
        )
        wrong = _CenteredParameterCoordinates(
            layout, (None,) * len(layout.free_parameters)
        )
    else:
        wrong = _CenteredParameterCoordinates(
            replace(space.layout, parameters=tuple(records)),
            space.origins,
            space.transforms,
            space.references,
        )

    def forbidden(*args, **kwargs):
        pytest.fail("Context validation must precede solver/model evaluation")

    monkeypatch.setattr(backend, "_separable_jacobian", forbidden)
    monkeypatch.setattr(backend, "least_squares", forbidden)
    with pytest.raises(ValueError, match="complete historical peak layout"):
        private_fit(source, specs, wrong)


def test_private_route_rejects_unqualified_combinations(monkeypatch):
    source, specs, *_ = problem()
    space = qualify(source, specs)["space"]
    uncentered = _CenteredParameterCoordinates(space.layout, (None, None, None))
    with pytest.raises(ValueError, match="explicit linear origins"):
        private_fit(source, specs, uncentered)
    wrong_origin = _CenteredParameterCoordinates(
        space.layout, (0, space.origins[1], None), space.transforms, space.references
    )
    with pytest.raises(ValueError, match="Only free historical centers"):
        private_fit(source, specs, wrong_origin)
    with pytest.raises(TypeError, match="prepared centered coordinate"):
        private_fit(source, specs, _ParameterCoordinates(space.layout))
    with pytest.raises(ValueError, match="shared or extended"):
        private_fit(source, specs, space, shared_parameters={})
    extended = deepcopy(specs)
    extended[0]["axes"]["x"]["model"] = "voigt"
    with pytest.raises(ValueError, match="shared or extended"):
        private_fit(source, extended, space)
    x = scp.Coord([-1, 0, 1])
    four = scp.NDDataset(np.ones((3, 3, 3, 3)), coordset=[x, x, x, x])
    with pytest.raises(ValueError, match="analytic 2D/3D"):
        private_fit(four, specs, space)

    def nonfinite(*args, **kwargs):
        return np.full((source.size - np.count_nonzero(source.mask) - 1, 3), np.nan)

    monkeypatch.setattr(backend, "_separable_jacobian", nonfinite)
    with pytest.raises(ValueError, match="finite analytic Jacobian.*fallback"):
        private_fit(source, specs, space)


def test_existing_optimize_result_contract_receives_private_config(monkeypatch):
    source, specs, *_ = problem()
    space = qualify(source, specs)["space"]
    module = importlib.import_module("spectrochempy.analysis.curvefitting.optimize")
    original = module.fit_separable_peaks

    def explicit(*args, **kwargs):
        return original(*args, **kwargs, parameter_coordinates=space)

    monkeypatch.setattr(module, "fit_separable_peaks", explicit)
    opt = scp.Optimize(peaks=specs).fit(source)
    result = opt.result
    name = space.layout.free_names[0]
    assert result.parameter_names == space.layout.free_names
    assert result.parameters["parameter_coordinates"][name]["initial_value"] == 1.3
    assert result.covariance is not None
    np.testing.assert_allclose(result.stderr, np.sqrt(np.diag(result.covariance)))
    result.parameters["parameter_coordinates"][name]["initial_value"] = 999
    assert opt.result.parameters["parameter_coordinates"][name]["initial_value"] == 1.3


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("model", ["gaussian", "lorentzian"])
def test_negative_height_with_log_width_remains_physical(ndim, model):
    source, specs, reference, center0, hz, _ = problem(ndim, model, 2**53)
    source = -source
    specs[0]["height"] = {
        "value": -1.3 * scp.ur.V,
        "bounds": (-3 * scp.ur.V, -0.1 * scp.ur.V),
    }
    space = qualify(source, specs, 2)["space"]
    fit, solution = private_fit(source, specs, space)
    assert solution.success and fit["diagnostics"]["covariance_reason"] is None
    h, c, w = fit["values"]
    assert h < 0 and space.transforms == ("lin", "lin", "log")
    fitted, matrix = reference(float(h), float((c - center0) / hz), float(w / hz))
    valid = ~source.mask & np.isfinite(source.data)
    expected = matrix[valid]
    np.testing.assert_allclose(fit["jacobian"], expected, rtol=3e-12, atol=1e-13)
    residual = (source.data - fitted)[valid]
    covariance = (
        np.linalg.inv(expected.T @ expected)
        * (residual @ residual)
        / (residual.size - 3)
    )
    assert_covariance(fit["covariance"], covariance)


def test_all_fixed_context_and_budget_exhaustion():
    source, specs, *_ = problem()
    space = qualify(source, specs)["space"]
    fit, solution = private_fit(source, specs, space, max_calls=1)
    assert not solution.success
    assert (
        fit["covariance"] is None
        and fit["diagnostics"]["covariance_reason"] == "not_converged"
    )
    for peak in specs:
        peak["height"]["fixed"] = True
        for axis in peak["axes"].values():
            for key in ("center", "width"):
                axis[key]["fixed"] = True
    _, units = backend._grid(source)
    _, _, layout = backend._prepare_peak_layout(specs, source.units, units)
    fixed_space = _CenteredParameterCoordinates(layout, ())
    fit, solution = private_fit(source, specs, fixed_space)
    assert solution is None and fit["jacobian"] is None
    assert fit["names"] == () and fit["values"].size == 0
    assert fit["config"]["parameter_coordinates"] == {}
    assert fit["diagnostics"]["covariance_reason"] == "no_free_parameters"
    legacy, _ = backend.fit_separable_peaks(source, specs, 300)
    np.testing.assert_array_equal(fit["fitted"].data, legacy["fitted"].data)


def test_active_bound_and_rank_deficiency_keep_common_diagnostics():
    source, specs, *_ = problem()
    specs[0]["height"] = {"value": 0.5, "bounds": (0.1, 0.8)}
    for axis in specs[0]["axes"].values():
        for key in ("center", "width"):
            axis[key] = fixed(axis[key]["value"])
    _, units = backend._grid(source)
    _, _, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(
        replace(layout, normalization_scales=(1e5,)), (None,), ("log",), (0.25,)
    )
    fit, solution = private_fit(source, specs, space)
    assert solution.success
    assert (
        fit["covariance"] is None
        and fit["diagnostics"]["covariance_reason"] == "active_bounds"
    )
    source, specs, _ = height_problem()
    specs[1]["axes"] = deepcopy(specs[0]["axes"])
    space = qualify(source, specs, height_log=True)["space"]
    fit, solution = private_fit(source, specs, space)
    assert solution.success
    assert (
        fit["covariance"] is None
        and fit["diagnostics"]["covariance_reason"] == "rank_deficient"
    )


def test_insufficient_degrees_of_freedom_precedes_rank_exclusion():
    x = scp.Coord([-1, 1])
    template = scp.NDDataset(np.zeros((2, 2)), coordset=[x, x])
    plain = {
        "height": 1,
        "axes": {
            d: {"model": "gaussian", "center": 0, "width": 2} for d in template.dims
        },
    }
    source = scp.simulate_peaks(template, [plain])
    source.mask = np.array([[True, False], [False, False]])
    specs = [
        {
            "height": {"value": 1, "bounds": (0.1, 3)},
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {"value": 0, "bounds": (-1, 1)},
                    "width": {"value": 2, "bounds": (1, 4)},
                },
                "y": {"model": "gaussian", "center": fixed(0), "width": fixed(2)},
            },
        }
    ]
    _, units = backend._grid(source)
    _, _, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(layout, (None, -1, None))
    fit, solution = private_fit(source, specs, space)
    assert solution.success and fit["diagnostics"]["degrees_of_freedom"] == 0
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "insufficient_degrees_of_freedom"


@pytest.mark.parametrize("magnitude", [1e-100, 1e100])
def test_lost_finite_inference_bound_excludes_covariance(magnitude):
    x = scp.Coord.linspace(-2, 2, 15, rounding=False)
    y = scp.Coord.linspace(-1, 1, 9, rounding=False)
    template = scp.NDDataset(np.zeros((9, 15)), coordset=[y, x])
    plain = {
        "height": 2 * magnitude,
        "axes": {
            d: {"model": "gaussian", "center": 0, "width": 2} for d in template.dims
        },
    }
    source = scp.simulate_peaks(template, [plain])
    source.data += 0.001 * magnitude * np.cos(x.data[None, :] + y.data[:, None])
    specs = [
        {
            "height": {"value": magnitude, "bounds": (1e-310, 4e300)},
            "axes": {
                d: {"model": "gaussian", "center": fixed(0), "width": fixed(2)}
                for d in source.dims
            },
        }
    ]
    _, units = backend._grid(source)
    _, _, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(layout, (None,), ("log",), (magnitude,))
    fit, solution = private_fit(source, specs, space)
    assert solution.success and np.isfinite(fit["jacobian"]).all()
    assert (
        fit["covariance"] is None
        and fit["diagnostics"]["covariance_reason"] == "numeric_range"
    )


def test_initial_configuration_is_copied_before_solver_runs(monkeypatch):
    source, specs, *_ = problem()
    space = qualify(source, specs)["space"]
    original = backend.least_squares
    origins = space.origins

    def mutate_caller(*args, **kwargs):
        specs[0]["height"]["value"] = 999
        object.__setattr__(space, "origins", (None,) * 3)
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "least_squares", mutate_caller)
    fit, solution = private_fit(source, specs, space)
    assert solution.success and fit["values"][0] < 3
    config = fit["config"]["parameter_coordinates"]
    assert config[space.layout.free_names[0]]["initial_value"] == 1.3
    assert config[space.layout.free_names[1]]["origin"] == origins[1]
    history = fit["fitted"].history_entries[-1]["parameters"]
    assert history["peak_specification"][0]["height"]["value"]["value"] == 1.3


def test_affine_log_units_and_log_center_are_rejected():
    source, specs, _ = height_problem()
    source = scp.NDDataset(
        source.data.copy(),
        coordset=source.coordset.copy(),
        mask=source.mask.copy(),
        units="degC",
    )
    _, units = backend._grid(source)
    _, _, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(layout, (None, None), ("log", "log"), (1, 1))
    with pytest.raises(ValueError, match="multiplicative parameter units"):
        private_fit(source, specs, space)
    source, specs, *_ = problem(origin=10)
    _, units = backend._grid(source)
    _, _, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(
        layout, (None, None, None), ("lin", "log", "log"), (None, 10, 2)
    )
    with pytest.raises(ValueError, match="explicit linear origins"):
        private_fit(source, specs, space)


@pytest.mark.parametrize("initial, low", [("0.1", "-4"), ("0", "-4.1")])
@pytest.mark.parametrize("normalization", [0.25, 1, 8])
def test_far_center_origin_cannot_lose_initial_or_finite_bound_anchors(
    initial, low, normalization, monkeypatch
):
    source, specs, *_ = problem()
    axis = specs[0]["axes"]["x"]
    axis["center"] = {
        "value": np.longdouble(initial),
        "bounds": (np.longdouble(low), 4),
    }
    _, units = backend._grid(source)
    _, free, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(
        replace(layout, normalization_scales=(normalization,) * len(free)),
        tuple(2**53 if key == "center" else None for _, key in free),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Unsuitable origin must be rejected before evaluating or fitting")

    monkeypatch.setattr(backend, "least_squares", forbidden)
    monkeypatch.setattr(backend, "_separable_jacobian", forbidden)
    with pytest.raises(
        ValueError, match="origin loses anchor offsets relative to the axis width"
    ):
        private_fit(source, specs, space)


@pytest.mark.parametrize("normalization", [1, 8])
def test_origin_subtraction_cannot_conceal_lost_open_bound_initial(
    normalization, monkeypatch
):
    source, specs, *_ = problem()
    specs[0]["axes"]["x"]["center"] = {
        "value": np.longdouble("0.1"),
        "bounds": (None, None),
    }
    _, units = backend._grid(source)
    _, free, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(
        replace(layout, normalization_scales=(normalization,) * len(free)),
        tuple(2**80 if key == "center" else None for _, key in free),
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Lost anchor must be rejected even when longdouble subtraction loses it"
        )

    monkeypatch.setattr(backend, "least_squares", forbidden)
    with pytest.raises(
        ValueError, match="origin loses anchor offsets relative to the axis width"
    ):
        private_fit(source, specs, space)
