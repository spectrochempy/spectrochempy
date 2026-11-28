"""
Qualify historical peak coordinates without selecting them in an estimator.

The test-local solve uses the existing centered model and internal analytic
Jacobian in 2D/3D. Virtual solver derivatives and final represented-point
inference are qualified separately; this is not native backend wiring.
"""

from copy import copy
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis.curvefitting import _optimize_nd as backend
from spectrochempy.analysis.curvefitting._centered_coordinates import (
    _CenteredParameterCoordinates,
)
from spectrochempy.analysis.curvefitting.simulation import _evaluate_separable_peak


def fixed(value):
    return {"value": value, "fixed": True}


def assert_covariance(actual, expected, *, rtol=2e-11):
    """Compare in standard-deviation units, including nearly zero cross terms."""
    std = np.sqrt(np.diag(expected))
    assert np.isfinite(std).all() and (std > 0).all()
    scale = std[:, None] * std
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=rtol, atol=2e-13)


def qualify(source, specs, normalization=1, *, height_log=False):
    assert source.ndim in (2, 3)
    coordinates, units = backend._grid(source)
    peaks, free, layout = backend._prepare_peak_layout(specs, source.units, units)
    origins = tuple(
        value if key == "center" else 0
        for (target, key), value in zip(free, layout.free_values, strict=True)
    )
    scales = tuple(
        abs(target[key]) or 1 if key == "height" else target["width"]
        for target, key in free
    )
    centered = tuple(
        value - scale / 2 if key == "center" else None
        for (_, key), value, scale in zip(free, layout.free_values, scales, strict=True)
    )
    transforms = tuple(
        "log" if key == "width" or (key == "height" and height_log) else "lin"
        for _, key in free
    )
    space = _CenteredParameterCoordinates(
        replace(
            layout,
            normalization_scales=tuple(
                normalization if transform == "log" else scale
                for transform, scale in zip(transforms, scales, strict=True)
            ),
        ),
        centered,
        transforms,
        tuple(
            target[key] / 2 if transform == "log" else None
            for (target, key), transform in zip(free, transforms, strict=True)
        ),
    )
    initial_offsets = space.to_offsets(space.initial)
    center_origins = {
        id(target): target[key] for target, key in free if key == "center"
    }
    solver_peaks, grids, targets = backend._solver_model(
        peaks, coordinates, center_origins
    )
    bindings = [(targets[id(target)], key) for target, key in free]
    columns = {
        (id(target), key): (i, np.log(factor))
        for i, ((target, factor), key) in enumerate(bindings)
    }
    valid = ~source.mask & np.isfinite(source.data)
    observed = source.data[valid]
    intensity = np.max(np.abs(observed)) or 1

    def evaluate(z):
        for i, (((target, factor), key), value) in enumerate(
            zip(bindings, space.to_offsets(z), strict=True)
        ):
            if key == "center":
                value -= initial_offsets[i]
            target[key] = float(value) if key == "height" else value / factor
        return sum(
            _evaluate_separable_peak(source.shape, grid, peak)
            for grid, peak in zip(grids, solver_peaks, strict=True)
        )

    def jacobian(z):
        evaluate(z)
        return backend._separable_jacobian(
            list(zip(solver_peaks, grids, strict=True)),
            columns,
            np.nonzero(valid),
            np.log(space.derivatives(z)) - np.log(intensity),
        )

    bounds = np.asarray(space.bounds).T
    solution = least_squares(
        lambda z: (observed - evaluate(z)[valid]) / intensity,
        space.initial,
        bounds=bounds,
        jac=jacobian,
        method="trf",
        x_scale=1,
        gtol=None,
        max_nfev=300,
    )
    virtual = space.to_offsets(solution.x).copy()
    for i, (_, key) in enumerate(free):
        if key == "center":
            virtual[i] -= initial_offsets[i]
    virtual_jacobian = space.physical_jacobian(
        jacobian(solution.x) * intensity, solution.x
    )
    values = space.to_physical(solution.x)
    for (target, key), value in zip(free, values, strict=True):
        target[key] = float(value) if key == "height" else value
    physical_jacobian = backend._separable_jacobian(
        [(peak, coordinates) for peak in peaks],
        {(id(target), key): (i, 0) for i, (target, key) in enumerate(free)},
        np.nonzero(valid),
        np.zeros(len(free)),
    )
    fitted = sum(
        _evaluate_separable_peak(source.shape, coordinates, peak) for peak in peaks
    )
    residual = observed - fitted[valid]
    inference = copy(solution)
    scales = np.asarray(scales, dtype=np.longdouble)
    origins = np.asarray(origins, dtype=np.longdouble)
    inference.x = np.asarray((values - origins) / scales, dtype=float)
    inference.fun = residual / intensity
    inference.jac = np.asarray(physical_jacobian * scales / intensity, dtype=float)
    physical_bounds = np.asarray(
        (np.asarray(layout.free_bounds, dtype=np.longdouble).T - origins) / scales,
        dtype=float,
    )
    diagnostics = backend._fit_diagnostics(observed, residual, len(free))
    covariance, reason = backend._physical_covariance(
        inference, physical_jacobian, scales, physical_bounds, diagnostics
    )
    return {
        "solution": solution,
        "space": space,
        "peaks": peaks,
        "layout": layout,
        "values": values,
        "virtual": virtual,
        "virtual_jacobian": virtual_jacobian,
        "jacobian": physical_jacobian,
        "fitted": fitted,
        "covariance": covariance,
        "reason": reason,
        "valid": valid,
        "residual": residual,
    }


def problem(ndim=2, model="gaussian", origin=0, unit="Hz", intensity_unit="V"):
    hz = (1 * scp.ur.Hz).to(unit).magnitude
    signal_unit = (1 * scp.ur.V).to(intensity_unit).magnitude
    rel = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])[::-1]
    x = scp.Coord(
        (np.longdouble(origin) + rel) * np.longdouble(hz),
        units=unit,
        dtype=np.longdouble,
        rounding=False,
    )
    y = scp.Coord([-2.1, -1.3, -0.6, -0.1, 0.4, 1.2, 2.2], units="s", rounding=False)
    coords = [y, x] if ndim == 2 else [scp.Coord([-1, -0.3, 0.2, 0.8, 1.7]), y, x]
    shape = tuple(len(c) for c in coords)
    specs = [
        {
            "height": {
                "value": 1.3 * scp.ur.V,
                "bounds": (0.1 * scp.ur.V, 3 * scp.ur.V),
            },
            "axes": {
                "x": {
                    "model": model,
                    "center": {
                        "value": origin * scp.ur.Hz,
                        "bounds": ((origin - 2) * scp.ur.Hz, (origin + 2) * scp.ur.Hz),
                    },
                    "width": {
                        "value": 3.6 * scp.ur.Hz,
                        "bounds": (1 * scp.ur.Hz, 7 * scp.ur.Hz),
                    },
                },
                "y": {"model": "lorentzian", "center": fixed(0.2), "width": fixed(1.8)},
            },
        }
    ]
    if ndim == 3:
        specs[0]["axes"]["z"] = {
            "model": "gaussian",
            "center": fixed(0.1),
            "width": fixed(1.2),
        }
    template = scp.NDDataset(np.zeros(shape), coordset=coords, units=intensity_unit)
    grid, units = backend._grid(template)
    prepared, _, _ = backend._prepare_peak_layout(specs, template.units, units)
    initial_center = prepared[0]["axes"]["x"]["center"]
    dx = np.asarray((grid["x"] - initial_center) / hz, dtype=float)
    base = 1 / (1 + 4 * ((grid["y"] - 0.2) / 1.8) ** 2)
    base = base[:, None]
    if ndim == 3:
        base = (
            np.exp(-4 * np.log(2) * ((grid["z"] - 0.1) / 1.2) ** 2)[:, None, None]
            * base
        )

    def reference(h, c, w):
        delta = dx - c
        if model == "gaussian":
            g = np.exp(-4 * np.log(2) * (delta / w) ** 2)
            dc = g * 8 * np.log(2) * delta / w**2
        else:
            g = 1 / (1 + 4 * (delta / w) ** 2)
            dc = g**2 * 8 * delta / w**2
        fitted = h * base * g
        jac = -np.stack(
            np.broadcast_arrays(
                base * g, h * base * dc / hz, h * base * dc * delta / w / hz
            ),
            axis=-1,
        )
        return fitted, jac

    template.data = reference(1.7 * signal_unit, 0.33, 3.2)[
        0
    ] + 0.0002 * signal_unit * np.cos(dx)
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[4:7] = True
    template.mask = mask
    template.data.reshape(-1)[0] = np.nan
    return template, specs, reference, initial_center, hz, signal_unit


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("model", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("origin", [0, 2**53])
@pytest.mark.parametrize("unit", ["Hz", "kHz"])
@pytest.mark.parametrize("intensity_unit", ["V", "mV"])
@pytest.mark.parametrize("normalization", [0.25, 2])
def test_centered_analytic_chain_and_represented_physical_covariance(
    ndim, model, origin, unit, intensity_unit, normalization
):
    source, specs, reference, center0, hz, intensity = problem(
        ndim, model, origin, unit, intensity_unit
    )
    saved, original = deepcopy(specs), source.copy()
    fit = qualify(source, specs, normalization)
    assert fit["solution"].success and fit["reason"] is None
    assert len(fit["layout"].parameters) == 1 + 2 * ndim
    assert fit["layout"].free_names == (
        "peaks[0].height",
        "peaks[0].axes['x'].center",
        "peaks[0].axes['x'].width",
    )
    h, c, w = fit["values"]
    physical_center = (c - center0) / hz
    fitted, expected = reference(float(h), float(physical_center), float(w / hz))
    valid = fit["valid"]
    expected = expected[valid]
    np.testing.assert_allclose(
        fit["fitted"][valid], fitted[valid], rtol=2e-13, atol=1e-14 * intensity
    )
    np.testing.assert_allclose(fit["jacobian"], expected, rtol=3e-12, atol=1e-13)
    rss = np.sum((source.data[valid] - fitted[valid]) ** 2)
    covariance = np.linalg.inv(expected.T @ expected) * rss / (valid.sum() - 3)
    assert_covariance(fit["covariance"], covariance)
    assert abs(covariance[0, 2]) > 0.1 * np.sqrt(covariance[0, 0] * covariance[2, 2])
    vh, vc, vw = fit["virtual"]
    _, virtual_expected = reference(float(vh), float(vc / hz), float(vw / hz))
    np.testing.assert_allclose(
        fit["virtual_jacobian"], virtual_expected[valid], rtol=3e-12, atol=1e-13
    )
    assert specs == saved
    np.testing.assert_array_equal(source.data, original.data)
    np.testing.assert_array_equal(source.mask, original.mask)
    for dim in source.dims:
        np.testing.assert_array_equal(
            getattr(source, dim).data, getattr(original, dim).data
        )
    legacy = scp.Optimize(peaks=specs).fit(source)
    result = legacy.result
    assert result.parameter_names == fit["layout"].free_names
    lh, lc, lw = result._parameter_values
    legacy_fitted, legacy_expected = reference(
        float(lh), float((lc - center0) / hz), float(lw / hz)
    )
    legacy_expected = legacy_expected[valid]
    np.testing.assert_allclose(legacy.jacobian, legacy_expected, rtol=3e-12, atol=1e-13)
    legacy_residual = (source.data - legacy_fitted)[valid]
    legacy_covariance = (
        np.linalg.inv(legacy_expected.T @ legacy_expected)
        * (legacy_residual @ legacy_residual)
        / (valid.sum() - 3)
    )
    assert_covariance(result.covariance, legacy_covariance)
    if not origin:
        oracle = least_squares(
            lambda p: (source.data - reference(*p)[0])[valid],
            [1.3 * intensity, 0, 3.6],
            jac=lambda p: reference(*p)[1][valid] * np.array([1, hz, hz]),
            bounds=([0.1 * intensity, -2, 1], [3 * intensity, 2, 7]),
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        np.testing.assert_allclose(
            [h, physical_center, w / hz], oracle.x, rtol=3e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            result._parameter_values, fit["values"], rtol=3e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            result.covariance, fit["covariance"], rtol=2e-5, atol=1e-15
        )


def height_problem(intensity=1):
    x = scp.Coord.linspace(-4, 4, 25, units="Hz", rounding=False)[::-1]
    y = scp.Coord.linspace(-1, 1, 9, units="s", rounding=False)
    centers, widths = [-0.7, 0.8], [2.3, 3]
    columns = np.stack(
        [
            np.exp(-4 * np.log(2) * ((x.data[None, :] - c) / w) ** 2)
            / (1 + (y.data[:, None]) ** 2)
            for c, w in zip(centers, widths, strict=True)
        ],
        axis=-1,
    )
    values = columns @ (np.array([1.2, 0.8]) * intensity)
    values += 0.0002 * intensity * np.cos(x.data[None, :] + y.data[:, None])
    source = scp.NDDataset(
        values, coordset=[y, x], units="V" if intensity == 1 else "mV"
    )
    source[1:3, 3:6] = scp.MASKED
    declaration = {"value": 0.5 * intensity, "bounds": (0.1 * intensity, 4 * intensity)}
    specs = [
        {
            "height": declaration,
            "axes": {
                "x": {"model": "gaussian", "center": fixed(c), "width": fixed(w)},
                "y": {"model": "lorentzian", "center": fixed(0), "width": fixed(2)},
            },
        }
        for c, w in zip(centers, widths, strict=True)
    ]
    return source, specs, columns


@pytest.mark.parametrize("normalization", [1e-5, 1, 1e5])
@pytest.mark.parametrize("intensity", [1, 1000])
def test_log_heights_are_independent_and_match_correlated_linear_inference(
    normalization, intensity
):
    source, specs, columns = height_problem(intensity)
    saved = deepcopy(specs)
    fit = qualify(source, specs, normalization, height_log=True)
    assert fit["solution"].success and fit["reason"] is None
    assert fit["layout"].free_names == ("peaks[0].height", "peaks[1].height")
    assert len(fit["layout"].parameters) == 10
    design = columns[fit["valid"]]
    target = source.data[fit["valid"]]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ expected
    covariance = (
        np.linalg.inv(design.T @ design) * (residual @ residual) / (target.size - 2)
    )
    np.testing.assert_allclose(fit["values"], expected, rtol=1e-7)
    np.testing.assert_allclose(fit["jacobian"], -design, rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(fit["virtual_jacobian"], -design, rtol=2e-13, atol=1e-14)
    assert_covariance(fit["covariance"], covariance, rtol=2e-7)
    first = fit["fitted"].copy()
    second_source = source * 0.8
    second = qualify(second_source, specs, normalization, height_log=True)
    np.testing.assert_allclose(second["values"], 0.8 * fit["values"], rtol=2e-7)
    np.testing.assert_array_equal(fit["fitted"], first)
    assert specs[0]["height"] is specs[1]["height"]
    assert specs == saved
    assert fit["space"].layout.free_values == (0.5 * intensity, 0.5 * intensity)


def test_active_bound_keeps_physical_inference_exclusion():
    source, specs, *_ = problem()
    specs[0]["height"] = {"value": 0.5, "bounds": (0.1, 0.8)}
    for axis in specs[0]["axes"].values():
        for key in ("center", "width"):
            value = axis[key].get("value")
            axis[key] = fixed(value)
    fit = qualify(source, specs, 1e5, height_log=True)
    assert fit["solution"].success
    assert fit["covariance"] is None and fit["reason"] == "active_bounds"
    assert np.isfinite(fit["jacobian"]).all()


def test_all_fixed_context_has_no_coordinate_vector_and_rejects_shared_syntax():
    source, specs, *_ = problem()
    for peak in specs:
        peak["height"]["fixed"] = True
        for axis in peak["axes"].values():
            for key in ("center", "width"):
                axis[key]["fixed"] = True
    _, units = backend._grid(source)
    _, free, layout = backend._prepare_peak_layout(specs, source.units, units)
    space = _CenteredParameterCoordinates(layout, (), (), ())
    assert not free and len(space.layout.parameters) == 5
    assert space.initial == space.bounds == ()
    specs[0]["axes"]["x"]["center"] = {"shared": "position"}
    with pytest.raises(ValueError, match="Invalid parameter specification"):
        backend._prepare_peak_layout(specs, source.units, units)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("model", ["gaussian", "lorentzian"])
def test_negative_height_remains_linear_with_log_width(ndim, model):
    source, specs, reference, center0, hz, _ = problem(ndim, model, 2**53)
    source = -source
    specs[0]["height"] = {
        "value": -1.3 * scp.ur.V,
        "bounds": (-3 * scp.ur.V, -0.1 * scp.ur.V),
    }
    fit = qualify(source, specs, 2)
    assert fit["solution"].success and fit["reason"] is None
    h, c, w = fit["values"]
    assert h < 0
    assert fit["space"].transforms == ("lin", "lin", "log")
    fitted, jacobian = reference(float(h), float((c - center0) / hz), float(w / hz))
    expected = jacobian[fit["valid"]]
    np.testing.assert_allclose(fit["jacobian"], expected, rtol=3e-12, atol=1e-13)
    residual = (source.data - fitted)[fit["valid"]]
    covariance = (
        np.linalg.inv(expected.T @ expected)
        * (residual @ residual)
        / (residual.size - 3)
    )
    assert_covariance(fit["covariance"], covariance)


def test_exact_integer_context_retains_platform_precision_boundary():
    exact = 2**53 + 1
    specs = [
        {
            "height": fixed(1),
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {"value": exact, "bounds": (exact - 2, exact + 2)},
                    "width": fixed(2),
                },
                "y": {"model": "lorentzian", "center": fixed(0), "width": fixed(2)},
            },
        }
    ]
    _, _, layout = backend._prepare_peak_layout(specs, None, {"x": None, "y": None})
    assert layout.free_values == (exact,)
    assert isinstance(layout.free_values[0], int)
    if np.finfo(np.longdouble).nmant < 53:
        with pytest.raises(ValueError, match="Integer magnitude loses precision"):
            _CenteredParameterCoordinates(layout, (2**53,))
    else:
        space = _CenteredParameterCoordinates(layout, (2**53,))
        assert int(space.to_physical(space.initial)[0]) == exact
        np.testing.assert_array_equal(space.bounds, [(-1, 3)])
