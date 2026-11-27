"""Qualify private affine coordinates without selecting a native fitting engine."""

import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._centered_coordinates import (
    _CenteredParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._optimize_profiles import (
    _prepare_profile_layout,
)
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting._prepared_parameter import _PreparedParameter


def center_layout(value, bounds=(None, None), scale=1):
    return _ParameterLayout(
        (_PreparedParameter(("center",), "center", value, bounds),), (scale,)
    )


@pytest.mark.parametrize("origin", [2**60 + 1, -(2**60 + 1)])
def test_integer_origins_are_subtracted_before_float64_coercion(origin):
    layout = center_layout(origin, (origin - 2, origin + 2), 0.5)
    if int(np.longdouble(origin)) != origin:
        with pytest.raises(ValueError, match="precision"):
            _CenteredParameterCoordinates(layout, (origin,))
        return
    space = _CenteredParameterCoordinates(layout, (origin,))
    assert space.initial == (0,)
    assert space.bounds == ((-4, 4),)
    assert space.to_solver([origin + 1])[0] == 2
    assert int(space.to_physical([2])[0]) == origin + 1
    assert space.to_physical([2]).dtype == np.dtype(np.longdouble)
    assert space.to_offsets([2])[0] == 1
    with pytest.raises(ValueError, match="Physical bounds|collapse"):
        _ParameterCoordinates(layout)


def test_distinct_integer_anchors_cannot_be_silently_rounded():
    origin = 2**62
    layout = center_layout(origin + 1, (origin - 2, origin + 2))
    if int(np.longdouble(origin + 1)) == origin + 1:
        space = _CenteredParameterCoordinates(layout, (origin,))
        assert space.initial == (1,)
        assert int(space.to_physical(space.initial)[0]) == origin + 1
    else:
        with pytest.raises(ValueError, match="precision"):
            _CenteredParameterCoordinates(layout, (origin,))


def test_large_python_integer_support_depends_on_precision_not_numpy_dtype():
    origin = 2**100
    step = 2**60
    space = _CenteredParameterCoordinates(
        center_layout(origin + step, (origin - 2 * step, origin + 2 * step), step),
        (origin,),
    )
    assert space.initial == (1,)
    assert space.bounds == ((-2, 2),)
    assert int(space.to_physical(space.initial)[0]) == origin + step
    assert space.to_solver([origin - step])[0] == -1
    for bad in (origin + 1, 2**20000):
        with pytest.raises(ValueError):
            _CenteredParameterCoordinates(center_layout(bad), (bad,))


def test_anchor_lookup_uses_the_validated_float64_solver_coordinate():
    value = np.nextafter(np.longdouble(1), np.longdouble(2))
    space = _CenteredParameterCoordinates(center_layout(value), (0,))
    np.testing.assert_array_equal(space.to_physical([value]), [value])
    np.testing.assert_array_equal(space.to_physical([float(value)]), [value])


def test_center_and_log_reference_and_conditioning_are_independent():
    origin = 2**53
    layout = _ParameterLayout(
        (
            _PreparedParameter(("fixed",), "fixed", "retained", fixed=True),
            _PreparedParameter(("center",), "center", origin, (origin - 4, origin + 4)),
            _PreparedParameter(("width",), "width", 2.0, (0.25, 8.0)),
            _PreparedParameter(("base",), "base", -1.0, (-5.0, 5.0)),
        ),
        (2.0, 0.5, 0.25),
    )
    space = _CenteredParameterCoordinates(
        layout, (origin, None, None), ("lin", "log", "lin"), (None, 1.0, None)
    )
    np.testing.assert_allclose(space.initial, [0, np.log(2) / 0.5, -4])
    np.testing.assert_allclose(space.derivatives(space.initial), [2, 1, 0.25])
    assert space.layout.parameters[0].value == "retained"
    assert space.layout.free_names == ("center", "width", "base")
    assert space.layout.free_values[0] == origin
    np.testing.assert_array_equal(space.to_physical(space.initial), [origin, 2.0, -1.0])
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    d = np.array([2.0, 1.0, 0.25])
    np.testing.assert_allclose(
        space.physical_jacobian(matrix * d, space.initial), matrix
    )
    covariance = matrix.T @ matrix + np.eye(3)
    np.testing.assert_allclose(
        space.physical_covariance(covariance / d[:, None] / d, space.initial),
        covariance,
    )


def test_configuration_is_snapshotted_and_results_are_independent():
    origin = np.array(1e6)
    value = np.array(1e6 + 0.25)
    layout = center_layout(value, (1e6 - 1, 1e6 + 1))
    space = _CenteredParameterCoordinates(layout, [origin])
    origin[...] = value[...] = -500
    assert space.layout.free_values[0] == 1e6 + 0.25
    assert space.to_physical(space.initial)[0] == 1e6 + 0.25
    result = space.to_physical([np.array(0.5)])
    result[0] = -100
    assert space.to_physical([0.5])[0] == 1e6 + 0.5
    for snapshot in (copy.deepcopy(space), pickle.loads(pickle.dumps(space))):  # noqa: S301
        np.testing.assert_array_equal(
            snapshot.to_physical(snapshot.initial), space.to_physical(space.initial)
        )
    with pytest.raises(FrozenInstanceError):
        space.origins = (0,)


@pytest.mark.parametrize("origin", [True, 1j, np.nan, np.inf, "1", [1], 1 * scp.ur.Hz])
def test_invalid_origins_are_rejected(origin):
    with pytest.raises(ValueError):
        _CenteredParameterCoordinates(center_layout(1), (origin,))


def test_invalid_coordinates_and_lost_anchors_are_rejected():
    layout = center_layout(1, (0.1, 10))
    with pytest.raises(ValueError, match="count"):
        _CenteredParameterCoordinates(layout, ())
    with pytest.raises(ValueError, match="Only linear"):
        _CenteredParameterCoordinates(layout, (1,), ("log",), (1,))
    with pytest.raises(ValueError, match="precision"):
        _CenteredParameterCoordinates(center_layout(1, (0, 2)), (1e300,))
    with pytest.raises(ValueError):
        _CenteredParameterCoordinates(center_layout(1, (2, 3)), (1,))
    space = _CenteredParameterCoordinates(layout, (1,))
    for bad in ([True], [1j], [np.nan], [np.inf], [1, 2], [[1]], [1 * scp.ur.Hz]):
        with pytest.raises(ValueError):
            space.to_solver(bad)


def test_zero_or_absent_origins_and_empty_layout_preserve_legacy_mapping():
    layout = center_layout(-2.0, (None, 5.0), 2.0)
    legacy = _ParameterCoordinates(layout)
    for origin in (None, 0):
        space = _CenteredParameterCoordinates(layout, (origin,))
        assert space.initial == legacy.initial and space.bounds == legacy.bounds
        np.testing.assert_array_equal(space.to_solver([3.0]), legacy.to_solver([3.0]))
        np.testing.assert_array_equal(
            space.to_physical([1.5]), legacy.to_physical([1.5])
        )
    space = _CenteredParameterCoordinates(_ParameterLayout(()), ())
    assert space.initial == space.bounds == ()
    assert space.to_physical([]).shape == (0,)
    assert space.physical_jacobian(np.empty((3, 0)), []).shape == (3, 0)
    assert space.physical_covariance(np.empty((0, 0)), []).shape == (0, 0)


def test_offsets_can_resolve_motion_below_the_physical_output_ulp():
    origin = np.longdouble(2**53)
    space = _CenteredParameterCoordinates(
        center_layout(origin, (origin - 2, origin + 2)), (origin,)
    )
    tiny = float(np.spacing(origin)) / 8
    assert space.to_offsets([tiny])[0] == tiny
    assert space.to_physical([tiny])[0] == origin
    assert space.derivatives([tiny])[0] == 1


def test_distinct_extended_anchors_collapsing_in_float64_are_rejected():
    value = np.nextafter(np.longdouble(1), np.longdouble(2))
    layout = center_layout(value, (1, 2))
    if value == float(value):
        space = _CenteredParameterCoordinates(layout, (0,))
        assert space.to_physical(space.initial)[0] == value
    else:
        with pytest.raises(ValueError, match="Distinct physical anchors"):
            _CenteredParameterCoordinates(layout, (0,))


def test_range_checks_and_unbounded_endpoints_are_preserved():
    space = _CenteredParameterCoordinates(center_layout(1, (None, None)), (1,))
    assert space.bounds == ((-np.inf, np.inf),)
    for bad in ([np.nan], [np.inf], [1j], [True], [[1]]):
        with pytest.raises(ValueError):
            space.to_physical(bad)
    extreme = _CenteredParameterCoordinates(center_layout(1, scale=1e300), (0,))
    with np.errstate(all="raise"), pytest.raises(ValueError):
        extreme.to_physical([1e300])
    with pytest.raises(ValueError):
        space.physical_jacobian([[np.inf]], space.initial)
    with pytest.raises(ValueError):
        space.physical_covariance([[np.nan]], space.initial)
    layout = _ParameterLayout((_PreparedParameter(("k",), "k", 0.2, (0, 1)),))
    with pytest.raises(ValueError, match="positive finite lower bound"):
        _CenteredParameterCoordinates(layout, (None,), ("log",), (0.1,))


@pytest.mark.parametrize("coordinate_unit", ["Hz", "kHz"])
@pytest.mark.parametrize("intensity_unit", ["V", "mV"])
@pytest.mark.parametrize("origin", [0, 2**53])
def test_shared_center_and_log_width_have_physical_jacobian_and_joint_covariance(
    coordinate_unit, intensity_unit, origin
):
    factor = np.longdouble(1 if coordinate_unit == "Hz" else 0.001)
    intensity = 1 if intensity_unit == "V" else 1000
    relative = np.array([6, 4, 2, 1, 0.5, 0, -0.5, -1, -2, -3, -4])
    x = scp.Coord(
        (np.longdouble(origin) + relative) * factor,
        units=coordinate_unit,
        dtype=np.longdouble,
        rounding=False,
    )
    y = scp.Coord([0, 1, 2], units="s")

    def flat(axis):
        return np.ones_like(axis)

    def fixed(value):
        return {"value": value, "fixed": True}

    specs = [
        {
            "height": {
                "value": 1.5 * intensity,
                "bounds": (0.5 * intensity, 3 * intensity),
            },
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {"shared": "position"},
                    "width": {
                        "value": 2 * factor,
                        "bounds": (0.5 * factor, 4 * factor),
                    },
                },
                "y": {"model": flat, "parameters": {}},
            },
        },
        {
            "height": fixed(0.4 * intensity),
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {"shared": "position"},
                    "width": fixed(3 * factor),
                },
                "y": {"model": flat, "parameters": {}},
            },
        },
    ]
    shared = {
        "position": {"value": origin, "unit": "Hz", "bounds": (origin - 2, origin + 2)}
    }
    saved = copy.deepcopy((specs, shared))
    peaks, groups, _, layout = _prepare_profile_layout(
        specs, shared, scp.ur.Unit(intensity_unit), {"x": x.units, "y": y.units}
    )
    layout = _ParameterLayout(layout.parameters, (intensity, 2, 0.5))
    space = _CenteredParameterCoordinates(
        layout, (None, origin, None), ("lin", "lin", "log"), (None, None, factor)
    )
    assert [len(g["bindings"]) for g in groups] == [1, 2, 1, 1, 1]
    initial_center = peaks[0]["axes"]["x"]["parameters"]["center"]
    relative = np.asarray((x.data - initial_center) / factor, dtype=float)
    a = 4 * np.log(2)

    def curve(height, offset, width):
        return height * np.exp(
            -a * ((relative - offset) / width) ** 2
        ) + 0.4 * intensity * np.exp(-a * ((relative - offset) / 3) ** 2)

    data = np.broadcast_to(curve(2 * intensity, 0.33, 1.6), (3, len(relative))).copy()
    data += 0.001 * intensity * np.cos(relative[None, :] + np.arange(3)[:, None])
    source = scp.NDDataset(data, coordset=[y, x], units=intensity_unit)
    source[1:2, 2:4] = scp.MASKED
    source.data[0, 0] = np.nan
    valid = np.isfinite(source.data) & ~source.mask

    def residual(z):
        height, offset, width = space.to_offsets(z)
        model = np.broadcast_to(
            curve(height, offset, width / float(factor)), source.shape
        )
        return (source.data - model)[valid] / intensity

    result = least_squares(
        residual,
        space.initial,
        bounds=np.asarray(space.bounds).T,
        jac="3-point",
        x_scale=1,
        gtol=1e-12,
        ftol=1e-12,
        xtol=1e-12,
    )
    assert result.success
    physical = space.to_physical(result.x)
    z = space.to_solver(physical)
    height, _, width = physical
    for group, value in zip(
        [g for g in groups if not g["parameter"].fixed], physical, strict=True
    ):
        for target, key, convert in group["bindings"]:
            target[key] = convert(value)
    center = peaks[0]["axes"]["x"]["parameters"]["center"]
    dx = np.asarray((x.data - center) / factor, dtype=float)
    width_hz = float(width / factor)
    one = np.exp(-a * (dx / width_hz) ** 2)
    two = np.exp(-a * (dx / 3) ** 2)
    expected_model = np.broadcast_to(
        float(height) * one + 0.4 * intensity * two, source.shape
    )
    center_column = (
        -2 * a * dx * (float(height) * one / width_hz**2 + 0.4 * intensity * two / 9)
    )
    width_column = -2 * a * float(height) * dx**2 * one / width_hz**3 / float(factor)
    expected_jacobian = np.broadcast_to(
        np.stack((-one, center_column, width_column), axis=-1), (*source.shape, 3)
    )[valid]
    pivot = space.to_offsets(z)

    def represented_residual(trial):
        h, offset, w = space.to_offsets(trial)
        local = dx - (offset - pivot[1])
        model = h * np.exp(
            -a * (local / (w / float(factor))) ** 2
        ) + 0.4 * intensity * np.exp(-a * (local / 3) ** 2)
        return (source.data - np.broadcast_to(model, source.shape))[valid] / intensity

    solver_jacobian = approx_derivative(
        represented_residual, z, method="3-point", bounds=np.asarray(space.bounds).T
    )
    jacobian = space.physical_jacobian(solver_jacobian, z) * intensity
    np.testing.assert_allclose(
        jacobian, expected_jacobian, rtol=3e-6, atol=3e-8 * intensity
    )
    rss = np.sum((source.data - expected_model)[valid] ** 2)
    variance = rss / (np.count_nonzero(valid) - 3)
    covariance = space.physical_covariance(
        np.linalg.inv(solver_jacobian.T @ solver_jacobian) * variance / intensity**2, z
    )
    expected_covariance = (
        np.linalg.inv(expected_jacobian.T @ expected_jacobian) * variance
    )
    np.testing.assert_allclose(covariance, expected_covariance, rtol=3e-6, atol=0)
    assert (
        np.count_nonzero(expected_covariance - np.diag(np.diag(expected_covariance)))
        == 6
    )
    assert (specs, shared) == saved
