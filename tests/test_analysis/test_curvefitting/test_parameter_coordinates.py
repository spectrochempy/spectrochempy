"""Numerical qualification of opt-in coordinates, independently of public syntax."""

import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.optimize import least_squares

from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting._prepared_parameter import _PreparedParameter
from spectrochempy.core.units import ur


def coordinates(value=2.0, bounds=(0.01, 100), reference=1.0, scale=1.0):
    layout = _ParameterLayout(
        (_PreparedParameter(("common", "k"), "k", value, bounds),), (scale,)
    )
    return _ParameterCoordinates(layout, ("log",), (reference,))


def test_mixed_layout_keeps_fixed_records_out_of_coordinates():
    layout = _ParameterLayout(
        (
            _PreparedParameter(("common", "fixed"), "fixed", "linked", fixed=True),
            _PreparedParameter(("common", "k"), "k", 0.2, (0.01, 10), unit=ur.s**-1),
            _PreparedParameter(("common", "b"), "b", -3, (None, 5), unit=ur.V),
        ),
        (2, 0.5),
    )
    space = _ParameterCoordinates(layout, ["log", "lin"], [0.1, None])
    assert layout.free_names == ("k", "b")
    np.testing.assert_allclose(space.initial, [np.log(2) / 2, -6])
    np.testing.assert_allclose(space.to_physical(space.initial), [0.2, -3])
    np.testing.assert_allclose(space.derivatives(space.initial), [0.4, 0.5])
    np.testing.assert_allclose(
        space.bounds, [[np.log(0.1) / 2, np.log(100) / 2], [-np.inf, 10]]
    )
    for snapshot in (copy.deepcopy(space), pickle.loads(pickle.dumps(space))):  # noqa: S301
        assert snapshot == space
    with pytest.raises(FrozenInstanceError):
        space.references = (2, None)
    output = space.to_physical(space.initial)
    output[0] = 500
    assert space.to_physical(space.initial)[0] == pytest.approx(0.2)


@pytest.mark.parametrize("value", [1e-300, 1e-120, 1e-12, 1, 1e12, 1e120, 1e300])
@pytest.mark.parametrize("reference", [1e-300, 1, 1e300])
def test_log_roundtrip_without_intermediate_ratio_overflow(value, reference):
    space = coordinates(value, (1e-305, None), reference)
    np.testing.assert_allclose(
        space.to_physical(space.initial), [value], rtol=5e-13, atol=0
    )
    np.testing.assert_allclose(
        space.derivatives(space.initial), [value], rtol=5e-13, atol=0
    )


@pytest.mark.parametrize("bad", [None, 0, -1, np.nan, np.inf, 1j, True, "1"])
def test_log_requires_an_explicit_positive_finite_real_reference(bad):
    with pytest.raises(ValueError):
        coordinates(reference=bad)


@pytest.mark.parametrize(
    "bounds", [(None, 10), (0, 10), (-1, 10), (2, 2), (10, 1), (1, np.nan)]
)
def test_invalid_domains_are_rejected_without_clipping(bounds):
    with pytest.raises(ValueError):
        coordinates(bounds=bounds)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, 200])
def test_initial_values_must_be_feasible_and_finite(value):
    with pytest.raises(ValueError):
        coordinates(value)


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, True])
def test_invalid_conditioning_is_separate_from_log(scale):
    with pytest.raises(ValueError):
        coordinates(scale=scale)


def test_reference_and_scale_inputs_are_snapshotted():
    scale = np.array(2.0)
    reference = np.array(0.5)
    space = coordinates(scale=scale, reference=reference)
    before = space.to_physical(space.initial)
    scale[...] = 10
    reference[...] = 10
    np.testing.assert_array_equal(space.to_physical(space.initial), before)


def test_linear_defaults_and_empty_free_layout():
    layout = _ParameterLayout((_PreparedParameter(("k",), "k", -2),), (2,))
    space = _ParameterCoordinates(layout)
    assert space.initial == (-1,)
    np.testing.assert_array_equal(space.to_physical([3]), [6])
    empty = _ParameterCoordinates(_ParameterLayout(()))
    assert empty.initial == empty.bounds == ()
    assert empty.to_physical([]).shape == (0,)
    assert empty.physical_jacobian(np.empty((3, 0)), []).shape == (3, 0)
    assert empty.physical_covariance(np.empty((0, 0)), []).shape == (0, 0)


def test_reject_unsupported_transforms_and_misaligned_configuration():
    layout = coordinates().layout
    for transforms, references in [
        ((), (1,)),
        (("log",), ()),
        (("sqrt",), (1,)),
        (("lin",), (1,)),
    ]:
        with pytest.raises(ValueError):
            _ParameterCoordinates(layout, transforms, references)


@pytest.mark.parametrize("value", [0.01, 100])
def test_positive_physical_endpoints_are_not_shifted(value):
    space = coordinates(value)
    assert space.initial[0] == space.bounds[0][int(value == 100)]
    assert space.to_physical(space.initial)[0] == pytest.approx(value)


def test_unit_magnitude_conversion_preserves_dimensionless_log_coordinate():
    k = 0.2 * ur.s**-1
    reference = 0.1 * ur.s**-1
    a = coordinates(k.magnitude, (0.01, 10), reference.magnitude)
    b = coordinates(
        k.to("1/ms").magnitude, (1e-5, 0.01), reference.to("1/ms").magnitude
    )
    np.testing.assert_allclose(a.initial, b.initial, rtol=2e-15)
    np.testing.assert_allclose(a.bounds, b.bounds, rtol=2e-15)
    np.testing.assert_allclose(
        a.derivatives(a.initial), 1000 * b.derivatives(b.initial)
    )


@pytest.mark.parametrize("value", [-1000, 1000, np.nan, np.inf])
def test_unrepresentable_solver_proposals_raise_without_warnings(value):
    with pytest.raises(ValueError):
        coordinates().to_physical([value])


def test_nonpositive_physical_proposals_and_bad_shapes_are_rejected():
    space = coordinates()
    for values in ([0], [-1], [1, 2], [[1]], [1j], [True]):
        with pytest.raises(ValueError):
            space.to_solver(values)
    with pytest.raises(ValueError):
        space.physical_jacobian([1], space.initial)
    with pytest.raises(ValueError):
        space.physical_covariance([[1, 2]], space.initial)


def test_linear_conditioning_does_not_silently_erase_or_overflow_values():
    layout = _ParameterLayout((_PreparedParameter(("p",), "p", 0),), (1e-100,))
    space = _ParameterCoordinates(layout)
    with pytest.raises(ValueError, match="underflow"):
        space.to_physical([1e-300])
    with pytest.raises(ValueError, match="not representable"):
        space.to_solver([1e300])


def test_log_near_reference_retains_narrow_bounds():
    reference = 2.0
    low = np.nextafter(reference, 0)
    high = np.nextafter(reference, np.inf)
    space = coordinates(reference, (low, high), reference)
    assert space.bounds[0][0] < 0 < space.bounds[0][1]
    assert space.initial == (0,)


@pytest.mark.parametrize("reference", [5.0, 1e20, 1.5417004529495173e-307])
def test_finite_maximum_initial_and_endpoint_have_exact_inverse_anchors(reference):
    maximum = np.finfo(float).max
    space = coordinates(maximum, (1, maximum), reference)
    np.testing.assert_array_equal(space.to_physical(space.initial), [maximum])
    np.testing.assert_array_equal(space.derivatives(space.initial), [maximum])
    with pytest.raises(ValueError, match="overflow"):
        space.to_physical([space.initial[0] + 1])
    inside = coordinates(2, (1, maximum), reference)
    np.testing.assert_array_equal(inside.to_physical([inside.bounds[0][1]]), [maximum])


def test_minimum_subnormal_endpoint_is_retained_without_clipping():
    minimum = np.nextafter(0.0, 1.0)
    space = coordinates(minimum, (minimum, 1), 1e300)
    np.testing.assert_array_equal(space.to_physical(space.initial), [minimum])
    with pytest.raises(ValueError, match="underflow"):
        space.to_physical([space.initial[0] - 2])


def test_distinct_initial_and_bound_collapsing_to_one_coordinate_are_rejected():
    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="collapse"):
        coordinates(np.nextafter(maximum, 0), (1, maximum), 5)


def test_physical_jacobian_rejects_overflow_and_underflow():
    for scale, value in [(1e-200, 1e200), (1e200, 1e-200)]:
        layout = _ParameterLayout((_PreparedParameter(("p",), "p", 0),), (scale,))
        with pytest.raises(ValueError, match="not representable"):
            _ParameterCoordinates(layout).physical_jacobian([[value]], [0])


def test_matrix_conversions_retain_correlations_and_do_not_apply_scale_twice():
    layout = _ParameterLayout(
        tuple(
            _PreparedParameter((name,), name, value, bounds)
            for name, value, bounds in [("k", 3, (1, 10)), ("b", -2, (None, None))]
        ),
        (2, 4),
    )
    space = _ParameterCoordinates(layout, ("log", "lin"), (1, None))
    physical_jac = np.array([[1.0, 2], [3, -4], [1, 1]])
    solver_jac = physical_jac * [6, 4]
    covariance = np.array([[2.0, -0.5], [-0.5, 3]])
    np.testing.assert_allclose(
        space.physical_jacobian(solver_jac, space.initial), physical_jac
    )
    np.testing.assert_allclose(
        space.physical_covariance(covariance, space.initial), [[72, -12], [-12, 48]]
    )
    step = 1e-6
    finite_difference = (
        space.to_physical(np.array(space.initial) + step)
        - space.to_physical(np.array(space.initial) - step)
    ) / (2 * step)
    np.testing.assert_allclose(finite_difference, [6, 4], rtol=1e-9)


def test_covariance_avoids_intermediate_overflow_but_refuses_lost_magnitudes():
    layout = _ParameterLayout(
        tuple(_PreparedParameter((str(i),), str(i), 0) for i in range(2)),
        (1e200, 1e-200),
    )
    space = _ParameterCoordinates(layout)
    covariance = [[0, 1e200], [1e200, 0]]
    np.testing.assert_allclose(
        space.physical_covariance(covariance, [0, 0]), covariance
    )
    for matrix in ([[1, 0], [0, 0]], [[0, 0], [0, 1]]):
        with pytest.raises(ValueError, match="not representable"):
            space.physical_covariance(matrix, [0, 0])


def test_mixed_log_fit_and_covariance_against_weighted_linear_reference():
    scales = np.array([1e-9, 1e9, 1.0])
    truth = np.array([0.7, 2.2, -0.3]) * scales
    x = np.linspace(-1, 1, 51)
    design = np.column_stack([np.ones(x.size), x, x * x])
    weights = np.linspace(0.5, 2, x.size)
    target = design @ (truth / scales) + 0.001 * np.cos(7 * x)
    valid = np.ones(x.size, dtype=bool)
    valid[5:9] = False
    matrix = (weights[:, None] * design)[valid]
    data = (weights * target)[valid]
    reference = np.linalg.lstsq(matrix, data, rcond=None)[0] * scales
    layout = _ParameterLayout(
        tuple(
            _PreparedParameter((str(i),), str(i), initial, bounds)
            for i, initial, bounds in [
                (0, 0.2e-9, (1e-12, 1e-8)),
                (1, 0.5e9, (1e6, 1e10)),
                (2, 0.0, (-2, 2)),
            ]
        ),
        (1, 2, 0.5),
    )
    space = _ParameterCoordinates(layout, ("log", "log", "lin"), (1e-9, 1e9, None))

    def residual(z):
        return matrix @ (space.to_physical(z) / scales) - data

    def jacobian(z):
        return matrix * (space.derivatives(z) / scales)

    solution = least_squares(
        residual,
        space.initial,
        jac=jacobian,
        bounds=np.array(space.bounds).T,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    assert solution.success
    np.testing.assert_allclose(space.to_physical(solution.x), reference, rtol=1e-10)
    variance = solution.fun @ solution.fun / (len(data) - 3)
    solver_covariance = np.linalg.inv(solution.jac.T @ solution.jac) * variance
    expected_covariance = (
        np.linalg.inv(matrix.T @ matrix) * variance * scales[:, None] * scales
    )
    np.testing.assert_allclose(
        space.physical_covariance(solver_covariance, solution.x),
        expected_covariance,
        rtol=1e-10,
        atol=0,
    )
    np.testing.assert_allclose(
        space.physical_jacobian(solution.jac, solution.x), matrix / scales, rtol=2e-15
    )
