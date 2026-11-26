"""Shared parameter metadata without changing scalar or ND coordinate policies."""

import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting import _optimize_model
from spectrochempy.analysis.curvefitting._optimize_model import _parameter_layout
from spectrochempy.analysis.curvefitting._optimize_model import _parameters
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting._prepared_parameter import _PreparedParameter


def test_layout_preserves_order_precision_units_and_conditioning_independently():
    exact = 2**53 + 1
    precise = np.longdouble("0.1234567890123456789")
    records = [
        _PreparedParameter(("model", "z"), "z", exact, unit=scp.ur.Hz),
        _PreparedParameter(("model", "fixed"), "fixed", 2, fixed=True),
        _PreparedParameter(("model", "a"), "a", precise, bounds=(0, 1)),
    ]
    scales = [4, 0.2]
    layout = _ParameterLayout(records, scales)
    records.clear()
    scales[0] = 99
    assert layout.free_names == ("z", "a")
    assert tuple(p.identifier for p in layout.free_parameters) == (
        ("model", "z"),
        ("model", "a"),
    )
    assert layout.free_values == (exact, precise)
    assert isinstance(layout.free_values[0], int)
    assert isinstance(layout.free_values[1], np.longdouble)
    assert layout.free_bounds == ((None, None), (0, 1))
    assert layout.free_units == (scp.ur.Hz, None)
    assert layout.free_scales == (4, 0.2)
    with pytest.raises(FrozenInstanceError):
        layout.free_names = ("changed",)
    for restored in (deepcopy(layout), pickle.loads(pickle.dumps(layout))):  # noqa: S301
        assert restored == layout


@pytest.mark.parametrize("collision", ["name", "identifier"])
def test_layout_rejects_implicit_identity_or_name_sharing(collision):
    first = _PreparedParameter(("model", "x"), "x", 1.0)
    second = _PreparedParameter(
        ("model", "y") if collision == "name" else first.identifier,
        "x" if collision == "name" else "y",
        2.0,
    )
    with pytest.raises(ValueError, match=f"{collision}s must be unique"):
        _ParameterLayout((first, second))


def test_layout_default_scales_and_empty_free_vector():
    records = (_PreparedParameter(("x",), "x", 2.0),)
    assert _ParameterLayout(records).free_scales == (1.0,)
    with pytest.raises(ValueError, match="match the free parameter count"):
        _ParameterLayout(records, ())
    fixed = _ParameterLayout((_PreparedParameter(("x",), "x", 2, fixed=True),))
    assert fixed.free_names == fixed.free_values == fixed.free_bounds == ()
    assert fixed.free_units == fixed.free_scales == ()


def test_callback_normalization_and_historical_records_scales_pair():
    specs = {
        "zrate": {
            "value": 0.002 * scp.ur.kHz,
            "unit": "Hz",
            "bounds": (0.5, 4),
            "scale": 0.004 * scp.ur.kHz,
        },
        "fixed": {
            "value": 298.15 * scp.ur.K,
            "unit": "degC",
            "fixed": True,
            "scale": "historically ignored",
        },
        "aoffset": -0.25,
    }
    layout = _parameter_layout(specs)
    assert layout.free_names == ("zrate", "aoffset")
    assert layout.free_values == (2.0, -0.25)
    assert layout.free_units == (scp.ur.Hz, None)
    assert layout.free_bounds == ((0.5, 4), (-np.inf, np.inf))
    assert layout.free_scales == (4.0, 1.0)
    assert layout.parameters[1].value == pytest.approx(25)
    assert layout.parameters[0].as_legacy_tuple() == (2.0, 0.5, 4, False)
    records, scales = _parameters(specs)
    assert isinstance(records, list)
    assert tuple(records) == layout.parameters
    np.testing.assert_array_equal(scales, layout.free_scales)
    assert specs["zrate"]["value"] == 0.002 * scp.ur.kHz


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (0, "scale must be finite and positive"),
        (-1, "scale must be finite and positive"),
        (np.nan, "must be a finite real scalar"),
        (np.inf, "must be a finite real scalar"),
    ],
)
def test_callback_scale_validation_stays_at_normalization_boundary(bad, message):
    with pytest.raises(ValueError, match=message):
        _parameter_layout({"x": {"value": 2, "scale": bad}})
    fixed = _parameter_layout({"x": {"value": 2, "scale": bad, "fixed": True}})
    assert fixed.free_names == fixed.free_scales == ()


@pytest.mark.parametrize("x_scale", [None, "jac", (4.0, 0.3)])
def test_nd_declaration_order_units_scales_and_covariance_match_reference(
    monkeypatch, x_scale
):
    x = scp.Coord.linspace(-2, 2, 17, rounding=False)
    y = scp.Coord.linspace(-1, 1, 9, rounding=False)
    xx, yy = x.data[None, :], y.data[:, None]
    values = 2.5 * xx + 0.4 + 0.2 * yy**2 + 0.001 * np.cos(3 * xx + 4 * yy)
    source = scp.NDDataset(values, coordset=[y, x], units="V")
    source[2:4, 4:7] = scp.MASKED
    source.data[0, 0] = np.nan
    support = ~source.mask & np.isfinite(source.data)
    design = np.column_stack(
        [np.broadcast_to(xx, source.shape)[support], np.ones(support.sum())]
    )
    target = (source.data - 0.2 * yy**2)[support]
    reference = np.linalg.lstsq(design, target, rcond=None)[0]
    expected_residual = target - design @ reference
    expected_covariance = (
        expected_residual @ expected_residual / (support.sum() - 2)
    ) * np.linalg.inv(design.T @ design)

    captured = {}
    real_solver = _optimize_model._least_squares

    def audited(residual, initial, **options):
        captured.update(initial=initial.copy(), x_scale=deepcopy(options["x_scale"]))
        return real_solver(residual, initial, **options)

    monkeypatch.setattr(_optimize_model, "_least_squares", audited)

    def model(template, z_gain, a_offset, fixed_floor):
        return (
            z_gain * template.x.data[None, :]
            + a_offset
            + fixed_floor * template.y.data[:, None] ** 2
        )

    specs = {
        "z_gain": {
            "value": 700 * scp.ur.mV,
            "unit": "V",
            "bounds": (0, 4),
            "scale": 1700 * scp.ur.mV,
        },
        "fixed_floor": {"value": 200 * scp.ur.mV, "unit": "V", "fixed": True},
        "a_offset": {
            "value": 100 * scp.ur.mV,
            "unit": "V",
            "bounds": (-0.5, 1.2),
            "scale": 200 * scp.ur.mV,
        },
    }
    options = {} if x_scale is None else {"x_scale": x_scale}
    fit = scp.Optimize(model=model, parameters=specs, solver_options=options)
    specs["z_gain"]["value"] = 99
    fit.fit(source)
    result = fit.result
    assert result.parameter_names == ("z_gain", "a_offset")
    assert tuple(result.fitted_parameters) == ("z_gain", "fixed_floor", "a_offset")
    np.testing.assert_allclose(captured["initial"], [0.7, 0.1])
    if x_scale == "jac":
        assert captured["x_scale"] == "jac"
    else:
        np.testing.assert_array_equal(
            captured["x_scale"], (1.7, 0.2) if x_scale is None else x_scale
        )
    actual = [
        result.fitted_parameters[name].to("V").magnitude
        for name in result.parameter_names
    ]
    np.testing.assert_allclose(actual, reference, atol=1e-7)
    np.testing.assert_allclose(result.covariance, expected_covariance, rtol=1e-4)
    np.testing.assert_allclose(fit.jacobian, -design, rtol=1e-5, atol=1e-7)
    assert result.n_varying_parameters == 2
    assert result.degrees_of_freedom == int(support.sum()) - 2
    np.testing.assert_array_equal(result.fitted.mask, ~support)
    np.testing.assert_array_equal(result.residuals.mask, ~support)
