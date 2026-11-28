"""Qualify copied native center origins independently of solver conditioning."""

from copy import deepcopy

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting import _optimize_profiles as backend
from spectrochempy.analysis.curvefitting._centered_coordinates import (
    _CenteredParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from tests.test_analysis.test_curvefitting.test_native_profile_coordinates import fixed
from tests.test_analysis.test_curvefitting.test_native_profile_coordinates import flat


def origin_problem(
    initial="0.1",
    bounds=(-4, 4),
    *,
    unit="Hz",
    shared=False,
    model="gaussian",
    normalization=1,
    widths=(1, 2),
):
    factor = np.longdouble(1 if unit == "Hz" else 0.001)
    x = scp.Coord.linspace(-2, 3, 23, units="Hz", rounding=False)
    y = scp.Coord([-0.002, 0, 0.002], units="kHz")
    position = {
        "value": np.longdouble(initial) * factor,
        "bounds": tuple(
            None if b is None else np.longdouble(str(b)) * factor for b in bounds
        ),
    }
    if not shared:
        x = x.to(unit)
    xaxis = {
        "model": model,
        "center": {"shared": "position"} if shared else position,
        "width": fixed(widths[0] if shared else widths[0] * factor),
    }
    yaxis = (
        {
            "model": model,
            "center": {"shared": "position"},
            "width": fixed(widths[1] / 1000),
        }
        if shared
        else {"model": flat, "parameters": {}}
    )
    if model in ("voigt", "pseudovoigt"):
        xaxis["ratio"] = fixed(0.4)
        if shared:
            yaxis["ratio"] = fixed(0.4)
    specs = [{"height": fixed(1), "axes": {"x": xaxis, "y": yaxis}}]
    definition = {"position": {**position, "unit": unit}} if shared else {}
    tx = x.to("Hz").data[None, :]
    ty = y.to("Hz").data[:, None]
    values = np.exp(-4 * np.log(2) * (tx - 1.1) ** 2)
    values = values * (
        np.exp(-4 * np.log(2) * ((ty - 1.1) / 2) ** 2) if shared else np.ones((3, 1))
    )
    source = scp.NDDataset(values, coordset=[y, x], units="V")
    mask = np.zeros(source.shape, dtype=bool)
    mask[0, 0] = True
    source.mask = mask
    _, units = backend._grid(source)
    _, _, _, layout = backend._prepare_profile_layout(
        specs, definition, source.units, units
    )
    return (
        source,
        specs,
        definition,
        _ParameterLayout(layout.parameters, (normalization,)),
        factor,
    )


@pytest.mark.parametrize(
    "case",
    [
        ("0.1", (-4, 4), 53),
        ("0", (-4.1, 4), 53),
        ("0.1", (None, None), 80),
        ("0.1", (0.1, 4), 53),
    ],
)
@pytest.mark.parametrize("unit", ["Hz", "kHz"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("normalization", [0.5, 1e5])
@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt", "pseudovoigt"])
def test_lost_initial_or_finite_bound_offsets_are_rejected_before_evaluation(
    case, unit, shared, normalization, model, monkeypatch
):
    initial, bounds, power = case
    source, specs, shared_spec, layout, factor = origin_problem(
        initial,
        bounds,
        unit=unit,
        shared=shared,
        model=model,
        normalization=normalization,
    )
    space = _CenteredParameterCoordinates(layout, (np.longdouble(2**power) * factor,))
    before = deepcopy((specs, shared_spec, space))

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Origin precision must be checked before evaluation or optimization"
        )

    monkeypatch.setattr(backend, "least_squares", forbidden)
    monkeypatch.setattr(backend, "evaluate_profiles", forbidden)
    with pytest.raises(
        ValueError,
        match="Native center origin loses anchor offsets relative to the axis width",
    ):
        backend.fit_profiles(
            source, specs, 100, shared_spec, parameter_coordinates=space
        )
    assert before == (specs, shared_spec, space)


class ReachedSolverError(Exception):
    pass


@pytest.mark.parametrize("unit", ["Hz", "kHz"])
@pytest.mark.parametrize("normalization", [0.5, 2])
@pytest.mark.parametrize("narrow_axis", [None, "x", "y"])
def test_shared_precision_uses_narrowest_converted_native_width(
    unit, normalization, narrow_axis, monkeypatch
):
    widths = (1e-6 if narrow_axis == "x" else 1, 1e-6 if narrow_axis == "y" else 2)
    source, specs, shared, layout, factor = origin_problem(
        unit=unit, shared=True, normalization=normalization, widths=widths
    )
    space = _CenteredParameterCoordinates(layout, (factor,))

    def reached(*args, **kwargs):
        raise ReachedSolverError

    monkeypatch.setattr(backend, "least_squares", reached)
    with pytest.raises(
        ReachedSolverError if narrow_axis is None else ValueError,
        match=None if narrow_axis is None else "origin loses anchor offsets",
    ):
        backend.fit_profiles(source, specs, 100, shared, parameter_coordinates=space)


def test_callable_only_parameter_does_not_acquire_native_center_scale(monkeypatch):
    source, specs, shared, _, _ = origin_problem()

    def personal(axis, center):
        return np.exp(-4 * np.log(2) * (axis - center) ** 2)

    specs[0]["axes"]["x"] = {
        "model": personal,
        "parameters": {"center": {"value": 0.1, "bounds": (-4, 4), "unit": "Hz"}},
    }
    _, units = backend._grid(source)
    _, _, _, layout = backend._prepare_profile_layout(
        specs, shared, source.units, units
    )
    space = _CenteredParameterCoordinates(layout, (1,))

    def forbidden(*args, **kwargs):
        pytest.fail(
            "A callable parameter name does not qualify a centered native binding"
        )

    monkeypatch.setattr(backend, "least_squares", forbidden)
    with pytest.raises(
        ValueError, match="Only free native centers support explicit origins"
    ):
        backend.fit_profiles(source, specs, 100, shared, parameter_coordinates=space)
