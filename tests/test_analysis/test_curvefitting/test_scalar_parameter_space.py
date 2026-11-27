"""Parity of plan-backed scalar vectors with the historical parameter adapters."""

from copy import deepcopy

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._parameter_plan import _compile_parameter_plan
from spectrochempy.analysis.curvefitting._parameter_space import _internal_vector
from spectrochempy.analysis.curvefitting._parameter_space import _prepared_parameters
from spectrochempy.analysis.curvefitting._parameter_space import _restore_external
from spectrochempy.analysis.curvefitting._parameter_space import _restore_physical
from spectrochempy.analysis.curvefitting._parameter_space import _ScalarParameterSpace
from spectrochempy.analysis.curvefitting._parameter_space import _transform_artifacts
from spectrochempy.analysis.curvefitting.optimize import _optimize


@pytest.fixture
def definition():
    return _FitModelSpec(
        common_params={
            "z": _ParamSpec("z", 7.0),
            "source": _ParamSpec("source", -1.0, bounds=(-4, 4)),
            "fixed": _ParamSpec("fixed", 3.0, vary=False),
            "linked": _ParamSpec("linked", vary=False, reference="source * 2"),
        },
        components=[
            _ComponentSpec(
                "peak_with_under",
                params={
                    "width": _ParamSpec("width", 2.0, bounds=(1, None)),
                    "strength": _ParamSpec("strength", 3.0, bounds=(None, 5)),
                    "shared": _ParamSpec("shared", vary=False, reference="linked"),
                },
            )
        ],
    )


def magnitudes(spec):
    return {p.name: p.value for p in _prepared_parameters(spec)}


@pytest.mark.parametrize("fold", [-1.5 * np.pi, -1.0, 0.0, 0.5 * np.pi, 3 * np.pi])
def test_internal_restore_and_signed_artifacts_match_legacy(definition, fold):
    space = _ScalarParameterSpace.from_plan(_compile_parameter_plan(definition))
    vector, names = _internal_vector(definition)
    planned, planned_names = space.internal_vector(definition)
    np.testing.assert_array_equal(planned, vector)
    assert planned_names == names
    trial = np.array([fold, -1.1, 0.4, -2.0])
    actual, expected = deepcopy(definition), deepcopy(definition)
    assert space.restore(actual, trial) is actual
    _restore_external(expected, trial, names)
    assert magnitudes(actual) == magnitudes(expected)
    assert space.transform_artifacts(trial) == _transform_artifacts(
        expected, trial, names
    )
    np.testing.assert_array_equal(
        space.physical_values(actual),
        [p.value for p in _prepared_parameters(expected) if not p.fixed],
    )
    assert actual.common_params["fixed"].value == 3
    assert actual.common_params["linked"].reference == "source * 2"
    assert space.physical_bounds == [
        (-4, 4),
        (-np.inf, 5),
        (1, np.inf),
        (-np.inf, np.inf),
    ]


def test_physical_restore_uses_each_trial_and_preserves_initial_snapshot(definition):
    plan = _compile_parameter_plan(definition)
    space = _ScalarParameterSpace.from_plan(plan)
    first, second = deepcopy(definition), deepcopy(definition)
    names = plan.free_names
    for trial, values in [(first, [0.2, 3.1, 2.2, 9]), (second, [-2, 4, 1.5, 1])]:
        expected = deepcopy(definition)
        _restore_physical(expected, values, names)
        space.restore(trial, values, physical=True)
        assert magnitudes(trial) == magnitudes(expected)
    assert first.common_params["source"].value == 0.2
    assert second.common_params["source"].value == -2
    assert definition.common_params["source"].value == -1
    assert next(p.value for p in plan.parameters if p.name == "source") == -1
    planned, _ = space.internal_vector(second)
    expected, _ = _internal_vector(second)
    np.testing.assert_array_equal(planned, expected)


@pytest.mark.parametrize("physical", [False, True])
def test_incomplete_vectors_fail_before_mutating_a_trial(definition, physical):
    space = _ScalarParameterSpace.from_plan(_compile_parameter_plan(definition))
    initial = magnitudes(definition)
    with pytest.raises(ValueError, match="vector length"):
        space.restore(definition, [1.0], physical=physical)
    assert magnitudes(definition) == initial


@pytest.mark.parametrize("bounds", [(None, None), (-1e20, 1e20), (-np.inf, np.inf)])
def test_sentinel_bounds_keep_historical_interpretation(bounds):
    spec = _FitModelSpec(common_params={"x": _ParamSpec("x", 2.0, bounds=bounds)})
    space = _ScalarParameterSpace.from_plan(_compile_parameter_plan(spec))
    assert space.physical_bounds == [(-np.inf, np.inf)]
    assert space.internal_vector(spec) == _internal_vector(spec)
    assert space.transform_artifacts([2.0]) == _transform_artifacts(spec, [2.0], ["x"])


def test_empty_free_vector_keeps_fixed_values():
    spec = _FitModelSpec(common_params={"fixed": _ParamSpec("fixed", 2, vary=False)})
    space = _ScalarParameterSpace.from_plan(_compile_parameter_plan(spec))
    assert space.internal_vector(spec) == ([], [])
    assert space.restore(spec, []) is spec
    assert space.physical_bounds == []
    assert space.physical_values(spec).shape == (0,)
    assert space.transform_artifacts([]) == {
        "parameter_transform_derivatives": (),
        "near_parameter_bounds": False,
    }
    assert spec.common_params["fixed"].value == 2


@pytest.mark.parametrize(
    "method, bounds_mode",
    [
        ("least_squares", "transform"),
        ("trf", "transform"),
        ("simplex", "transform"),
        ("least_squares", "native"),
    ],
)
def test_solver_vectors_outputs_and_physical_diagnostics_match(
    definition, method, bounds_mode
):
    x = np.linspace(-1, 1, 31)
    design = np.column_stack([np.ones(x.size), x, x**2, np.sin(x)])
    observed = design @ [0.6, 2.2, 2.4, -0.3] + 0.001 * np.cos(7 * x)
    outcomes, histories = [], []
    for prepared in (False, True):
        trial = deepcopy(definition)
        history = []

        def objective(spec, data, history=history):
            vector = [p.value for _, p in spec._iter_varying()]
            history.append(tuple(vector))
            residual = design @ vector - data
            return float(residual @ residual) if method == "simplex" else residual

        plan = _compile_parameter_plan(trial) if prepared else None
        outcomes.append(
            _optimize(
                objective,
                trial,
                args=(observed,),
                method=method,
                maxiter=3000,
                maxfun=10000,
                parameter_plan=plan,
                bounds_mode=bounds_mode,
            )
        )
        histories.append(history)
    before, after = outcomes
    assert magnitudes(before[0]) == magnitudes(after[0])
    assert before[1:3] == after[1:3]
    for key in before[3]:
        np.testing.assert_array_equal(before[3][key], after[3][key])
    assert histories[0] == histories[1]


def test_unqualified_plan_keeps_historical_optimizer_path(monkeypatch):
    spec = _FitModelSpec(
        common_params={
            "a": _ParamSpec("a", 1.0, bounds=(0, 4)),
            "square": _ParamSpec("square", vary=False, reference="a**2"),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert plan.fallback

    def forbidden(*args):
        pytest.fail("Unqualified plans must retain the historical adapters")

    monkeypatch.setattr(_ScalarParameterSpace, "from_plan", forbidden)
    fitted, _, _, _ = _optimize(
        lambda values, data: (values.common_params["a"].value - 2) * data,
        spec,
        args=(np.linspace(0, 1, 31),),
        parameter_plan=plan,
    )
    assert fitted.common_params["a"].value == pytest.approx(2)


@pytest.mark.parametrize("initial", [0, 0.5])
@pytest.mark.parametrize("bounds_mode", ["transform", "native"])
def test_estimator_uses_plan_for_vectors_and_feasible_physical_probes(
    monkeypatch, initial, bounds_mode
):
    import spectrochempy.analysis.curvefitting.optimize as module

    def line(x, slope):
        return slope * x

    opt = scp.Optimize(
        script="",
        usermodels={"myline": line},
        autobase=False,
        autoampl=False,
        amplitude_mode="area",
        method="least_squares",
        bounds_mode=bounds_mode,
    )
    opt.script = f"""
COMMON:
$ a: {initial}, 0, none
> twice: 2*a
MODEL: line
shape: myline
> slope: twice
"""

    def forbidden(*args, **kwargs):
        pytest.fail("Qualified Optimize fits must use the prepared vector adapter")

    for helper in (
        "_internal_vector",
        "_restore_external",
        "_restore_native",
        "_restore_physical",
        "_transform_artifacts",
    ):
        monkeypatch.setattr(module, helper, forbidden)
    x = np.linspace(0, 1, 31)
    opt.fit(scp.NDDataset(2 * x, coordset=[scp.Coord(x)]))
    diagnostics = opt.result.diagnostics
    if initial == 0 and bounds_mode == "transform":
        assert diagnostics["physical_stationarity_status"] == "not_satisfied"
        assert diagnostics["stationarity_evaluations"] == 3
        assert opt.fp["a"] == 0
    else:
        assert opt.fp["a"] == pytest.approx(1, abs=1e-4)
        assert diagnostics["physical_stationarity_status"] == "satisfied"
