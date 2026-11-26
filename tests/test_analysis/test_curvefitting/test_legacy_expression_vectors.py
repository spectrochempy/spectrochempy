"""Qualify scalar vector preparation independently from expression compilation."""

import pickle
from copy import deepcopy

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model
from spectrochempy.analysis.curvefitting._parameter_plan import _compile_parameter_plan
from spectrochempy.analysis.curvefitting._parameter_space import _ScalarParameterSpace
from spectrochempy.analysis.curvefitting.optimize import _optimize


@pytest.mark.parametrize(
    "expression",
    ["source**2", "source % 2", "1 if source else 2", "1.e3", "2 # comment"],
)
def test_legacy_resolution_and_copy_are_independent_of_vector_qualification(
    expression, monkeypatch
):
    import spectrochempy.analysis.curvefitting._parameter_plan as module

    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", 2.0, bounds=(0.1, 4)),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert plan.fallback
    assert plan.vector_compatible
    calls = []

    def record(definition):
        calls.append(definition)
        return prepare_model(definition)

    monkeypatch.setattr(module, "prepare_model", record)
    for restored in (plan, deepcopy(plan), pickle.loads(pickle.dumps(plan))):  # noqa: S301
        trial = deepcopy(spec)
        space = _ScalarParameterSpace.from_plan(restored)
        space.restore(trial, [3.0], physical=True)
        resolved = restored.resolve(trial)
        expected = prepare_model(trial)
        assert (
            resolved.common_params["linked"].value
            == expected.common_params["linked"].value
        )
        assert trial.common_params["linked"].reference == expression
        assert spec.common_params["source"].value == 2.0
    assert len(calls) == 3


@pytest.mark.parametrize(
    "kind",
    [
        "experiments",
        "constraints",
        "array",
        "bounds",
        "duplicate_labels",
        "duplicate_keys",
    ],
)
def test_unqualified_structures_do_not_enable_prepared_vectors(kind):
    spec = _FitModelSpec(common_params={"source": _ParamSpec("source", 2.0)})
    if kind == "experiments":
        spec.expvars = ["source"]
    elif kind == "constraints":
        spec.constraints = {}
    elif kind == "array":
        spec.common_params["source"].value = [2.0]
    elif kind == "bounds":
        spec.common_params["source"].bounds = ([0.0], 4.0)
    elif kind == "duplicate_labels":
        spec.components = [_ComponentSpec("peak"), _ComponentSpec("peak")]
    else:
        spec.common_params["width_peak"] = _ParamSpec("width_peak", 2.0)
        spec.components = [
            _ComponentSpec("peak", params={"width": _ParamSpec("width", 2.0)})
        ]
    plan = _compile_parameter_plan(spec)
    assert plan.fallback
    assert not plan.vector_compatible
    with pytest.raises(ValueError, match="qualified layout"):
        _ScalarParameterSpace.from_plan(plan)


@pytest.mark.parametrize("expression", ["source**2", "source % 2", "1.e3"])
@pytest.mark.parametrize("method", ["least_squares", "trf", "simplex"])
def test_legacy_expression_fits_match_historical_vectors(expression, method):
    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", 1.5, bounds=(0.1, 4)),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
            "fixed": _ParamSpec("fixed", 0.2, vary=False),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert plan.fallback and plan.vector_compatible
    x = np.linspace(-1, 1, 31)
    target = deepcopy(spec)
    target.common_params["source"].value = 2.0
    target_linked = prepare_model(target).common_params["linked"].value
    histories = []
    outcomes = []
    for selected in (None, plan):
        history = []

        def residual(definition, data, selected=selected, history=history):
            resolved = (
                prepare_model(definition)
                if selected is None
                else selected.resolve(definition)
            )
            source = definition.common_params["source"].value
            linked = resolved.common_params["linked"].value
            history.append((source, linked))
            residuals = (source - 2) * data + (linked - target_linked) * 0.03
            return float(np.sum(residuals**2)) if method == "simplex" else residuals

        outcomes.append(
            _optimize(
                residual,
                deepcopy(spec),
                args=(x,),
                method=method,
                maxiter=3000,
                maxfun=10000,
                parameter_plan=selected,
            )
        )
        histories.append(history)
    before, after = outcomes
    assert (
        before[0].common_params["source"].value
        == after[0].common_params["source"].value
    )
    assert before[1:3] == after[1:3]
    for name in before[3]:
        np.testing.assert_array_equal(before[3][name], after[3][name])
    assert histories[0] == histories[1]


def test_estimator_uses_prepared_vectors_but_keeps_legacy_domain_failures(monkeypatch):
    import spectrochempy.analysis.curvefitting.optimize as module

    def line(x, slope):
        return slope * x

    opt = scp.Optimize(
        usermodels={"personal": line},
        autobase=False,
        autoampl=False,
        amplitude_mode="area",
    )
    opt.script = "COMMON:\n$ source: 1.5, 0.1, 4\nMODEL: line\nshape: personal\n> slope: source**2"

    def forbidden(*args, **kwargs):
        pytest.fail("Qualified scalar layouts must use their prepared vectors")

    for name in (
        "_internal_vector",
        "_restore_external",
        "_restore_physical",
        "_transform_artifacts",
    ):
        monkeypatch.setattr(module, name, forbidden)
    x = np.linspace(-1, 1, 31)
    opt.fit(scp.NDDataset(4 * x, coordset=[scp.Coord(x)]))
    assert opt._parameter_plan.fallback and opt._parameter_plan.vector_compatible
    assert opt.fp["source"] == pytest.approx(2.0, abs=1e-4)
    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", 0.0),
            "linked": _ParamSpec("linked", vary=False, reference="source**-1"),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert plan.vector_compatible and plan.fallback
    with pytest.raises(ZeroDivisionError):
        plan.resolve(spec)
