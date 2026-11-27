"""Qualification of fit-local compilation against the legacy resolver."""

import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model
from spectrochempy.analysis.curvefitting._parameter_plan import _compile_parameter_plan


def values(spec):
    return {
        **{name: p.value for name, p in spec.common_params.items()},
        **{
            f"{name}_{c.label}": p.value
            for c in spec.components
            for name, p in c.params.items()
        },
    }


def definition(common):
    return _FitModelSpec(
        common_params=common,
        components=[
            _ComponentSpec(
                "peak",
                "gaussianmodel",
                {
                    "width": _ParamSpec("width", vary=False, reference="linked"),
                    "ampl": _ParamSpec("ampl", 1.0, bounds=(0, 5)),
                },
            )
        ],
    )


@pytest.mark.parametrize("source", [-2.0, 0.0, 3.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_forward_backward_expansion_matches_legacy(source, reverse):
    declarations = [
        ("linked", _ParamSpec("linked", vary=False, reference="twice * 3")),
        ("twice", _ParamSpec("twice", vary=False, reference="source + 1")),
        ("source", _ParamSpec("source", source)),
    ]
    spec = definition(dict(declarations[::-1] if reverse else declarations))
    plan = _compile_parameter_plan(spec)
    assert not plan.fallback
    assert values(plan.resolve(spec)) == values(prepare_model(spec))
    assert spec.components[0].params["width"].reference == "linked"
    assert plan.free_names == ("ampl_peak", "source")
    snapshot = plan.parameters
    spec.common_params["source"].value = 4
    assert values(plan.resolve(spec)) == values(prepare_model(spec))
    assert plan.parameters == snapshot
    assert next(p for p in snapshot if p.name == "source").value == source
    with pytest.raises(FrozenInstanceError):
        plan.fallback = True


@pytest.mark.parametrize(
    "expression", ["-source/2", "(source+1)*2", "1e-3 * source", "1+1"]
)
def test_numeric_arithmetic_and_signs(expression):
    spec = definition(
        {
            "source": _ParamSpec("source", -2),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert not plan.fallback
    assert values(plan.resolve(spec)) == values(prepare_model(spec))


@pytest.mark.parametrize(
    "expression", ["source**2", "source % 2", "1 if source else 2"]
)
def test_unqualified_expressions_keep_legacy_path(expression):
    spec = definition(
        {
            "source": _ParamSpec("source", -2),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert plan.fallback
    assert values(plan.resolve(spec)) == values(prepare_model(spec))


def test_unknown_name_keeps_legacy_error_and_cycle_fails_before_evaluation():
    spec = definition({"linked": _ParamSpec("linked", vary=False, reference="missing")})
    plan = _compile_parameter_plan(spec)
    assert plan.fallback
    with pytest.raises(NameError):
        plan.resolve(spec)
    spec.common_params["linked"].reference = "other"
    spec.common_params["other"] = _ParamSpec("other", vary=False, reference="linked")
    with pytest.raises(ValueError, match="Cyclic parameter expression"):
        _compile_parameter_plan(spec)


def test_resolution_results_and_physical_trials_are_independent():
    spec = definition(
        {
            "source": _ParamSpec("source", 2),
            "linked": _ParamSpec("linked", vary=False, reference="2 * source"),
        }
    )
    plan = _compile_parameter_plan(spec)
    first = plan.resolve(spec)
    trial = deepcopy(spec)
    trial.common_params["source"].value = 3
    second = plan.resolve(trial)
    assert first.common_params["linked"].value == 4
    assert second.common_params["linked"].value == 6
    first.common_params["source"].value = 99
    first.components[0].params["width"].value = 99
    assert values(plan.resolve(spec)) == values(prepare_model(spec))


def test_plan_is_rebuilt_per_fit_and_script_assignment_invalidates_it(monkeypatch):
    import spectrochempy.analysis.curvefitting.optimize as module

    original = module._compile_parameter_plan
    compiled = []

    def record(spec):
        result = original(spec)
        compiled.append(result)
        return result

    monkeypatch.setattr(module, "_compile_parameter_plan", record)
    opt = scp.Optimize(
        script="MODEL: line\nshape: polynomialbaseline\n$ ampl: 1, 0, 5",
        autobase=False,
        autoampl=False,
        amplitude_mode="area",
    )
    x = np.linspace(-1, 1, 51)
    data = scp.NDDataset(np.full(x.size, 2.0), coordset=[scp.Coord(x)])
    opt.fit(data)
    first = opt._parameter_plan
    assert len(compiled) == 1
    opt.fit(data)
    assert len(compiled) == 2
    assert opt._parameter_plan is not first
    opt.script = "MODEL: line\nshape: polynomialbaseline\n$ ampl: 1.5, 0, 5"
    assert opt._parameter_plan is None


def test_runtime_domain_failure_is_preserved():
    spec = definition(
        {
            "source": _ParamSpec("source", 2),
            "linked": _ParamSpec("linked", vary=False, reference="1/source"),
        }
    )
    plan = _compile_parameter_plan(spec)
    spec.common_params["source"].value = 0
    for resolve in (plan.resolve, prepare_model):
        with pytest.raises(ZeroDivisionError):
            resolve(spec)


def test_compiled_plan_remains_copyable_and_serializable():
    spec = definition({"linked": _ParamSpec("linked", vary=False, reference="2 + 3")})
    plan = _compile_parameter_plan(spec)
    for restored in (deepcopy(plan), pickle.loads(pickle.dumps(plan))):  # noqa: S301
        assert restored == plan
        assert values(restored.resolve(spec)) == values(prepare_model(spec))


@pytest.mark.parametrize(
    ("expression", "expected", "fallback"),
    [
        ("1.e3", 1000.0, True),
        ("2 # comment", 2.0, True),
        ("1.E-3", 0.001, False),
        ("1.0e3", 1000.0, False),
        ("0xdead", 57005.0, False),
    ],
)
def test_legacy_numeric_tokens_and_comments(expression, expected, fallback):
    spec = definition(
        {"linked": _ParamSpec("linked", vary=False, reference=expression)}
    )
    plan = _compile_parameter_plan(spec)
    assert plan.fallback is fallback
    assert values(plan.resolve(spec)) == values(prepare_model(spec))
    assert plan.resolve(spec).common_params["linked"].value == expected


def test_script_numeric_literal_legacy_fallback():
    x = np.linspace(-1, 1, 51)
    data = scp.NDDataset(
        scp.gaussian(x, ampl=1000, pos=0, width=1), coordset=[scp.Coord(x)]
    )
    opt = scp.Optimize(
        script="MODEL: peak\nshape: gaussianmodel\n> ampl: 1.e3\n* pos: 0\n* width: 1",
        autobase=False,
        autoampl=False,
        amplitude_mode="area",
        dry=True,
    ).fit(data)
    resolved = opt._resolve_model(opt._model_spec)
    assert resolved.components[0].params["ampl"].value == 1000.0
    np.testing.assert_allclose(opt.predict().data.squeeze(), data.data, rtol=1e-14)


def test_unlinked_magnitudes_preserve_their_original_precision():
    source = np.longdouble("0.1234567890123456789")
    spec = definition(
        {
            "source": _ParamSpec("source", source),
            "linked": _ParamSpec("linked", vary=False, reference="source * 2"),
        }
    )
    plan = _compile_parameter_plan(spec)
    actual = plan.resolve(spec)
    assert actual.common_params["source"].value == source
    assert type(actual.common_params["source"].value) is type(source)
    assert values(actual) == values(prepare_model(spec))
