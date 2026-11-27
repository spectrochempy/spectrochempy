"""Qualify existing power expressions without changing legacy precedence."""

import pickle
from copy import deepcopy

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model
from spectrochempy.analysis.curvefitting._parameter_plan import _compile_parameter_plan


@pytest.mark.parametrize(
    ("expression", "fallback"),
    [
        ("(source)**2", False),
        ("source**2", True),
        ("-source**2", True),
        ("(-source)**2", False),
        ("(source+1)**2", False),
        ("2**source", False),
        ("2**source**2", True),
        ("2**(source)**2", False),
        ("(source)**(source)", False),
        ("(source)**-2", False),
        ("2**3**2", False),
        ("(-2)**0.5", False),
        ("(source)**0.5", False),
        ("(source)**10000", False),
        ("1.e3**2", True),
        ("(source)**2 # comment", True),
    ],
)
def test_power_values_domains_and_sign_changes_match_legacy(expression, fallback):
    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", 2.0),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert plan.fallback is fallback
    for source in (-2.0, -0.0, 0.0, 0.5, 2.0):
        spec.common_params["source"].value = source
        try:
            expected = prepare_model(spec).common_params["linked"].value
        except (TypeError, ZeroDivisionError, OverflowError) as error:
            with pytest.raises(type(error)):
                plan.resolve(spec)
        else:
            actual = plan.resolve(spec).common_params["linked"].value
            assert actual == expected
            assert np.signbit(actual) == np.signbit(expected)
    assert spec.common_params["linked"].reference == expression


@pytest.mark.parametrize("reverse", [False, True])
def test_power_chains_keep_textual_expansion_copy_and_pickle(reverse):
    entries = [
        ("linked", _ParamSpec("linked", vary=False, reference="(twice)**2")),
        ("twice", _ParamSpec("twice", vary=False, reference="source + 1")),
        ("source", _ParamSpec("source", -2.0)),
    ]
    spec = _FitModelSpec(common_params=dict(entries[::-1] if reverse else entries))
    plan = _compile_parameter_plan(spec)
    assert not plan.fallback
    for restored in (plan, deepcopy(plan), pickle.loads(pickle.dumps(plan))):  # noqa: S301
        trial = deepcopy(spec)
        for value in (-3.0, -0.0, 0.0, 2.0):
            trial.common_params["source"].value = value
            actual = restored.resolve(trial)
            expected = prepare_model(trial)
            assert {k: p.value for k, p in actual.common_params.items()} == {
                k: p.value for k, p in expected.common_params.items()
            }
    assert spec.common_params["source"].value == -2


def test_compilation_occurs_once_not_during_resolution(monkeypatch):
    import spectrochempy.analysis.curvefitting._parameter_plan as module

    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", 2.0),
            "linked": _ParamSpec("linked", vary=False, reference="(source)**2"),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert not plan.fallback

    def forbidden(*args, **kwargs):
        pytest.fail("A qualified expression must be compiled only during preparation")

    monkeypatch.setattr(module, "compile", forbidden, raising=False)
    monkeypatch.setattr(module, "prepare_model", forbidden)
    for value in (-3.0, 0.0, 2.0):
        spec.common_params["source"].value = value
        assert plan.resolve(spec).common_params["linked"].value == value**2


def test_common_and_python_power_fits_have_same_results_and_initial_configs():
    x = np.linspace(-1, 1, 41)
    mask = np.zeros(x.size, dtype=bool)
    mask[5:8] = True
    data = scp.NDDataset(
        4 * x + 1 + 0.002 * np.cos(7 * x), coordset=[scp.Coord(x)], mask=mask
    )
    model = scp.Linear()
    params = model.make_params(intercept=1)
    params.add("source", value=1.5, min=0.1, max=4)
    params["slope"].set(expr="(source)**2")
    python = scp.Optimize(model=model, params=params).fit(data)

    def line(x, slope, intercept):
        return slope * x + intercept

    script = """
COMMON:
$ source: 1.5, 0.1, 4
MODEL: line
shape: personal
> slope: (source)**2
$ intercept: 1, none, none
"""
    common = scp.Optimize(
        usermodels={"personal": line},
        autobase=False,
        autoampl=False,
        amplitude_mode="area",
    )
    common.script = script
    common.fit(data)
    assert not common._parameter_plan.fallback
    assert not python._parameter_plan.fallback
    np.testing.assert_allclose(
        python.result.fitted.data, common.result.fitted.data, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        python.result.residuals.data, common.result.residuals.data, rtol=1e-8, atol=1e-9
    )
    assert params["source"].value == 1.5
    assert params["slope"].expr == "(source)**2"
    second = scp.Optimize(model=model, params=params).fit(data)
    np.testing.assert_array_equal(python.result.fitted.data, second.result.fitted.data)
    assert python.result.parameter_names == second.result.parameter_names
    assert params["source"].value == 1.5


@pytest.mark.parametrize("terms", [250, 300, 500])
def test_deep_power_expressions_keep_legacy_evaluation(terms):
    expression = "+".join(["(source)**2"] * terms)
    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", 2.0),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
        }
    )
    plan = _compile_parameter_plan(spec)
    # No fixed AST-depth threshold: a platform may safely qualify the expression.
    # If it cannot, preparing the optional plan must still preserve legacy fits.
    for source in (2.0, -2.0, 0.5):
        spec.common_params["source"].value = source
        actual = plan.resolve(spec).common_params["linked"].value
        expected = prepare_model(spec).common_params["linked"].value
        assert actual == expected == source**2 * terms
    assert spec.common_params["linked"].reference == expression
