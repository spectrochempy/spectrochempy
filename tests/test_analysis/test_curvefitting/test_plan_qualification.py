"""Inventory conservative plan qualification without changing execution routes."""

import pickle
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model
from spectrochempy.analysis.curvefitting._parameter_plan import _compile_parameter_plan


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("source+1", None),
        ("(source)**2", None),
        ("source**2", "signed_literal_precedence"),
        ("source % 2", "unsupported_expression_syntax"),
        ("source // 2", "unsupported_expression_syntax"),
        ("1 if source else 2", "unsupported_expression_syntax"),
        ("sin(source)", "unsupported_expression_syntax"),
        ("np.pi", "unsupported_expression_syntax"),
        ("source.real", "unsupported_expression_syntax"),
        ("source_peak", "legacy_name_tokens"),
        ("missing", "unknown_expression_names"),
        ("a+a1", "overlapping_expression_names"),
        ("1.e3", "legacy_text_tokens"),
        ("2 # comment", "legacy_text_tokens"),
    ],
)
def test_expression_inventory_records_first_gate_without_evaluating(expression, reason):
    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", -2.0),
            "a": _ParamSpec("a", 1.0),
            "a1": _ParamSpec("a1", 2.0),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
        }
    )
    plan = _compile_parameter_plan(spec)
    assert plan.fallback_reason == reason
    assert plan.fallback is (reason is not None)
    assert plan.vector_compatible
    assert spec.common_params["linked"].reference == expression


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("expvars", "experiment_layout"),
        ("expnumber", "experiment_layout"),
        ("constraints", "reserved_constraints"),
        ("keys", "duplicate_parameter_keys"),
        ("labels", "duplicate_component_labels"),
        ("value", "non_scalar_value"),
        ("bound", "non_scalar_bound"),
    ],
)
def test_structural_inventory_remains_outside_prepared_vectors(kind, reason):
    spec = _FitModelSpec(common_params={"source": _ParamSpec("source", 2.0)})
    if kind == "expvars":
        spec.expvars = ["source"]
    elif kind == "expnumber":
        spec.expnumber = 2
    elif kind == "constraints":
        spec.constraints = {}
    elif kind == "keys":
        spec.common_params["width_peak"] = _ParamSpec("width_peak", 2.0)
        spec.components = [
            _ComponentSpec("peak", params={"width": _ParamSpec("width", 2.0)})
        ]
    elif kind == "labels":
        spec.components = [_ComponentSpec("peak"), _ComponentSpec("peak")]
    elif kind == "value":
        spec.common_params["source"].value = [2.0]
    else:
        spec.common_params["source"].bounds = ([0.0], 3.0)
    plan = _compile_parameter_plan(spec)
    assert plan.fallback_reason == reason
    assert plan.fallback and not plan.vector_compatible


def test_first_gate_is_deterministic_and_cycles_still_raise():
    spec = _FitModelSpec(
        common_params={"linked": _ParamSpec("linked", vary=False, reference="missing")},
        constraints={},
        expnumber=2,
    )
    assert _compile_parameter_plan(spec).fallback_reason == "experiment_layout"
    spec.expnumber = 1
    assert _compile_parameter_plan(spec).fallback_reason == "reserved_constraints"
    spec.constraints = None
    assert _compile_parameter_plan(spec).fallback_reason == "unknown_expression_names"
    spec.common_params["linked"].reference = "other"
    spec.common_params["other"] = _ParamSpec("other", vary=False, reference="linked")
    with pytest.raises(ValueError, match="Cyclic parameter expression"):
        _compile_parameter_plan(spec)


def test_expression_expansion_is_identified_before_evaluation():
    declarations = {
        f"v{i}": _ParamSpec(f"v{i}", vary=False, reference=f"v{i-1}+v{i-1}")
        for i in range(18, 0, -1)
    }
    declarations["v0"] = _ParamSpec("v0", 2.0)
    plan = _compile_parameter_plan(_FitModelSpec(common_params=declarations))
    assert plan.fallback_reason == "expression_expansion_limit"
    assert plan.fallback and plan.vector_compatible


@pytest.mark.parametrize("expression", ["source**2", "(source)**2", "1.e3"])
def test_reason_survives_copy_and_pickle_but_does_not_control_resolution(expression):
    spec = _FitModelSpec(
        common_params={
            "source": _ParamSpec("source", -2.0),
            "linked": _ParamSpec("linked", vary=False, reference=expression),
        }
    )
    plan = _compile_parameter_plan(spec)
    for restored in (deepcopy(plan), pickle.loads(pickle.dumps(plan))):  # noqa: S301
        assert restored == plan
        for value in (-3.0, -0.0, 0.0, 2.0):
            trial = deepcopy(spec)
            trial.common_params["source"].value = value
            expected = prepare_model(trial).common_params["linked"].value
            for selected in (
                restored,
                replace(restored, fallback_reason="observational_only"),
            ):
                actual = selected.resolve(trial).common_params["linked"].value
                assert actual == expected
                assert np.signbit(actual) == np.signbit(expected)
    assert spec.common_params["source"].value == -2


@pytest.mark.parametrize(
    ("expression", "reason"),
    [("source**2", "signed_literal_precedence"), ("(source)**2", None)],
)
def test_qualification_does_not_add_a_public_result_surface(expression, reason):
    def line(x, slope):
        return slope * x

    opt = scp.Optimize(
        usermodels={"personal": line},
        autobase=False,
        autoampl=False,
        amplitude_mode="area",
    )
    opt.script = (
        "COMMON:\n$ source: 1.5, 0.1, 4\nMODEL: line\nshape: personal\n> slope: "
        + expression
    )
    x = np.linspace(-1, 1, 41)[::-1]
    mask = np.zeros(x.size, dtype=bool)
    mask[5:8] = True
    data = scp.NDDataset(4 * x, coordset=[scp.Coord(x)], mask=mask)
    opt.fit(data)
    assert opt._parameter_plan.fallback_reason == reason
    assert opt.fp["source"] == pytest.approx(2, abs=1e-4)
    assert "fallback_reason" not in opt.result.diagnostics
    assert "fallback_reason" not in opt.result.parameters
    np.testing.assert_array_equal(opt.result.fitted.mask.squeeze(), mask)
    np.testing.assert_allclose(
        opt.result.fitted.data.squeeze()[~mask], data.data[~mask], atol=1e-4
    )
