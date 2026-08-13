# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.analysis.curvefitting.optimize import ConstraintError
from spectrochempy.analysis.curvefitting.optimize import ScriptError
from spectrochempy.analysis.curvefitting.optimize import (
    _extract_varying_parameter_values,
)
from spectrochempy.analysis.curvefitting.optimize import _modelspec_parameter_names
from spectrochempy.analysis.curvefitting.optimize import _validate_script_content


# -----------------------------------------------------------------------------------
# validate_script
# -----------------------------------------------------------------------------------
VALID_SCRIPT = """
COMMON:
  $ gratio: 0.1, 0.0, 1.0

MODEL: LINE_1
shape: asymmetricvoigtmodel
    * ampl:  1.0, 0.0, none
    $ pos:   3620, 3400.0, 3700.0
    $ ratio: 0.0147, 0.0, 1.0
    $ asym: 0.1, 0, 1
    $ width: 200, 0, 1000
"""


class TestValidateScript:
    """Tests for Optimize.validate_script()."""

    def test_valid_script_returns_empty_list(self):
        opt = scp.Optimize()
        errors = opt.validate_script(VALID_SCRIPT)
        assert errors == []

    def test_valid_script_can_be_assigned_after_validation(self):
        opt = scp.Optimize()
        errors = opt.validate_script(VALID_SCRIPT)
        assert errors == []
        opt.script = VALID_SCRIPT
        # Assigning a valid script must not raise
        assert opt.fp is not None

    def test_syntax_error_missing_colon(self):
        opt = scp.Optimize()
        script = "MODEL: X\nshape gaussianmodel\n"
        errors = opt.validate_script(script)
        messages = [e.message for e in errors]
        assert any("semi-column" in m for m in messages)
        assert any("no shape" in m for m in messages)

    def test_unknown_model(self):
        opt = scp.Optimize()
        script = "MODEL: X\nshape: unknownmodel\n"
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert "unknownmodel" in errors[0].message
        assert "not found" in errors[0].message

    def test_invalid_parameter_prefix(self):
        opt = scp.Optimize()
        script = "MODEL: X\nshape: gaussianmodel\n% ampl: 1.0, 0.0, none\n"
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert "*,$ or >" in errors[0].message

    def test_missing_model_label(self):
        opt = scp.Optimize()
        script = "$ ampl: 1.0, 0.0, none\n"
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert "first definition" in errors[0].message

    def test_malformed_bounds_too_many_items(self):
        opt = scp.Optimize()
        script = "MODEL: X\nshape: gaussianmodel\n" "$ ampl: 1.0, 0.0, none, extra\n"
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert "min, max" in errors[0].message

    def test_malformed_bounds_two_items(self):
        opt = scp.Optimize()
        script = "MODEL: X\nshape: gaussianmodel\n" "$ ampl: 1.0, 0.0\n"
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert "two" in errors[0].message.lower()

    def test_duplicated_model_name_is_error(self):
        opt = scp.Optimize()
        script = (
            "MODEL: X\nshape: gaussianmodel\n"
            "    $ ampl: 1.0, 0.0, none\n"
            "MODEL: X\nshape: lorentzianmodel\n"
            "    $ ampl: 0.5, 0.0, none\n"
        )
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert "Duplicate model label" in errors[0].message

    def test_empty_script(self):
        opt = scp.Optimize()
        errors = opt.validate_script("")
        assert errors == []

    def test_comment_only_script(self):
        opt = scp.Optimize()
        errors = opt.validate_script("# just a comment\n# another comment\n")
        assert errors == []

    def test_validate_current_script(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_script()
        assert errors == []

    def test_validate_none_uses_current_script(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_script(None)
        assert errors == []

    def test_trait_validator_still_raises_on_invalid(self):
        opt = scp.Optimize()
        with pytest.raises(ValueError, match="semi-column"):
            opt.script = "MODEL: X\nshape gaussianmodel\n"

    def test_script_error_attributes(self):
        err = ScriptError(line=3, text="bad line", message="something wrong")
        assert err.line == 3
        assert err.text == "bad line"
        assert err.message == "something wrong"

    def test_script_error_repr(self):
        err = ScriptError(line=1, text="bad", message="error")
        r = repr(err)
        assert "ScriptError" in r
        assert "line=1" in r

    def test_script_error_str(self):
        err = ScriptError(line=5, text="bad line", message="error msg")
        s = str(err)
        assert "Line 5" in s
        assert "error msg" in s
        assert "bad line" in s

    def test_unknown_model_reported_with_line(self):
        opt = scp.Optimize()
        script = (
            "COMMON:\n"
            "  $ gratio: 0.5, 0.0, 1.0\n"
            "MODEL: PEAK\n"
            "shape: nonexistent_shape\n"
        )
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert errors[0].line == 4
        assert "nonexistent_shape" in errors[0].message

    def test_cannot_evaluate_value(self):
        opt = scp.Optimize()
        script = (
            "MODEL: X\n" "shape: gaussianmodel\n" "  $ ampl: not_a_number, 0.0, none\n"
        )
        errors = opt.validate_script(script)
        assert len(errors) == 1
        assert "Cannot evaluate" in errors[0].message
        assert "not_a_number" in errors[0].message

    def test_validate_after_script_assignment_does_not_alter_fp(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        fp_before = opt.fp
        _ = opt.validate_script()
        # validate_script must not mutate self.fp
        assert opt.fp is fp_before

    def test_validate_script_repeated_calls_are_stable(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        fp_before = opt.fp
        first = opt.validate_script()
        second = opt.validate_script()
        assert first == second == []
        # repeated validation must neither mutate fp nor rebuild the spec
        assert opt.fp is fp_before

    def test_validate_script_invalid_repeated_calls_are_stable(self):
        opt = scp.Optimize()
        script = "MODEL: X\nshape: unknownmodel\n"
        first = opt.validate_script(script)
        second = opt.validate_script(script)
        assert len(first) == len(second) == 1
        assert first[0].message == second[0].message
        assert "unknownmodel" in first[0].message


class TestValidateConstraints:
    """Tests for Optimize.validate_constraints()."""

    def test_empty_constraints_are_valid(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        assert opt.validate_constraints(None) == []
        assert opt.validate_constraints({}) == []
        assert opt.validate_constraints([]) == []

    def test_max_connections_short_form_is_valid(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_constraints({"max_connections": 2})
        assert errors == []

    def test_max_connections_long_form_with_parameters_is_valid(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_constraints(
            {
                "type": "max_connections",
                "limit": 2,
                "parameters": ["pos_line_1", "width_line_1"],
            }
        )
        assert errors == []

    def test_constraints_must_be_mapping_or_sequence_of_mappings(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_constraints("bad")
        assert len(errors) == 1
        assert "dict or a list/tuple" in errors[0].message

    def test_unknown_constraint_type_is_reported(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_constraints(
            {"type": "order", "parameters": ["pos_line_1"]}
        )
        assert len(errors) == 1
        assert "Unsupported constraint type" in errors[0].message

    def test_unknown_parameter_name_is_reported(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_constraints(
            {
                "type": "max_connections",
                "limit": 1,
                "parameters": ["pos_line_1", "missing_parameter"],
            }
        )
        assert len(errors) == 1
        assert "Unknown parameter name" in errors[0].message
        assert "missing_parameter" in errors[0].message

    def test_invalid_limit_is_reported(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        errors = opt.validate_constraints({"max_connections": 0})
        assert len(errors) == 1
        assert "positive integer" in errors[0].message

    def test_invalid_script_blocks_constraint_validation(self):
        opt = scp.Optimize()
        errors = opt.validate_constraints(
            {"max_connections": 1},
            script="MODEL: X\nshape gaussianmodel\n",
        )
        assert len(errors) == 1
        assert "fitting script is invalid" in errors[0].message

    def test_constraint_error_attributes(self):
        err = ConstraintError(index=1, constraint={"max_connections": 2}, message="bad")
        assert err.index == 1
        assert err.constraint == {"max_connections": 2}
        assert err.message == "bad"

    def test_constraint_error_repr_and_str(self):
        err = ConstraintError(index=1, constraint={"max_connections": 2}, message="bad")
        assert "ConstraintError" in repr(err)
        assert "Constraint 1" in str(err)

    def test_constraints_trait_normalizes_short_form(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        opt.constraints = {"max_connections": 2}
        assert opt.constraints == {
            "type": "max_connections",
            "limit": 2,
            "parameters": None,
        }

    def test_constraints_trait_normalizes_sequence(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        opt.constraints = [{"max_connections": 2}]
        assert opt.constraints == [
            {"type": "max_connections", "limit": 2, "parameters": None}
        ]

    def test_constraints_trait_rejects_unknown_parameter(self):
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        with pytest.raises(ValueError, match="Unknown parameter name"):
            opt.constraints = {
                "type": "max_connections",
                "limit": 1,
                "parameters": ["missing_parameter"],
            }

    def test_constraints_trait_validates_against_canonical_spec(self):
        # Parameter-name validation must rely on the canonical _FitModelSpec
        # and keep working even when the legacy Optimize.fp view is absent.
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        opt.fp = None
        with pytest.raises(ValueError, match="Unknown parameter name"):
            opt.constraints = {
                "type": "max_connections",
                "limit": 1,
                "parameters": ["missing_parameter"],
            }

    def test_constraint_parameter_names_match_fitparameters_keys(self):
        # The canonical spec and the legacy FitParameters view expose exactly
        # the same flat parameter names for constraint-name validation.
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        fp_ref, errors = _validate_script_content(VALID_SCRIPT)
        assert errors == []
        assert _modelspec_parameter_names(opt._model_spec) == set(fp_ref.keys())

    def test_fp_matches_historical_parser_output(self):
        # Optimize.fp keeps its public type and reproduces the historical
        # _validate_script_content output for the same script.
        opt = scp.Optimize()
        opt.script = VALID_SCRIPT
        fp_ref, errors = _validate_script_content(VALID_SCRIPT)
        assert errors == []
        fp = opt.fp
        assert isinstance(fp, FitParameters)
        assert fp.models == fp_ref.models
        assert fp.model == fp_ref.model
        assert list(fp.keys()) == list(fp_ref.keys())
        for key in fp_ref.keys():
            assert fp[key] == fp_ref[key]
            assert fp.fixed[key] == fp_ref.fixed[key]
            assert fp.reference[key] == fp_ref.reference[key]
            assert fp.lob.get(key) == fp_ref.lob.get(key)
            assert fp.upb.get(key) == fp_ref.upb.get(key)

    def test_fit_does_not_crash_when_constraints_are_present(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.constraints = {"max_connections": 2}

        result = opt.fit(synthetic_two_peak_dataset)

        assert result is opt


# -----------------------------------------------------------------------------------
# fit behaviour (unchanged)
# -----------------------------------------------------------------------------------
def test_fit_single_dataset(synthetic_two_peak_dataset, optimize_script):
    dataset = synthetic_two_peak_dataset

    f1 = scp.Optimize()
    f1.script = optimize_script
    f1.autobase = True
    f1.max_iter = 10
    result = f1.fit(dataset)

    assert result is f1
    assert f1.n_components == 2
    assert f1.components.shape == (3, dataset.size)
    assert f1.predict().shape == (1, dataset.size)
    assert f1.transform().shape == (1, 2)

    residual = f1.predict().squeeze() - dataset
    assert abs(residual.data).max() < 1e-6
    assert_allclose(
        [
            f1.fp["pos_line_1"],
            f1.fp["pos_line_2"],
            f1.fp["width_line_1"],
            f1.fp["width_line_2"],
        ],
        [3620.0, 3520.0, 200.0, 200.0],
        rtol=0.02,
        atol=3.0,
    )


# -----------------------------------------------------------------------------------
# post-fit fp / script contract (round-trip alignment)
# -----------------------------------------------------------------------------------
def test_fit_preserves_fp_identity_and_type(
    synthetic_two_peak_dataset, optimize_script
):
    # The canonical spec is the source of truth and the public fp view is kept
    # in place after a fit: identity is preserved and the type is unchanged.
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    fp_before = opt.fp
    opt.fit(synthetic_two_peak_dataset)
    assert opt.fp is fp_before
    assert isinstance(opt.fp, FitParameters)


def test_fit_syncs_full_precision_values_into_fp(
    synthetic_two_peak_dataset, optimize_script
):
    # The synced fp view mirrors the canonical spec exactly (full precision);
    # it is not rebuilt from the 4-decimal rendered script.
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    opt.fit(synthetic_two_peak_dataset)
    spec = opt._model_spec
    for key, ps in spec._iter_varying():
        assert opt.fp[key] == ps.value


def test_fit_renders_script_at_display_precision(
    synthetic_two_peak_dataset, optimize_script
):
    # The post-fit script is a rendered public representation of the fitted
    # values: syntactically valid, models, references and the COMMON block
    # survive, and values approximate the internal full-precision state to the
    # historical 4-decimal display precision.
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    opt.fit(synthetic_two_peak_dataset)

    assert "COMMON" in opt.script
    assert "MODEL: line_1" in opt.script
    assert "MODEL: line_2" in opt.script
    assert "> ratio:gratio" in opt.script
    assert opt.validate_script(opt.script) == []

    rendered_fp, errors = _validate_script_content(opt.script)
    assert errors == []
    for key in rendered_fp.keys():
        if isinstance(opt.fp[key], str) or isinstance(rendered_fp[key], str):
            continue
        assert float(rendered_fp[key]) == pytest.approx(float(opt.fp[key]), abs=1e-4)


def test_explicit_script_assignment_after_fit_rebuilds_canonical_state(
    synthetic_two_peak_dataset, optimize_script
):
    # Only an explicit user assignment of a new script is authoritative: it
    # rebuilds the canonical spec and the fp view.
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    opt.fit(synthetic_two_peak_dataset)

    new_script = """
MODEL: X
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  500, 0, 1000
    $ width: 10, 0, 100
"""
    opt.script = new_script
    assert opt.fp["pos_x"] == 500.0
    assert opt._model_spec.components[0].params["pos"].value == 500.0


def test_direct_fp_fixed_mutation_does_not_corrupt_reported_state(
    synthetic_two_peak_dataset, optimize_script
):
    # Direct fp-view mutations are not authoritative: the canonical spec wins
    # and the whole public view is re-synced from it after the fit. The
    # reported diagnostics, the fp view and the rendered script must all stay
    # consistent with what was actually optimized (9 varying parameters), not
    # with the ignored fp.fixed mutation.
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    opt.fp.fixed["pos_line_1"] = True
    opt.fit(synthetic_two_peak_dataset)

    diag = opt.result.diagnostics
    assert diag["n_varying_parameters"] == 9
    assert opt._model_spec.components[0].params["pos"].vary is True
    assert opt.fp.fixed["pos_line_1"] is False
    assert "$ pos" in opt.script
    assert "* pos" not in opt.script


def test_direct_fp_value_mutation_does_not_change_fit(
    synthetic_two_peak_dataset, optimize_script
):
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    opt.fp["pos_line_1"] = (3500.0, 3000.0, 4000.0)  # ignored by the canonical spec
    opt.fit(synthetic_two_peak_dataset)

    assert opt.result.diagnostics["n_varying_parameters"] == 9
    assert opt._model_spec.components[0].params["pos"].value == pytest.approx(
        3620.0, abs=3.0
    )
    assert opt.fp["pos_line_1"] == pytest.approx(3620.0, abs=3.0)
    assert opt.fp.lob["pos_line_1"] == pytest.approx(3400.0)
    assert opt.fp.upb["pos_line_1"] == pytest.approx(3700.0)


def test_direct_fp_bounds_mutation_does_not_survive_in_view(
    synthetic_two_peak_dataset, optimize_script
):
    # Direct bound mutations in the fp view are not authoritative: the bounds
    # are restored from the canonical spec after the fit.
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    opt.fp.lob["pos_line_1"] = 3000.0
    opt.fp.upb["pos_line_1"] = 4000.0
    opt.fit(synthetic_two_peak_dataset)

    assert opt.fp.lob["pos_line_1"] == pytest.approx(3400.0)
    assert opt.fp.upb["pos_line_1"] == pytest.approx(3700.0)
    assert opt._model_spec.components[0].params["pos"].bounds == (3400.0, 3700.0)


def test_direct_fp_reference_mutation_does_not_survive_in_view(
    synthetic_two_peak_dataset, optimize_script
):
    # Reference flags are part of the canonical structure: replacing the
    # ``ratio_line_2`` reference by an inline value in the fp view must not
    # survive the fit.
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    assert opt.fp.reference["ratio_line_2"] is True
    opt.fp.reference["ratio_line_2"] = False
    opt.fp.data["ratio_line_2"] = 0.25
    opt.fp.fixed["ratio_line_2"] = False
    opt.fp.lob["ratio_line_2"] = 0.0
    opt.fp.upb["ratio_line_2"] = 1.0
    opt.fit(synthetic_two_peak_dataset)

    assert opt.fp.reference["ratio_line_2"] is True
    assert opt.fp.data["ratio_line_2"] == "gratio"
    assert opt.fp.fixed["ratio_line_2"] is True
    assert "> ratio:gratio" in opt.script


def test_postfit_script_assignment_notifies_observers(
    synthetic_two_peak_dataset, optimize_script
):
    # The post-fit script render is an observable change of the ``script``
    # trait: observers are notified as for a normal assignment, even though
    # the rendered text is never re-parsed into canonical state.
    opt = scp.Optimize()
    opt.script = optimize_script
    changes = []
    opt.observe(
        lambda c: changes.append((c["name"], c["old"], c["new"])), names="script"
    )
    opt.autobase = True
    opt.max_iter = 10
    opt.fit(synthetic_two_peak_dataset)

    assert len(changes) == 1
    name, old, new = changes[0]
    assert name == "script"
    assert old == optimize_script
    assert new == opt.script
    assert "MODEL: line_1" in new
    assert "MODEL: line_2" in new


def test_fp_only_entry_fits_and_preserves_identity(
    synthetic_two_peak_dataset, optimize_script
):
    # The fp-only legacy entry path remains live: a FitParameters view can be
    # fitted directly, the same public object is kept, and the post-fit script
    # is rendered.
    ref = scp.Optimize()
    ref.script = optimize_script
    ref.autobase = True
    ref.max_iter = 10
    ref.fit(synthetic_two_peak_dataset)

    opt = scp.Optimize()
    fp = ref.fp.copy()
    opt.fp = fp
    opt.autobase = True
    opt.max_iter = 10
    opt.fit(synthetic_two_peak_dataset)

    assert opt.fp is fp
    assert isinstance(opt.fp, FitParameters)
    assert opt.result.diagnostics["n_varying_parameters"] == 9
    assert opt.fp["pos_line_1"] == pytest.approx(3620.0, abs=3.0)
    assert "MODEL" in opt.script
    assert_allclose(
        opt._model_spec.extract_varying_values(),
        _extract_varying_parameter_values(opt.fp),
    )


def test_successive_fits_are_stable(synthetic_two_peak_dataset, optimize_script):
    opt = scp.Optimize()
    opt.script = optimize_script
    opt.autobase = True
    opt.max_iter = 10
    opt.fit(synthetic_two_peak_dataset)
    first_cost = opt.result.diagnostics["cost"]
    first_pos = opt.fp["pos_line_1"]
    opt.fit(synthetic_two_peak_dataset)
    assert opt.result.diagnostics["cost"] == pytest.approx(first_cost, rel=1e-6)
    assert opt.fp["pos_line_1"] == pytest.approx(first_pos, rel=1e-6)
