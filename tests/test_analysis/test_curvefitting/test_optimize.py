# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis.curvefitting.optimize import ConstraintError
from spectrochempy.analysis.curvefitting.optimize import ScriptError


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
        assert len(errors) == 1
        assert errors[0].line == 2
        assert "semi-column" in errors[0].message

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

    def test_duplicated_model_name(self):
        opt = scp.Optimize()
        script = (
            "MODEL: X\nshape: gaussianmodel\n"
            "    $ ampl: 1.0, 0.0, none\n"
            "MODEL: X\nshape: lorentzianmodel\n"
            "    $ ampl: 0.5, 0.0, none\n"
        )
        errors = opt.validate_script(script)
        # Duplicate model name is not an error (models are appended),
        # so this should be valid
        assert errors == []

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
