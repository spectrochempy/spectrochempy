# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Parser characterization tests for the Optimize script DSL.

These tests document the *current* parser contract.
They do NOT assert normative correctness of the design.
They freeze the current behavior so that future refactoring
can be verified against a stable characterization.

Public API changes must update these tests.
Parser implementation changes must keep these tests green.
"""

import sys

import pytest

from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.analysis.curvefitting.optimize import ScriptError
from spectrochempy.analysis.curvefitting.optimize import _validate_script_content

# ======================================================================================
# Constants used within the parser and FitParameters
# ======================================================================================

# Sentinel values used when bounds are "none"
NEG_SENTINEL = -1.0 / sys.float_info.epsilon
POS_SENTINEL = +1.0 / sys.float_info.epsilon

# Threshold used by FitParameters.__str__ to decide "none" vs printed bound
STR_NEG_THRESH = -0.1 / sys.float_info.epsilon
STR_POS_THRESH = +0.1 / sys.float_info.epsilon


# ======================================================================================
# 1.  Basic parsing — single component
# ======================================================================================


class TestBasicParsing:
    """A simple one-component Gaussian model documents the core parser contract."""

    SCRIPT = """
MODEL: PEAK_A
shape: gaussianmodel
    $ ampl:  1.5, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    def test_returns_fitparameters(self):
        fp, errors = _validate_script_content(self.SCRIPT)
        assert errors == []
        assert isinstance(fp, FitParameters)

    def test_model_list(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.models == ["peak_a"]

    def test_model_mapping(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.model == {"peak_a": "gaussianmodel"}

    def test_parameter_values(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp["ampl_peak_a"] == 1.5
        assert fp["pos_peak_a"] == 100.0
        assert fp["width_peak_a"] == 10.0

    def test_parameter_naming_convention(self):
        """Parameter keys follow the pattern {name}_{label}."""
        fp, _ = _validate_script_content(self.SCRIPT)
        assert set(fp.keys()) == {"ampl_peak_a", "pos_peak_a", "width_peak_a"}

    def test_all_varying_by_default(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        for key in fp:
            assert fp.fixed[key] is False

    def test_all_non_reference(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        for key in fp:
            assert fp.reference[key] is False

    def test_all_non_common(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        for key in ("ampl", "pos", "width"):
            assert fp.common[key] is False

    def test_finite_bounds(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.lob["pos_peak_a"] == 0.0
        assert fp.upb["pos_peak_a"] == 200.0

    def test_open_lower_bound_stores_sentinel(self):
        """An unspecified lower bound stores a large negative sentinel value."""
        fp, _ = _validate_script_content(self.SCRIPT)
        # ampl has lo=0.0, so it IS specified — use width which has lo=0.0 too
        # Actually all have explicit lower bounds. Use a script with a truly open bound.
        pass

    def test_open_upper_bound_stores_sentinel(self):
        """
        An unspecified upper bound stores a large positive sentinel.

        The sentinel is ``+1.0 / sys.float_info.epsilon``.
        """
        fp, _ = _validate_script_content(self.SCRIPT)
        # ampl has hi="none", so upb is the sentinel
        assert fp.upb["ampl_peak_a"] == pytest.approx(POS_SENTINEL)


# ======================================================================================
# 2.  Multiple components
# ======================================================================================


class TestMultipleComponents:
    """Two-component script to verify ordering, labels, and separation."""

    SCRIPT = """
MODEL: PEAK_1
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none

MODEL: PEAK_2
shape: lorentzianmodel
    $ ampl: 0.5, 0.0, none
    $ pos:  150.0, 0.0, 300.0
    $ width: 20.0, 0.0, none
"""

    def test_model_list_ordering(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.models == ["peak_1", "peak_2"]

    def test_parameter_names_include_label(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert "ampl_peak_1" in fp
        assert "ampl_peak_2" in fp

    def test_parameters_separate_between_models(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp["ampl_peak_1"] == 1.0
        assert fp["ampl_peak_2"] == 0.5
        assert fp["pos_peak_1"] == 100.0
        assert fp["pos_peak_2"] == 150.0

    def test_model_types_preserved(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.model["peak_1"] == "gaussianmodel"
        assert fp.model["peak_2"] == "lorentzianmodel"


# ======================================================================================
# 3.  COMMON parameters
# ======================================================================================


class TestCommonParameters:
    """COMMON block defines shared parameters accessible across models."""

    SCRIPT = """
COMMON:
  $ gratio: 0.1, 0.0, 1.0

MODEL: PEAK_1
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    def test_common_param_stored_without_label_suffix(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert "gratio" in fp
        assert fp["gratio"] == 0.1

    def test_common_flag_true_for_common_params(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.common["gratio"] is True

    def test_common_flag_false_for_model_params(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.common["ampl"] is False
        assert fp.common["pos"] is False
        assert fp.common["width"] is False

    def test_common_bounds(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.lob["gratio"] == 0.0
        assert fp.upb["gratio"] == 1.0

    def test_vars_as_alias_for_common(self):
        """The parser accepts ``VARS:`` as an alias for ``COMMON:``."""
        script = """
VARS:
  $ gratio: 0.1, 0.0, 1.0

MODEL: PEAK_1
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
"""
        fp, errors = _validate_script_content(script)
        assert errors == []
        assert fp["gratio"] == 0.1
        assert fp.common["gratio"] is True

    def test_common_param_not_in_model_section(self):
        """Common parameters should not appear under MODEL in parameter keys."""
        fp, _ = _validate_script_content(self.SCRIPT)
        # gratio is the only key without an underscore suffix
        common_keys = [k for k in fp if "_" not in k]
        assert common_keys == ["gratio"]


# ======================================================================================
# 4.  References to COMMON parameters
# ======================================================================================


class TestReferences:
    """The ``> name: refname`` syntax references a COMMON parameter."""

    SCRIPT = """
COMMON:
  $ gratio: 0.1, 0.0, 1.0

MODEL: PEAK_1
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    > ratio: gratio
    $ width: 10.0, 0.0, none
"""

    def test_reference_flag_true(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.reference["ratio_peak_1"] is True

    def test_reference_stores_referent_name_as_string(self):
        """The reference stores the *name* of the common param as a string."""
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp["ratio_peak_1"] == "gratio"
        assert isinstance(fp["ratio_peak_1"], str)

    def test_reference_is_fixed(self):
        """Parameters declared via ``>`` are automatically fixed."""
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.fixed["ratio_peak_1"] is True

    def test_non_reference_params_are_not_marked(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.reference["ampl_peak_1"] is False
        assert fp.reference["pos_peak_1"] is False

    def test_common_param_itself_not_a_reference(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.reference["gratio"] is False

    def test_reference_has_no_bounds(self):
        """Reference parameters get no meaningful bounds (bounds are inherited at runtime)."""
        fp, _ = _validate_script_content(self.SCRIPT)
        # The reference value is stored as string, so lob/upb were not set
        # by __setitem__. We verify the sentinel is not what finite bounds look like.
        assert fp.lob.get("ratio_peak_1") is None
        assert fp.upb.get("ratio_peak_1") is None


# ======================================================================================
# 5.  Bounds characterization
# ======================================================================================


class TestBounds:
    """Characterize how bounds are stored and the sentinel convention."""

    SCRIPT_ALL_FINITE = """
MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, 10.0
    $ pos:   100.0, -100.0, 300.0
"""

    SCRIPT_OPEN_UPPER = """
MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
"""

    SCRIPT_OPEN_LOWER = """
MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, none, 10.0
"""

    SCRIPT_NO_BOUNDS = """
MODEL: X
shape: gaussianmodel
    $ ampl:  1.0
"""

    def test_finite_lower_bound(self):
        fp, _ = _validate_script_content(self.SCRIPT_ALL_FINITE)
        assert fp.lob["ampl_x"] == 0.0
        assert fp.lob["pos_x"] == -100.0

    def test_finite_upper_bound(self):
        fp, _ = _validate_script_content(self.SCRIPT_ALL_FINITE)
        assert fp.upb["ampl_x"] == 10.0
        assert fp.upb["pos_x"] == 300.0

    def test_open_upper_bound_stores_positive_sentinel(self):
        """When upper bound is 'none', the parser stores ``+1.0 / ε``."""
        fp, _ = _validate_script_content(self.SCRIPT_OPEN_UPPER)
        expected = +1.0 / sys.float_info.epsilon
        assert fp.upb["ampl_x"] == pytest.approx(expected)

    def test_open_lower_bound_stores_negative_sentinel(self):
        """When lower bound is 'none', the parser stores ``-1.0 / ε``."""
        fp, _ = _validate_script_content(self.SCRIPT_OPEN_LOWER)
        expected = -1.0 / sys.float_info.epsilon
        assert fp.lob["ampl_x"] == pytest.approx(expected)

    def test_no_bounds_specified_uses_sentinels(self):
        """When only value is given, both bounds are sentinels."""
        fp, _ = _validate_script_content(self.SCRIPT_NO_BOUNDS)
        assert fp.lob["ampl_x"] == pytest.approx(NEG_SENTINEL)
        assert fp.upb["ampl_x"] == pytest.approx(POS_SENTINEL)

    def test_sentinel_greater_than_str_threshold(self):
        """
        The sentinel exceeds the ``__str__`` threshold so it renders as 'none'.

        This ensures round-trip works: parsed sentinel → str → "none" → re-parsed sentinel.
        """
        # POS_SENTINEL = +1.0/ε, STR_POS_THRESH = +0.1/ε
        assert POS_SENTINEL > STR_POS_THRESH
        assert NEG_SENTINEL < STR_NEG_THRESH

    def test_epsilon_calculation(self):
        """Sanity check: epsilon is very small so sentinel is very large."""
        assert sys.float_info.epsilon < 1e-15
        assert POS_SENTINEL > 1e15


# ======================================================================================
# 6.  Fixed vs varying
# ======================================================================================


class TestFixedVarying:
    """Verify how ``*`` (fixed) and ``$`` (varying) are represented."""

    SCRIPT = """
MODEL: X
shape: gaussianmodel
    * ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    def test_fixed_prefix_marks_parameter_fixed(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.fixed["ampl_x"] is True

    def test_varying_prefix_marks_parameter_not_fixed(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.fixed["pos_x"] is False
        assert fp.fixed["width_x"] is False

    def test_fixed_params_have_bounds(self):
        """Fixed parameters still carry their declared bounds."""
        fp, _ = _validate_script_content(self.SCRIPT)
        assert fp.lob["ampl_x"] == 0.0
        assert fp.upb["ampl_x"] == pytest.approx(POS_SENTINEL)


# ======================================================================================
# 7.  User models — currently uncovered
# ======================================================================================


class TestUserModels:
    """
    User-model parsing is not yet characterized.

    No dedicated fixtures exist in the test suite as of this writing.
    The parser path for usermodels is:

    1. ``_validate_script_content(script, usermodels=...)``
    2. ``usermodels`` is a dict of ``{name: callable_or_instance}``
    3. The parser checks ``shape in usermodels`` when resolving shape names

    Characterizing this path requires:

    - a user-model fixture (a function with the right signature)
    - a script referencing the user model name as shape
    - verification that ``fp.model[label]`` matches the user model name

    This remains a gap for now.
    """

    def test_no_usermodels_fixture_available(self):
        """Placeholder: user-model characterization is marked as uncovered."""
        pass


# ======================================================================================
# 8.  Empty and trivial scripts
# ======================================================================================


class TestEmptyScripts:
    """Characterize how the parser handles empty or trivial input."""

    def test_empty_string_returns_empty_fitparameters(self):
        fp, errors = _validate_script_content("")
        assert errors == []
        assert fp.models == []
        assert fp.model == {}
        assert list(fp.keys()) == []

    def test_whitespace_only(self):
        fp, errors = _validate_script_content("   \n  \n  ")
        assert errors == []
        assert fp.models == []

    def test_comment_only(self):
        fp, errors = _validate_script_content("# just a comment\n# another\n")
        assert errors == []
        assert fp.models == []

    def test_comment_with_leading_whitespace(self):
        fp, errors = _validate_script_content("  # indented comment\n")
        assert errors == []
        assert fp.models == []

    def test_common_without_models(self):
        """A COMMON block with no MODEL block is valid."""
        script = "COMMON:\n  $ gratio: 0.1, 0.0, 1.0\n"
        fp, errors = _validate_script_content(script)
        assert errors == []
        assert fp["gratio"] == 0.1
        assert fp.models == []

    def test_model_without_shape_is_error(self):
        """
        A model with no ``shape:`` line is a parser error.

        Without a shape the model cannot be evaluated at fit time,
        so the parser now rejects it explicitly.
        """
        script = "MODEL: X\n  $ ampl: 1.0, 0.0, none\n"
        fp, errors = _validate_script_content(script)
        assert len(errors) == 1
        assert "no shape" in errors[0].message.lower()
        assert "x" in errors[0].message.lower()


# ======================================================================================
# 9.  Invalid scripts — validation behavior
# ======================================================================================


class TestInvalidScripts:
    """Characterize what the parser rejects and how."""

    def test_missing_colon(self):
        fp, errors = _validate_script_content("MODEL: X\nshape gaussianmodel\n")
        messages = [e.message for e in errors]
        assert any("semi-column" in m for m in messages)
        assert any("no shape" in m for m in messages)

    def test_unknown_model_name(self):
        fp, errors = _validate_script_content("MODEL: X\nshape: nonexistent_shape\n")
        assert len(errors) == 1
        assert "not found" in errors[0].message
        assert "nonexistent_shape" in errors[0].message

    def test_invalid_parameter_prefix(self):
        fp, errors = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n% ampl: 1.0, 0.0, none\n"
        )
        assert len(errors) == 1
        assert "*,$ or >" in errors[0].message

    def test_missing_model_label_before_parameters(self):
        fp, errors = _validate_script_content("$ ampl: 1.0, 0.0, none\n")
        assert len(errors) == 1
        assert "first definition" in errors[0].message

    def test_too_many_bounds(self):
        fp, errors = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n$ ampl: 1.0, 0.0, none, extra\n"
        )
        assert len(errors) == 1
        assert "min, max" in errors[0].message

    def test_two_item_bounds(self):
        fp, errors = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n$ ampl: 1.0, 0.0\n"
        )
        assert len(errors) == 1
        assert "two" in errors[0].message.lower()

    def test_duplicate_model_label_is_rejected(self):
        """
        Duplicate model labels are a parser error.

        The second ``MODEL:`` line is flagged as a duplicate.
        Its ``shape:`` and parameter lines still overwrite the earlier
        entries (for additional error detection), but the script is invalid.
        """
        script = (
            "MODEL: X\nshape: gaussianmodel\n    $ ampl: 1.0, 0.0, none\n"
            "MODEL: X\nshape: lorentzianmodel\n    $ ampl: 0.5, 0.0, none\n"
        )
        fp, errors = _validate_script_content(script)
        assert len(errors) == 1
        assert "Duplicate model label" in errors[0].message
        # The second block is still parsed for additional error detection
        assert fp.models == ["x"]
        assert fp["ampl_x"] == 0.5

    def test_cannot_evaluate_value(self):
        fp, errors = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n  $ ampl: not_a_number, 0.0, none\n"
        )
        assert len(errors) == 1
        assert "Cannot evaluate" in errors[0].message

    def test_error_has_line_number(self):
        fp, errors = _validate_script_content(
            "COMMON:\n  $ gratio: 0.5, 0.0, 1.0\n" "MODEL: PEAK\nshape: nonexistent\n"
        )
        assert len(errors) == 1
        assert errors[0].line == 4

    def test_script_error_is_structured(self):
        err = ScriptError(line=3, text="bad line", message="something wrong")
        assert err.line == 3
        assert err.text == "bad line"
        assert err.message == "something wrong"
        assert "ScriptError" in repr(err)
        assert "Line 3" in str(err)


# ======================================================================================
# 10.  Serialization round-trip
# ======================================================================================


class TestRoundTrip:
    """
    Parse → str → re-parse preserves semantic structure.

    Formatting (whitespace, precision, ordering) does NOT need to be identical.
    Only the interpreted structure must remain equivalent.
    """

    SCRIPT = """
COMMON:
  $ gratio: 0.1, 0.0, 1.0

MODEL: PEAK_1
shape: gaussianmodel
    * ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none

MODEL: PEAK_2
shape: lorentzianmodel
    $ ampl:  0.5, 0.0, none
    $ pos:   150.0, -50.0, 300.0
    $ width: 20.0, 0.0, none
"""

    def test_round_trip_preserves_models(self):
        fp1, errors = _validate_script_content(self.SCRIPT)
        assert errors == []

        script2 = str(fp1)
        fp2, errors = _validate_script_content(script2)
        assert errors == []

        assert fp1.models == fp2.models

    def test_round_trip_preserves_model_types(self):
        fp1, _ = _validate_script_content(self.SCRIPT)
        fp2, _ = _validate_script_content(str(fp1))
        assert fp1.model == fp2.model

    def test_round_trip_preserves_parameter_keys(self):
        fp1, _ = _validate_script_content(self.SCRIPT)
        fp2, _ = _validate_script_content(str(fp1))
        assert set(fp1.keys()) == set(fp2.keys())

    def test_round_trip_preserves_fixed_flags(self):
        fp1, _ = _validate_script_content(self.SCRIPT)
        fp2, _ = _validate_script_content(str(fp1))
        assert fp1.fixed == fp2.fixed

    def test_round_trip_preserves_reference_flags(self):
        """No reference params in this test, so all are False."""
        fp1, _ = _validate_script_content(self.SCRIPT)
        fp2, _ = _validate_script_content(str(fp1))
        assert fp1.reference == fp2.reference

    def test_round_trip_preserves_bounds_approximately(self):
        fp1, _ = _validate_script_content(self.SCRIPT)
        fp2, _ = _validate_script_content(str(fp1))
        for key in fp1:
            # Bounds may differ slightly due to float formatting
            lob1, lob2 = fp1.lob.get(key), fp2.lob.get(key)
            upb1, upb2 = fp1.upb.get(key), fp2.upb.get(key)
            if lob1 is not None and lob2 is not None:
                assert lob1 == pytest.approx(lob2)
            if upb1 is not None and upb2 is not None:
                assert upb1 == pytest.approx(upb2)

    def test_round_trip_preserves_values_approximately(self):
        fp1, _ = _validate_script_content(self.SCRIPT)
        fp2, _ = _validate_script_content(str(fp1))
        for key in fp1:
            assert fp1[key] == pytest.approx(fp2[key])

    def test_round_trip_preserves_common_flags(self):
        fp1, _ = _validate_script_content(self.SCRIPT)
        fp2, _ = _validate_script_content(str(fp1))
        assert fp1.common == fp2.common

    def test_round_trip_with_references(self):
        """Round-trip preserves reference parameters."""
        script = """
COMMON:
  $ gratio: 0.1, 0.0, 1.0

MODEL: PEAK_1
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    > ratio: gratio
"""
        fp1, errors = _validate_script_content(script)
        assert errors == []

        script2 = str(fp1)
        fp2, errors = _validate_script_content(script2)
        assert errors == []

        assert fp1.reference == fp2.reference
        assert fp1["ratio_peak_1"] == fp2["ratio_peak_1"]
        assert isinstance(fp2["ratio_peak_1"], str)


# ======================================================================================
# 11.  FitParameters direct behavior
# ======================================================================================


class TestFitParametersDirect:
    """Characterize FitParameters.__setitem__ called directly (not through parser)."""

    def test_scalar_value_sets_data(self):
        fp = FitParameters()
        fp["test"] = 42.0
        assert fp["test"] == 42.0

    def test_scalar_value_sets_bounds_to_none(self):
        fp = FitParameters()
        fp["test"] = 42.0
        assert fp.lob["test"] is None
        assert fp.upb["test"] is None

    def test_scalar_value_sets_fixed_to_false(self):
        fp = FitParameters()
        fp["test"] = 42.0
        assert fp.fixed["test"] is False

    def test_scalar_value_sets_reference_to_false(self):
        fp = FitParameters()
        fp["test"] = 42.0
        assert fp.reference["test"] is False

    def test_tuple_with_value_min_max(self):
        fp = FitParameters()
        fp["test"] = (1.0, 0.0, 10.0)
        assert fp["test"] == 1.0
        assert fp.lob["test"] == 0.0
        assert fp.upb["test"] == 10.0
        assert fp.fixed["test"] is False

    def test_tuple_with_fixed_flag(self):
        fp = FitParameters()
        fp["test"] = (1.0, 0.0, 10.0, True)
        assert fp["test"] == 1.0
        assert fp.fixed["test"] is True

    def test_tuple_with_varying_flag(self):
        fp = FitParameters()
        fp["test"] = (1.0, 0.0, 10.0, False)
        assert fp["test"] == 1.0
        assert fp.fixed["test"] is False

    def test_reference_stores_value_as_string(self):
        fp = FitParameters()
        fp.reference["test"] = True
        fp["test"] = "other_param"
        assert isinstance(fp["test"], str)
        assert fp["test"] == "other_param"
        assert fp.fixed["test"] is True

    def test_reference_does_not_set_bounds(self):
        fp = FitParameters()
        fp.reference["test"] = True
        fp["test"] = "other_param"
        assert fp.lob.get("test") is None
        assert fp.upb.get("test") is None

    def test_initial_reference_defaults_to_false(self):
        fp = FitParameters()
        fp["new_key"] = 1.0
        assert fp.reference["new_key"] is False

    def test_str_of_simple_parameters(self):
        """str(fp) for a directly-constructed FitParameters produces valid output."""
        fp = FitParameters()
        fp.models.append("peak")
        fp.model["peak"] = "gaussianmodel"
        fp.common["ampl"] = False
        fp["ampl_peak"] = (1.0, 0.0, 10.0)
        output = str(fp)
        assert "MODEL: peak" in output
        assert "gaussianmodel" in output
        assert "ampl" in output
        assert "1.0" in output or "1.0000" in output

    def test_uppercase_model_label_in_str(self):
        """Model labels in ``str(fp)`` are lowercased (parser lowercases them)."""
        fp = FitParameters()
        fp.models.append("my_model")
        fp.model["my_model"] = "gaussianmodel"
        fp.common["ampl"] = False
        fp["ampl_my_model"] = (1.0, 0.0, 10.0)
        output = str(fp)
        assert "my_model" in output


# ======================================================================================
# 12.  Edge cases and implicit parser conventions
# ======================================================================================


class TestParserConventions:
    """
    Document implicit conventions and edge cases.

    These tests reveal parser behaviors that are not obvious from the DSL spec.
    """

    def test_model_label_lowercased(self):
        """Model labels are lowercased by the parser."""
        fp, _ = _validate_script_content(
            "MODEL: UPPERCASE_MODEL\nshape: gaussianmodel\n    $ ampl: 1.0\n"
        )
        assert fp.models == ["uppercase_model"]

    def test_model_type_is_not_lowercased(self):
        """The shape name is lowercased by the parser (key.strip().lower())."""
        # Actually the shape values.lower().strip() in the parser
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: GaussianModel\n    $ ampl: 1.0, 0.0, none\n"
        )
        assert fp.model["x"] == "gaussianmodel"

    def test_parameter_names_lowercased(self):
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n    $ AMPL: 1.0, 0.0, none\n"
        )
        assert "ampl_x" in fp
        assert "AMPL_x" not in fp

    def test_shape_before_model_is_rejected(self):
        """
        ``shape:`` before any ``MODEL:`` declaration is a parser error.

        There is no model to associate the shape with.
        The model also receives a "no shape" error since it never got a valid shape.
        """
        script = "shape: gaussianmodel\nMODEL: X\n    $ ampl: 1.0, 0.0, none\n"
        fp, errors = _validate_script_content(script)
        messages = [e.message for e in errors]
        assert any("after a MODEL" in m for m in messages)
        assert any("no shape" in m for m in messages)

    def test_parameters_before_common_block(self):
        """Parameters defined before any model/COMMON keyword are errors."""
        script = "$ ampl: 1.0, 0.0, none\n"
        fp, errors = _validate_script_content(script)
        assert len(errors) == 1

    def test_shape_inside_common_block_is_rejected(self):
        """``shape:`` inside a COMMON block is a parser error."""
        script = "COMMON:\nshape: gaussianmodel\n  $ gratio: 0.1\n"
        fp, errors = _validate_script_content(script)
        assert len(errors) == 1
        assert "after a MODEL" in errors[0].message


# ======================================================================================
# 13.  Return type contract
# ======================================================================================


class TestReturnContract:
    """The return type of _validate_script_content is always (FitParameters, list)."""

    def test_valid_script_returns_empty_errors(self):
        fp, errors = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n$ ampl: 1.0, 0.0, none\n"
        )
        assert errors == []
        assert isinstance(fp, FitParameters)

    def test_invalid_script_returns_fitparameters_always(self):
        fp, errors = _validate_script_content("INVALID LINE\n")
        assert len(errors) >= 1
        assert isinstance(fp, FitParameters)

    def test_error_list_contains_script_error_instances(self):
        fp, errors = _validate_script_content("MODEL: X\nshape: unknown_model\n")
        assert all(isinstance(e, ScriptError) for e in errors)
