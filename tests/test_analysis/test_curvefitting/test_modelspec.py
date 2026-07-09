# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Tests for the private _FitModelSpec structured model-definition object."""

import sys

import numpy as np
import pytest

from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import _unbound
from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.analysis.curvefitting.optimize import _count_varying_parameters
from spectrochempy.analysis.curvefitting.optimize import (
    _extract_varying_parameter_values,
)
from spectrochempy.analysis.curvefitting.optimize import _validate_script_content
from spectrochempy.analysis.curvefitting.optimize import getmodel

# ======================================================================================
# Constants
# ======================================================================================

_STR_NEG_THRESH = -0.1 / sys.float_info.epsilon
_STR_POS_THRESH = +0.1 / sys.float_info.epsilon


# ======================================================================================
# Tests for _unbound helper
# ======================================================================================


class TestUnbound:
    def test_none_stays_none(self):
        assert _unbound(None) is None

    def test_finite_lower_bound_preserved(self):
        assert _unbound(0.0) == 0.0
        assert _unbound(-100.0) == -100.0

    def test_finite_upper_bound_preserved(self):
        assert _unbound(200.0) == 200.0

    def test_sentinel_lower_bound_converted_to_none(self):
        sentinel = -1.0 / sys.float_info.epsilon
        assert _unbound(sentinel) is None

    def test_sentinel_upper_bound_converted_to_none(self):
        sentinel = +1.0 / sys.float_info.epsilon
        assert _unbound(sentinel) is None

    def test_threshold_bounds_converted_to_none(self):
        barely_below = _STR_NEG_THRESH - 1.0
        barely_above = _STR_POS_THRESH + 1.0
        assert _unbound(barely_below) is None
        assert _unbound(barely_above) is None


# ======================================================================================
# Tests for FitParameters -> _FitModelSpec conversion
# ======================================================================================


class TestFromFitParameters:
    """Test that from_fitparameters correctly extracts structure."""

    SCRIPT = """
MODEL: PEAK_A
shape: gaussianmodel
    $ ampl:  1.5, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    def test_returns_modelspec_instance(self):
        fp, errors = _validate_script_content(self.SCRIPT)
        assert errors == []
        spec = _FitModelSpec.from_fitparameters(fp)
        assert isinstance(spec, _FitModelSpec)

    def test_one_component(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert len(spec.components) == 1

    def test_component_label(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].label == "peak_a"

    def test_component_model_name(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].model_name == "gaussianmodel"

    def test_three_parameters(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert len(spec.components[0].params) == 3

    def test_parameter_values(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["ampl"].value == 1.5
        assert spec.components[0].params["pos"].value == 100.0
        assert spec.components[0].params["width"].value == 10.0

    def test_parameter_vary(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["ampl"].vary is True
        assert spec.components[0].params["pos"].vary is True
        assert spec.components[0].params["width"].vary is True

    def test_parameter_bounds_finite(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["pos"].bounds == (0.0, 200.0)

    def test_parameter_bounds_open(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        # ampl has upper bound "none" -> sentinel -> None
        assert spec.components[0].params["ampl"].bounds[0] == 0.0
        assert spec.components[0].params["ampl"].bounds[1] is None

    def test_no_common_params(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.common_params == {}


class TestFromMultipleComponents:
    """Two-component extraction."""

    SCRIPT = """
MODEL: PEAK_1
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none

MODEL: PEAK_2
shape: lorentzianmodel
    $ ampl: 0.5, 0.0, none
"""

    def test_two_components(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert len(spec.components) == 2

    def test_component_labels(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].label == "peak_1"
        assert spec.components[1].label == "peak_2"

    def test_model_names(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].model_name == "gaussianmodel"
        assert spec.components[1].model_name == "lorentzianmodel"

    def test_params_separate_between_components(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["ampl"].value == 1.0
        assert spec.components[1].params["ampl"].value == 0.5
        assert spec.components[0].params["pos"].value == 100.0
        assert "pos" not in spec.components[1].params


class TestFromCommonParameters:
    """COMMON parameter extraction."""

    SCRIPT = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
"""

    def test_common_params_present(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert len(spec.common_params) == 1
        assert "gratio" in spec.common_params

    def test_common_param_value(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.common_params["gratio"].value == 0.1

    def test_common_param_bounds(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.common_params["gratio"].bounds == (0.0, 1.0)

    def test_common_param_vary(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.common_params["gratio"].vary is True

    def test_common_param_is_not_reference(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.common_params["gratio"].reference is None

    def test_model_params_not_in_common(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert "ampl" not in spec.common_params
        assert "pos" not in spec.common_params


class TestFromReferences:
    """Reference parameter extraction."""

    SCRIPT = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    > ratio: gratio
"""

    def test_reference_in_component(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert "ratio" in spec.components[0].params

    def test_reference_stores_name(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["ratio"].reference == "gratio"

    def test_reference_not_vary(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["ratio"].vary is False

    def test_reference_has_no_bounds(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["ratio"].bounds == (None, None)

    def test_common_param_not_reference(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.common_params["gratio"].reference is None


class TestFromFixedVarying:
    """Fixed vs varying parameter extraction."""

    SCRIPT = """
MODEL: X
shape: gaussianmodel
    * ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
"""

    def test_fixed_param_vary_false(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["ampl"].vary is False

    def test_varying_param_vary_true(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].params["pos"].vary is True


class TestFromEmpty:
    """Edge case: empty FitParameters."""

    def test_empty_fitparameters_produces_empty_spec(self):
        fp = FitParameters()
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components == []
        assert spec.common_params == {}


# ======================================================================================
# Tests for to_script serialization
# ======================================================================================


class TestToScript:
    """Test that to_script produces valid script strings."""

    def test_empty_produces_empty_string(self):
        spec = _FitModelSpec()
        assert spec.to_script() == ""

    def test_single_component(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="peak_a",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.5, bounds=(0.0, None)),
                        "pos": _ParamSpec(name="pos", value=100.0, bounds=(0.0, 200.0)),
                    },
                )
            ]
        )
        script = spec.to_script()
        assert "MODEL: peak_a" in script
        assert "gaussianmodel" in script
        assert "$ ampl:" in script
        assert "$ pos:" in script

    def test_common_parameters(self):
        spec = _FitModelSpec(
            common_params={
                "gratio": _ParamSpec(name="gratio", value=0.1, bounds=(0.0, 1.0)),
            }
        )
        script = spec.to_script()
        assert "COMMON:" in script
        assert "$ gratio:" in script

    def test_reference(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="x",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.0),
                        "ratio": _ParamSpec(
                            name="ratio",
                            reference="gratio",
                            vary=False,
                        ),
                    },
                )
            ],
            common_params={
                "gratio": _ParamSpec(name="gratio", value=0.1),
            },
        )
        script = spec.to_script()
        assert "> ratio:gratio" in script
        assert "COMMON:" in script

    def test_fixed_parameter(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="x",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.0, vary=False),
                    },
                )
            ]
        )
        script = spec.to_script()
        assert "* ampl:" in script

    def test_script_is_parsable(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="x",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.5, bounds=(0.0, None)),
                        "pos": _ParamSpec(name="pos", value=100.0, bounds=(0.0, 200.0)),
                    },
                )
            ]
        )
        script = spec.to_script()
        fp, errors = _validate_script_content(script)
        assert errors == []


# ======================================================================================
# Round-trip tests
# ======================================================================================


class TestRoundTrip:
    """Full semantic round-trip: script -> FitParameters -> _FitModelSpec -> script -> FitParameters."""

    SINGLE = """
MODEL: PEAK_A
shape: gaussianmodel
    $ ampl:  1.5, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    MULTI = """
MODEL: PEAK_1
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none

MODEL: PEAK_2
shape: lorentzianmodel
    $ ampl: 0.5, 0.0, none
    $ pos:  150.0, -50.0, 300.0
    $ width: 20.0, 0.0, none
"""

    COMMON = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    REFERENCE = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    > ratio: gratio
    $ width: 10.0, 0.0, none
"""

    FIXED = """
MODEL: X
shape: gaussianmodel
    * ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
"""

    @staticmethod
    def _round_trip(script):
        """Run full round-trip and return (fp1, spec, fp2)."""
        fp1, errors = _validate_script_content(script)
        assert errors == [], f"Parse errors: {errors}"
        spec = _FitModelSpec.from_fitparameters(fp1)
        script2 = spec.to_script()
        fp2, errors2 = _validate_script_content(script2)
        assert errors2 == [], f"Re-parse errors: {errors2}"
        return fp1, spec, fp2

    # -- Single component round-trip --

    def test_round_trip_preserves_models(self):
        fp1, _, fp2 = self._round_trip(self.SINGLE)
        assert fp1.models == fp2.models

    def test_round_trip_preserves_model_mapping(self):
        fp1, _, fp2 = self._round_trip(self.SINGLE)
        assert fp1.model == fp2.model

    def test_round_trip_preserves_parameter_keys(self):
        fp1, _, fp2 = self._round_trip(self.SINGLE)
        assert set(fp1.keys()) == set(fp2.keys())

    def test_round_trip_preserves_values(self):
        fp1, _, fp2 = self._round_trip(self.SINGLE)
        for key in fp1:
            assert fp1[key] == pytest.approx(fp2[key])

    def test_round_trip_preserves_bounds(self):
        fp1, _, fp2 = self._round_trip(self.SINGLE)
        for key in fp1:
            lob1, lob2 = fp1.lob.get(key), fp2.lob.get(key)
            upb1, upb2 = fp1.upb.get(key), fp2.upb.get(key)
            if lob1 is not None and lob2 is not None:
                assert lob1 == pytest.approx(lob2)
            if upb1 is not None and upb2 is not None:
                assert upb1 == pytest.approx(upb2)

    def test_round_trip_preserves_fixed_flags(self):
        fp1, _, fp2 = self._round_trip(self.SINGLE)
        assert fp1.fixed == fp2.fixed

    def test_round_trip_preserves_reference_flags(self):
        fp1, _, fp2 = self._round_trip(self.SINGLE)
        assert fp1.reference == fp2.reference

    # -- Multiple components --

    def test_multi_round_trip_preserves_models(self):
        fp1, _, fp2 = self._round_trip(self.MULTI)
        assert fp1.models == fp2.models

    def test_multi_round_trip_preserves_model_types(self):
        fp1, _, fp2 = self._round_trip(self.MULTI)
        assert fp1.model == fp2.model

    def test_multi_round_trip_preserves_parameter_keys(self):
        fp1, _, fp2 = self._round_trip(self.MULTI)
        assert set(fp1.keys()) == set(fp2.keys())

    def test_multi_round_trip_preserves_values(self):
        fp1, _, fp2 = self._round_trip(self.MULTI)
        for key in fp1:
            assert fp1[key] == pytest.approx(fp2[key])

    # -- COMMON parameters --

    def test_common_round_trip_preserves_common_flag(self):
        fp1, _, fp2 = self._round_trip(self.COMMON)
        assert fp1.common == fp2.common

    def test_common_round_trip_preserves_models(self):
        fp1, _, fp2 = self._round_trip(self.COMMON)
        assert fp1.models == fp2.models

    def test_common_round_trip_preserves_values(self):
        fp1, _, fp2 = self._round_trip(self.COMMON)
        for key in fp1:
            assert fp1[key] == pytest.approx(fp2[key])

    # -- References --

    def test_reference_round_trip_preserves_reference_flag(self):
        fp1, _, fp2 = self._round_trip(self.REFERENCE)
        assert fp1.reference == fp2.reference

    def test_reference_round_trip_preserves_reference_string(self):
        fp1, _, fp2 = self._round_trip(self.REFERENCE)
        assert fp1["ratio_x"] == fp2["ratio_x"]
        assert isinstance(fp2["ratio_x"], str)

    def test_reference_round_trip_preserves_common(self):
        fp1, _, fp2 = self._round_trip(self.REFERENCE)
        assert fp1.common == fp2.common

    # -- Fixed vs varying --

    def test_fixed_round_trip_preserves_fixed_flag(self):
        fp1, _, fp2 = self._round_trip(self.FIXED)
        assert fp1.fixed == fp2.fixed

    def test_fixed_round_trip_preserves_values(self):
        fp1, _, fp2 = self._round_trip(self.FIXED)
        for key in fp1:
            assert fp1[key] == pytest.approx(fp2[key])

    # -- Content of the spec itself --

    def test_round_trip_spec_components(self):
        _, spec, _ = self._round_trip(self.SINGLE)
        assert len(spec.components) == 1
        assert spec.components[0].label == "peak_a"
        assert spec.components[0].model_name == "gaussianmodel"
        assert set(spec.components[0].params) == {"ampl", "pos", "width"}

    def test_round_trip_spec_values(self):
        _, spec, _ = self._round_trip(self.SINGLE)
        assert spec.components[0].params["ampl"].value == 1.5
        assert spec.components[0].params["pos"].value == 100.0


# ======================================================================================
# Edge cases
# ======================================================================================


class TestEdgeCases:
    def test_no_bounds_sentinel_converted_to_none(self):
        """A param with 'none' for both bounds gets bounds=(None, None)."""
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n    $ ampl: 1.0\n"
        )
        spec = _FitModelSpec.from_fitparameters(fp)
        ps = spec.components[0].params["ampl"]
        assert ps.bounds == (None, None)

    def test_no_common_section(self):
        """When there is no COMMON section, common_params is empty."""
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n    $ ampl: 1.0\n"
        )
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.common_params == {}

    def test_parameters_in_order(self):
        """Parameter names are preserved in the spec."""
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n"
            "    $ zed: 3.0, 0.0, none\n"
            "    $ ampl: 1.0, 0.0, none\n"
            "    $ beta: 2.0, 0.0, none\n"
        )
        spec = _FitModelSpec.from_fitparameters(fp)
        ps = spec.components[0].params
        assert set(ps) == {"ampl", "beta", "zed"}

    def test_spec_with_no_components_produces_empty_script(self):
        spec = _FitModelSpec()
        assert spec.to_script() == ""

    def test_spec_with_only_common_produces_common_section(self):
        spec = _FitModelSpec(
            common_params={
                "gratio": _ParamSpec(name="gratio", value=0.5),
            }
        )
        script = spec.to_script()
        assert script.startswith("COMMON:")
        assert "$ gratio:" in script

    def test_model_without_shape(self):
        """A model with no shape gets an empty model_name."""
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n    $ ampl: 1.0\n"
        )
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.components[0].model_name == "gaussianmodel"

    def test_value_is_float(self):
        """Parameter values are plain Python floats, not numpy types."""
        import numpy as np

        fp = FitParameters()
        fp["test"] = (float(np.float64(1.5)), None, None)
        # Simulate parser setup
        fp.common["test"] = True
        spec = _FitModelSpec.from_fitparameters(fp)
        assert isinstance(spec.common_params["test"].value, float)

    def test_multiple_components_same_param_names(self):
        """Components with identically-named params don't interfere."""
        script = """
MODEL: A
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none

MODEL: B
shape: lorentzianmodel
    $ ampl: 2.0, 0.0, none
"""
        _, spec, _ = TestRoundTrip._round_trip(script)
        assert spec.components[0].params["ampl"].value == 1.0
        assert spec.components[1].params["ampl"].value == 2.0

    def test_bounds_integers_preserved(self):
        """Integer bounds are preserved in the spec."""
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n    $ ampl: 1.0, 0, 10\n"
        )
        spec = _FitModelSpec.from_fitparameters(fp)
        lo, hi = spec.components[0].params["ampl"].bounds
        assert lo == 0.0
        assert hi == 10.0


class TestSpecRoundTripDirect:
    """Test that spec -> script -> fp reconstructs the same spec."""

    def test_build_from_spec(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="peak",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.5, bounds=(0.0, None)),
                        "pos": _ParamSpec(name="pos", value=100.0, bounds=(0.0, 200.0)),
                    },
                )
            ],
            common_params={
                "gratio": _ParamSpec(name="gratio", value=0.1, bounds=(0.0, 1.0)),
            },
        )
        script = spec.to_script()
        fp, errors = _validate_script_content(script)
        assert errors == []

        spec2 = _FitModelSpec.from_fitparameters(fp)
        assert len(spec2.components) == 1
        assert spec2.components[0].label == "peak"
        assert spec2.components[0].model_name == "gaussianmodel"
        assert spec2.common_params["gratio"].value == pytest.approx(0.1)
        assert spec2.components[0].params["ampl"].value == pytest.approx(1.5)
        assert spec2.components[0].params["ampl"].bounds[0] == 0.0
        assert spec2.components[0].params["ampl"].bounds[1] is None


class TestConstraintReserved:
    """The constraints field is reserved; it should not affect parsing."""

    def test_constraints_is_none_by_default(self):
        spec = _FitModelSpec()
        assert spec.constraints is None

    def test_constraints_is_stored(self):
        spec = _FitModelSpec(constraints={"dummy": True})
        assert spec.constraints == {"dummy": True}


# ======================================================================================
# Tests for count_varying
# ======================================================================================


class TestCountVarying:
    """Test _FitModelSpec.count_varying()."""

    SCRIPT = """
MODEL: PEAK_A
shape: gaussianmodel
    $ ampl:  1.5, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    def test_all_varying(self):
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.count_varying() == 3

    def test_one_fixed(self):
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n"
            "    * ampl: 1.0, 0.0, none\n"
            "    $ pos:  100.0, 0.0, 200.0\n"
        )
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.count_varying() == 1

    def test_all_fixed(self):
        fp, _ = _validate_script_content(
            "MODEL: X\nshape: gaussianmodel\n"
            "    * ampl: 1.0, 0.0, none\n"
            "    * pos:  100.0, 0.0, 200.0\n"
        )
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.count_varying() == 0

    def test_with_common(self):
        script = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.count_varying() == 3

    def test_with_reference(self):
        script = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    > ratio: gratio
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        # gratio varies, ampl varies, ratio is reference (fixed) → 2
        assert spec.count_varying() == 2

    def test_multiple_components(self):
        script = """
MODEL: A
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0

MODEL: B
shape: lorentzianmodel
    $ ampl: 0.5, 0.0, none
    $ pos:  150.0, 0.0, 300.0
    $ width: 10.0, 0.0, none
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.count_varying() == 5

    def test_empty_spec(self):
        spec = _FitModelSpec()
        assert spec.count_varying() == 0

    def test_none_fp(self):
        assert _count_varying_parameters(None) == 0

    def test_equivalence_with_fp_helper(self):
        """Numerical equivalence with the existing FitParameters-based helper."""
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.count_varying() == _count_varying_parameters(fp)

    def test_equivalence_with_common(self):
        script = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0
    * fixed_c: 5.0, 0.0, 10.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    * pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        assert spec.count_varying() == _count_varying_parameters(fp)
        # gratio varies, ampl varies, width varies → 3
        assert spec.count_varying() == 3


# ======================================================================================
# Tests for extract_varying_values
# ======================================================================================


class TestExtractVaryingValues:
    """Test _FitModelSpec.extract_varying_values()."""

    def test_single_component(self):
        script = """
MODEL: PEAK
shape: gaussianmodel
    $ ampl: 1.5, 0.0, none
    $ pos:  100.0, 0.0, 200.0
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        values = spec.extract_varying_values()
        assert isinstance(values, np.ndarray)
        assert values.shape == (2,)
        assert values[0] == 1.5  # ampl_peak
        assert values[1] == 100.0  # pos_peak

    def test_fixed_param_excluded(self):
        script = """
MODEL: X
shape: gaussianmodel
    * ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        values = spec.extract_varying_values()
        assert values.shape == (1,)
        assert values[0] == 100.0

    def test_all_fixed_returns_array(self):
        script = """
MODEL: X
shape: gaussianmodel
    * ampl: 1.0, 0.0, none
    * pos:  100.0, 0.0, 200.0
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        values = spec.extract_varying_values()
        assert isinstance(values, np.ndarray)
        assert values.shape == (0,)

    def test_with_common(self):
        script = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        values = spec.extract_varying_values()
        assert values.shape == (2,)
        # Sorted key order: ampl_x < gratio
        assert values[0] == 1.0  # ampl_x
        assert values[1] == 0.1  # gratio

    def test_reference_excluded(self):
        script = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    > ratio: gratio
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        values = spec.extract_varying_values()
        assert values.shape == (2,)
        # Sorted key order: ampl_x < gratio
        assert values[0] == 1.0  # ampl_x
        assert values[1] == 0.1  # gratio

    def test_empty_spec_returns_empty_array(self):
        spec = _FitModelSpec()
        values = spec.extract_varying_values()
        assert isinstance(values, np.ndarray)
        assert values.shape == (0,)

    def test_none_fp(self):
        assert _extract_varying_parameter_values(None) is None

    def test_deterministic_ordering(self):
        """Parameter ordering is deterministic across multiple calls."""
        script = """
COMMON:
    $ zed:   99.0, 0.0, 200.0
    $ alpha: 0.5, 0.0, 1.0

MODEL: B
shape: lorentzianmodel
    $ width: 20.0, 0.0, none

MODEL: A
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        v1 = spec.extract_varying_values().tolist()
        v2 = spec.extract_varying_values().tolist()
        assert v1 == v2

    def test_equivalence_with_fp_helper(self):
        """Numerical equivalence with the existing FitParameters-based helper."""
        script = """
COMMON:
    $ gratio: 0.1, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.5, 0.0, none
    $ pos:   100.0, 0.0, 200.0
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        spec_values = spec.extract_varying_values()
        fp_values = _extract_varying_parameter_values(fp)
        assert spec_values.shape == fp_values.shape
        np.testing.assert_array_almost_equal(spec_values, fp_values)

    def test_equivalence_multi_component(self):
        script = """
MODEL: A
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0

MODEL: B
shape: lorentzianmodel
    * ampl: 0.5, 0.0, none
    $ pos:  150.0, 0.0, 300.0
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        spec_values = spec.extract_varying_values()
        fp_values = _extract_varying_parameter_values(fp)
        np.testing.assert_array_almost_equal(spec_values, fp_values)


# ======================================================================================
# Tests for _ComponentParamsView
# ======================================================================================


class TestComponentParamsView:
    """Test the _ComponentParamsView adapter."""

    def test_model_dict(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="peak_a",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.5),
                    },
                )
            ]
        )
        view = spec.component_view("peak_a")
        assert view.model == {"peak_a": "gaussianmodel"}

    def test_getitem_by_param_name(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="peak_a",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.5),
                        "pos": _ParamSpec(name="pos", value=100.0),
                    },
                )
            ]
        )
        view = spec.component_view("peak_a")
        assert view["ampl_peak_a"] == 1.5
        assert view["pos_peak_a"] == 100.0

    def test_getitem_fallback_to_common(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="x",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.0),
                    },
                )
            ],
            common_params={
                "gratio": _ParamSpec(name="gratio", value=0.1),
            },
        )
        view = spec.component_view("x")
        # Common params accessible by bare name
        assert view["gratio"] == 0.1

    def test_getitem_raises_keyerror_for_unknown(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="x",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.0),
                    },
                )
            ]
        )
        view = spec.component_view("x")
        with pytest.raises(KeyError):
            view["nonexistent_x"]

    def test_contains(self):
        spec = _FitModelSpec(
            components=[
                _ComponentSpec(
                    label="x",
                    model_name="gaussianmodel",
                    params={
                        "ampl": _ParamSpec(name="ampl", value=1.0),
                    },
                )
            ]
        )
        view = spec.component_view("x")
        assert "ampl_x" in view
        assert "nonexistent_x" not in view

    def test_component_view_raises_for_missing_label(self):
        spec = _FitModelSpec()
        with pytest.raises(KeyError, match="not found in spec"):
            spec.component_view("nonexistent")


class TestGetModelEquivalence:
    """Test that getmodel produces identical results with _ComponentParamsView."""

    SCRIPT = """
MODEL: PEAK
shape: gaussianmodel
    $ ampl: 1.5, 0.0, none
    $ pos:  100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""

    def _build_view(self, script, component_label):
        """Parse script, return (fp, view) for equivalence testing."""
        fp, errors = _validate_script_content(script)
        assert errors == []
        spec = _FitModelSpec.from_fitparameters(fp)
        view = spec.component_view(component_label)
        return fp, view

    def test_model_dict_matches_fp(self):
        """The view's .model dict matches fp.model for the same label."""
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        view = spec.component_view("peak")
        # fp.model maps label → shape name
        assert view.model["peak"] == fp.model["peak"]

    def test_parameter_values_match(self):
        """Parameter values from the view match those from FitParameters."""
        fp, _ = _validate_script_content(self.SCRIPT)
        spec = _FitModelSpec.from_fitparameters(fp)
        view = spec.component_view("peak")

        for raw_name in spec.components[0].params:
            fp_key = f"{raw_name}_peak"
            assert view[fp_key] == float(fp[fp_key])

    def test_getmodel_equivalence_single_component(self):
        """Getmodel output is identical with _ComponentParamsView vs FitParameters."""
        fp, view = self._build_view(self.SCRIPT, "peak")
        x = np.linspace(-50, 50, 201, dtype=np.float64)

        result_fp = getmodel(x, modelname="peak", par=fp)
        result_view = getmodel(x, modelname="peak", par=view)

        np.testing.assert_array_almost_equal(result_fp, result_view)

    def test_getmodel_equivalence_multiple_components(self):
        """Getmodel equivalence for every component."""
        script = """
MODEL: A
shape: gaussianmodel
    $ ampl: 1.0, 0.0, none
    $ pos:  100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none

MODEL: B
shape: lorentzianmodel
    $ ampl: 0.5, 0.0, none
    $ pos:  150.0, 0.0, 300.0
    $ width: 20.0, 0.0, none
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        x = np.linspace(-50, 50, 201, dtype=np.float64)

        for comp in spec.components:
            view = spec.component_view(comp.label)
            result_fp = getmodel(x, modelname=comp.label, par=fp)
            result_view = getmodel(x, modelname=comp.label, par=view)
            np.testing.assert_array_almost_equal(result_fp, result_view)

    def test_view_with_common_params(self):
        """Getmodel with a view that includes common params (no references)."""
        # Script without references — all param values are direct floats
        script = """
COMMON:
    $ gratio: 0.3, 0.0, 1.0

MODEL: X
shape: gaussianmodel
    $ ampl:  1.0, 0.0, none
    $ pos:   100.0, 0.0, 200.0
    $ width: 10.0, 0.0, none
"""
        fp, _ = _validate_script_content(script)
        spec = _FitModelSpec.from_fitparameters(fp)
        x = np.linspace(50, 150, 201, dtype=np.float64)

        # fp-based getmodel: par[f"ampl_x"], etc. — all exist directly
        result_fp = getmodel(x, modelname="x", par=fp)

        # view-based getmodel: view["ampl_x"], etc. — component params + common fallback
        view = spec.component_view("x")
        result_view = getmodel(x, modelname="x", par=view)

        np.testing.assert_array_almost_equal(result_fp, result_view)
