"""Shared preparation metadata without changing the model or solver contracts."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.optimize import least_squares

from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._optimize_nd import _parameter
from spectrochempy.analysis.curvefitting._optimize_nd import _prepare_parameter
from spectrochempy.analysis.curvefitting._optimize_profiles import _prepare_profiles
from spectrochempy.analysis.curvefitting._parameter_space import _internal_vector
from spectrochempy.analysis.curvefitting._parameter_space import _prepared_parameters
from spectrochempy.analysis.curvefitting._parameter_space import _vector_from_prepared
from spectrochempy.analysis.curvefitting._parameter_transform import _to_external
from spectrochempy.analysis.curvefitting._prepared_parameter import _PreparedParameter
from spectrochempy.core.units import Quantity
from spectrochempy.core.units import ur


def test_canonical_identities_preserve_component_ownership():
    spec = _FitModelSpec(
        common_params={"width": _ParamSpec("width", 2.0)},
        components=[
            _ComponentSpec("b_c", params={"a": _ParamSpec("a", 3.0)}),
            _ComponentSpec("c", params={"a_b": _ParamSpec("a_b", 4.0)}),
        ],
    )
    records = _prepared_parameters(spec)
    assert [record.identifier for record in records] == [
        ("components", "b_c", "a"),
        ("components", "c", "a_b"),
        ("common", "width"),
    ]
    assert records[0].name == records[1].name == "a_b_c"
    assert _prepared_parameters(spec) == records
    with pytest.raises(FrozenInstanceError):
        records[0].name = "changed"


@pytest.mark.parametrize("legacy", [False, True])
def test_experiment_records_are_snapshots_with_stable_ids(legacy):
    spec = _FitModelSpec(
        common_params={
            "k": _ParamSpec("k", [2.0, 3.0]),
            "fixed": _ParamSpec("fixed", [4.0, 5.0], vary=False),
        },
        expvars=["k"],
        expnumber=2,
    )
    definition = spec.to_fitparameters() if legacy else spec
    records = _prepared_parameters(definition)
    assert [record.name for record in records] == ["fixed", "k_exp0", "k_exp1"]
    assert records[1].identifier[-2:] == ("experiments", 0)
    assert records[2].identifier[-2:] == ("experiments", 1)
    assert len({record.identifier for record in records}) == 3
    if legacy:
        definition.data["fixed"][0] = 100.0
        definition.data["k"][0] = 100.0
    else:
        spec.common_params["fixed"].value[0] = 100.0
        spec.common_params["k"].value[0] = 100.0
    assert records[0].value == [4.0, 5.0]
    assert records[1].value == 2.0


def test_fixed_legacy_reference_does_not_require_numeric_bounds():
    spec = _FitModelSpec(
        common_params={"ratio": _ParamSpec("ratio", 0.5)},
        components=[
            _ComponentSpec(
                "peak",
                params={
                    "ratio": _ParamSpec("ratio", vary=False, reference="ratio"),
                },
            ),
        ],
    )
    legacy = spec.to_fitparameters()
    assert "ratio_peak" not in legacy.lob
    records = _prepared_parameters(legacy)
    assert records[1].fixed
    assert records[1].value == "ratio"
    assert _internal_vector(legacy) == ([0.5], ["ratio"])


@pytest.mark.parametrize("value", [2**53 + 1, np.longdouble("0.1234567890123456789")])
def test_nd_preparation_preserves_numeric_precision(value):
    record = _prepare_parameter(
        {"value": value, "fixed": True},
        "center",
        None,
        identifier=("peaks", 0, "axes", "x", "center"),
    )
    assert isinstance(record, _PreparedParameter)
    assert record.value == value
    assert type(record.value) is type(value)
    assert _parameter({"value": value, "fixed": True}, "center", None) == (
        value,
        -np.inf,
        np.inf,
        True,
    )


def test_nd_width_metadata_uses_interval_units():
    record = _prepare_parameter(
        {"value": Quantity(2, "delta_degC"), "bounds": (1, 3)},
        "width",
        ur.degC,
        width=True,
    )
    assert record.value == 2
    assert record.bounds == (1, 3)
    assert record.unit == ur.delta_degC


def test_shared_profile_parameter_has_one_identity_and_multiple_unit_bindings():
    def decay(t, tau):
        return np.exp(-t / tau)

    peak = {
        "height": {"value": 1, "fixed": True},
        "axes": {
            dim: {
                "model": decay,
                "parameters": {
                    "tau": {"shared": "tau", "unit": "ms" if dim == "x" else "s"},
                },
            }
            for dim in ("x", "y")
        },
    }
    peaks, groups, _ = _prepare_profiles(
        [peak],
        {"tau": {"value": 2, "bounds": (0.1, 10), "unit": "s"}},
        None,
        {"x": ur.ms, "y": ur.s},
    )
    assert len(groups) == 1
    record = groups[0]["parameter"]
    assert isinstance(record, _PreparedParameter)
    assert record.identifier == ("shared_parameters", "tau")
    assert record.unit == ur.s
    assert record.value == 2
    assert record.bounds == (0.1, 10)
    assert len(groups[0]["bindings"]) == 2
    assert peaks[0]["axes"]["x"]["parameters"]["tau"] == 2000
    assert peaks[0]["axes"]["y"]["parameters"]["tau"] == 2


def test_parameter_record_supports_a_kinetic_residual_without_spectral_topology():
    rate = _PreparedParameter(
        ("kinetics", "reaction_1", "k"), "k", 0.2, (0.01, 2.0), unit=ur.s**-1
    )
    start, names = _vector_from_prepared([rate])
    time = np.linspace(0, 5, 21)
    observed = np.exp(-0.7 * time)

    def residual(vector):
        k = _to_external(vector[0], *rate.bounds)
        return np.exp(-k * time) - observed

    result = least_squares(residual, start)
    assert names == ["k"]
    assert result.success
    assert _to_external(result.x[0], *rate.bounds) == pytest.approx(0.7)
