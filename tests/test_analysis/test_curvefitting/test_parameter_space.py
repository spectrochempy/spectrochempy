"""Compatibility of parameter preparation across fitting representations."""

import numpy as np
import pytest

from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._parameter_space import _internal_vector
from spectrochempy.analysis.curvefitting._parameter_space import _restore_external


@pytest.fixture(params=["canonical", "legacy"])
def parameter_definition(request):
    spec = _FitModelSpec(
        common_params={
            "z": _ParamSpec("z", 8.0, bounds=(None, None)),
            "b": _ParamSpec("b", 3.0, bounds=(1.0, None)),
            "a": _ParamSpec("a", 4.0, bounds=(0.0, 10.0)),
            "c": _ParamSpec("c", 2.0, bounds=(None, 5.0)),
            "fixed": _ParamSpec("fixed", 7.0, vary=False),
        },
    )
    return spec if request.param == "canonical" else spec.to_fitparameters()


def physical_values(definition):
    if isinstance(definition, _FitModelSpec):
        return {name: param.value for name, param in definition.common_params.items()}
    return dict(definition.data)


def test_ordered_roundtrip_preserves_bounds_and_fixed_values(parameter_definition):
    initial = physical_values(parameter_definition)
    vector, keys = _internal_vector(parameter_definition)
    assert keys == ["a", "b", "c", "z"]
    restored = _restore_external(parameter_definition, vector, keys)
    assert restored is parameter_definition
    for key, value in initial.items():
        np.testing.assert_allclose(physical_values(restored)[key], value, atol=1e-14)


def test_partial_reordered_restore_uses_names(parameter_definition):
    initial = physical_values(parameter_definition)
    restored = _restore_external(parameter_definition, [12.0, 0.0], ["z", "a"])
    actual = physical_values(restored)
    assert actual == {**initial, "z": 12.0, "a": 5.0}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("n_experiments", [1, 2])
def test_experiment_parameter_roundtrip(legacy, n_experiments):
    values = [2.0, 4.0][:n_experiments]
    spec = _FitModelSpec(
        common_params={
            "a": _ParamSpec("a", values, bounds=(0.0, 10.0)),
            "z": _ParamSpec("z", 5.0),
        },
        expvars=["a"],
        expnumber=n_experiments,
    )
    definition = spec.to_fitparameters() if legacy else spec
    vector, keys = _internal_vector(definition)
    assert keys == [*(f"a_exp{i}" for i in range(n_experiments)), "z"]
    restored = _restore_external(definition, vector, keys)
    actual = physical_values(restored)
    np.testing.assert_allclose(actual["a"], values, atol=1e-14)
    assert actual["z"] == 5.0
    if legacy and n_experiments == 1:
        assert np.isscalar(actual["a"])
    else:
        assert isinstance(actual["a"], list)


@pytest.mark.parametrize("legacy", [False, True])
def test_no_free_parameters(legacy):
    spec = _FitModelSpec(common_params={"a": _ParamSpec("a", 3.0, vary=False)})
    definition = spec.to_fitparameters() if legacy else spec
    assert _internal_vector(definition) == ([], [])
    assert _restore_external(definition, [], []) is definition
    assert physical_values(definition) == {"a": 3.0}
