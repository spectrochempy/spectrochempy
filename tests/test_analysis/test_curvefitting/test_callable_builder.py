"""Private preparation of canonical models from Python callables."""

from functools import partial

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._callable_spec import _callable_model_spec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec


def _decay(time, ampl=2.0, rate=0.5, offset=0.2):
    return ampl * np.exp(-rate * time) + offset


def test_function_defaults_and_parameter_definitions_fit_without_script():
    offset = _ParamSpec("offset", 0.2, vary=False)
    rate = _ParamSpec("rate", 0.5, bounds=(0.1, 2.0))
    estimator = scp.Optimize._from_callable(
        _decay, {"offset": offset, "rate": rate}, label="signal", method="least_squares"
    )
    assert estimator.fp["ampl_signal"] == 2.0
    assert estimator.amplitude_mode == "area"
    assert not estimator.autoampl
    assert not estimator.autobase
    rate.value = 1.8
    offset.value = 0.7
    assert estimator.fp["rate_signal"] == 0.5
    assert estimator.fp["offset_signal"] == 0.2

    time = np.linspace(0, 5, 101)
    data = scp.NDDataset(_decay(time, 3.0, 0.8, 0.2), coordset=[scp.Coord(time)])
    estimator.fit(data)
    assert estimator.result.success
    assert estimator.fp["ampl_signal"] == pytest.approx(3.0, abs=1e-6)
    assert estimator.fp["rate_signal"] == pytest.approx(0.8, abs=1e-6)
    assert estimator.fp["offset_signal"] == 0.2


def test_plain_callable_does_not_treat_first_parameter_as_peak_height():
    def exponential(time, rate=0.5):
        return np.exp(-rate * time)

    time = np.linspace(0, 4, 81)
    data = scp.NDDataset(exponential(time, 0.8), coordset=[scp.Coord(time)])
    estimator = scp.Optimize._from_callable(exponential, method="least_squares").fit(
        data
    )
    assert estimator.fp["rate_model"] == pytest.approx(0.8, abs=1e-6)


@pytest.mark.parametrize("kind", ["method", "object", "partial"])
def test_callable_forms_preserve_bound_state(kind):
    class Decay:
        def __init__(self, offset):
            self.offset = offset

        def evaluate(self, time, ampl=2.0, rate=0.5):
            return _decay(time, ampl, rate, self.offset)

        __call__ = evaluate

    model = Decay(0.4)
    function = {"method": model.evaluate, "object": model, "partial": partial(_decay)}[
        kind
    ]
    estimator = scp.Optimize._from_callable(function)
    adapter = estimator.usermodels["pythonmodel"]()
    time = np.array([0.0, 1.0])
    args = [estimator.fp[f"{name}_model"] for name in adapter.args]
    expected = _decay(time, offset=0.2 if kind == "partial" else 0.4)
    np.testing.assert_array_equal(adapter.f(time, *args), expected)


@pytest.mark.parametrize("shared_name", ["sharedrate", "r2"])
def test_existing_common_reference_is_kept_in_canonical_parameters(shared_name):
    definition = _ParamSpec("rate", vary=False, reference=shared_name)
    common = _ParamSpec(shared_name, 0.5, bounds=(0.1, 2.0))
    estimator = scp.Optimize._from_callable(
        _decay,
        {"rate": definition, "offset": _ParamSpec("offset", 0.2, vary=False)},
        common_params={shared_name: common},
        method="least_squares",
    )
    time = np.linspace(0, 5, 101)
    data = scp.NDDataset(_decay(time, 3.0, 0.8), coordset=[scp.Coord(time)])
    estimator.fit(data)
    assert estimator.fp[shared_name] == pytest.approx(0.8, abs=1e-6)
    assert estimator._model_spec.components[0].params["rate"].reference == shared_name
    assert common.value == 0.5


@pytest.mark.parametrize("name", ["r2x", "a1b2"])
def test_common_identifiers_outside_legacy_reference_language_are_rejected(name):
    with pytest.raises(ValueError, match="COMMON"):
        scp.Optimize._from_callable(
            _decay,
            {"rate": _ParamSpec("rate", vary=False, reference=name)},
            common_params={name: _ParamSpec(name, 0.5)},
        )


def test_function_parameter_with_interleaved_digits_uses_positional_binding():
    def line(x, r2x=1.0):
        return r2x * x

    x = np.linspace(1, 3, 41)
    data = scp.NDDataset(3 * x, coordset=[scp.Coord(x)])
    estimator = scp.Optimize._from_callable(line, method="least_squares").fit(data)
    assert estimator.fp["r2x_model"] == pytest.approx(3.0, abs=1e-6)


def test_required_parameters_need_explicit_canonical_definitions():
    def line(x, ampl):
        return ampl * x

    with pytest.raises(ValueError, match="initial definition"):
        scp.Optimize._from_callable(line)
    estimator = scp.Optimize._from_callable(line, {"ampl": _ParamSpec("ampl", 2.0)})
    assert estimator.fp["ampl_model"] == 2.0


@pytest.mark.parametrize(
    "function",
    [lambda: 1, lambda x, *args: x, lambda x, **kwargs: x, lambda x, *, rate=1: x],
)
def test_unsupported_signatures_are_rejected(function):
    with pytest.raises(ValueError, match="axis and positional"):
        scp.Optimize._from_callable(function)


@pytest.mark.parametrize("value", [None, [], float("nan"), float("inf"), 1j])
def test_non_scalar_or_non_finite_defaults_require_override(value):
    def model(x, ampl=value):
        return x

    with pytest.raises(ValueError, match="finite real scalar"):
        scp.Optimize._from_callable(model)
    assert scp.Optimize._from_callable(model, {"ampl": _ParamSpec("ampl", 1.0)})


@pytest.mark.parametrize(
    "function", [lambda x, UPPER=1: x, lambda x, has_underscore=1: x]
)
def test_parameter_names_outside_current_1d_convention_are_rejected(function):
    with pytest.raises(ValueError, match="lowercase letters and digits"):
        scp.Optimize._from_callable(function)


@pytest.mark.parametrize("label", ["Upper", "", "has-dash", 1])
def test_invalid_component_labels_are_rejected(label):
    with pytest.raises(ValueError, match="label"):
        scp.Optimize._from_callable(_decay, label=label)


def test_bad_parameter_mapping_and_concurrent_registry_are_rejected():
    with pytest.raises(TypeError, match="callable"):
        scp.Optimize._from_callable(3)
    with pytest.raises(TypeError, match="mapping"):
        scp.Optimize._from_callable(_decay, [])
    with pytest.raises(ValueError, match="Unknown"):
        scp.Optimize._from_callable(_decay, {"unknown": _ParamSpec("unknown", 1)})
    with pytest.raises(ValueError, match="matching"):
        scp.Optimize._from_callable(_decay, {"rate": _ParamSpec("other", 1)})
    with pytest.raises(ValueError, match="usermodels"):
        scp.Optimize._from_callable(_decay, usermodels={})
    with pytest.raises(TypeError, match="common_params"):
        _callable_model_spec(_decay, common_params=[])
    with pytest.raises(ValueError, match="COMMON"):
        _callable_model_spec(_decay, common_params={"rate": _ParamSpec("other", 1)})
