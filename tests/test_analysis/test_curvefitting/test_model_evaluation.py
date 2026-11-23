"""Exercise the callable evaluation boundary and its historical model adapter."""

import numpy as np
import pytest

from spectrochempy.analysis.curvefitting import _models
from spectrochempy.analysis.curvefitting._fit_backend import _least_squares
from spectrochempy.analysis.curvefitting._model_evaluation import _evaluate_callable
from spectrochempy.analysis.curvefitting.optimize import getmodel


@pytest.mark.parametrize("mode", ["area", "height", "HEIGHT"])
@pytest.mark.parametrize(
    "name, values",
    [
        ("gaussianmodel", [2.0, 0.3, 0.7]),
        ("lorentzianmodel", [2.0, 0.3, 0.7]),
        ("voigtmodel", [2.0, 0.3, 0.7, 0.4]),
        ("pseudovoigtmodel", [2.0, 0.3, 0.7, 0.4]),
        ("asymmetricvoigtmodel", [2.0, 0.3, 0.7, 0.4, 0.2]),
        ("sigmoidmodel", [2.0, 0.3, 0.2]),
        ("polynomialbaseline", [2.0, 0.3, 0.2]),
    ],
)
def test_builtin_adapter_preserves_profile_and_scaling(name, values, mode):
    x = np.linspace(-2, 2, 101, dtype=np.float32)
    model = getattr(_models, name)()
    par = {f"{key}_peak": val for key, val in zip(model.args, values, strict=False)}
    par = type("Parameters", (dict,), {"model": {"peak": name}})(par)
    expected = model.f(np.array(x, dtype=np.float64), *values)
    if mode.lower() == "height":
        expected = values[0] * expected / expected.max()

    actual = getmodel(x, modelname="peak", par=par, amplitude_mode=mode)
    np.testing.assert_array_equal(actual, expected)


def test_callable_forwards_second_axis_keywords_and_exceptions():
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.float32)
    seen = []

    def plane(x, y, slope, *, offset):
        assert x.dtype == y.dtype == np.float64
        seen.append((x, y))
        return slope * x + y + offset

    actual = _evaluate_callable(plane, x, [2], y=y, offset=5)
    np.testing.assert_array_equal(actual, [10, 13])
    assert not np.shares_memory(seen[0][0], x)
    assert not np.shares_memory(seen[0][1], y)

    failure = RuntimeError("model failure")

    def broken(x):
        raise failure

    with pytest.raises(RuntimeError) as error:
        _evaluate_callable(broken, x, [])
    assert error.value is failure


def test_custom_model_keeps_legacy_arguments_and_height_normalization():
    class Plane:
        args = ["AMPL", "slope"]

        def f(self, x, y, ampl, slope, *, offset, function, parameters):
            return ampl * (function(x) + slope * y + offset + parameters)

    class Parameters(dict):
        model = {"custom": "plane"}

    values = Parameters(ampl_custom=3, slope_custom=2)
    actual = getmodel(
        [1, 2],
        y=[2, 3],
        modelname="custom",
        par=values,
        usermodels={"plane": Plane},
        offset=4,
        function=lambda x: x,
        parameters=0,
    )
    raw = np.array([27.0, 36.0])
    np.testing.assert_array_equal(actual, 3 * raw / raw.max())


def test_kinetic_callable_fit_uses_shared_backend_without_script_or_peak_scaling():
    def decay(t, amplitude, rate, offset):
        return amplitude * np.exp(-rate * t) + offset

    time = np.linspace(0, 5, 101)
    target = decay(time, 3.0, 0.8, 0.2)

    def residual(parameters):
        return _evaluate_callable(decay, time, parameters) - target

    result = _least_squares(residual, [2.0, 0.5, 0.1], bounds=([0, 0, 0], [10, 5, 1]))
    assert result.success
    np.testing.assert_allclose(result.x, [3.0, 0.8, 0.2], atol=1e-7)
    np.testing.assert_allclose(result.fun, 0, atol=1e-7)


def test_legacy_model_function_is_resolved_after_axis_conversion():
    accesses = []
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])

    class Model:
        args = ["ampl"]

        @property
        def f(self):
            accesses.append(True)
            x[:] = 100
            y[:] = 200
            return lambda x, y, ampl: ampl * (x + y)

    class Parameters(dict):
        model = {"peak": "custom"}

    par = Parameters(ampl_peak=2)
    with pytest.raises(ValueError):
        getmodel(["invalid"], par=par, modelname="peak", usermodels={"custom": Model})
    assert accesses == []

    actual = getmodel(
        x,
        y=y,
        par=par,
        modelname="peak",
        usermodels={"custom": Model},
        amplitude_mode="area",
    )
    np.testing.assert_array_equal(actual, [8.0, 12.0])
    assert accesses == [True]
