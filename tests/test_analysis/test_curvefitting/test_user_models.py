"""Regression tests for independent user-defined fitting models."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting import _models


def _linear(x, ampl):
    return ampl * x


def _quadratic(x, ampl):
    return ampl * x**2


def test_registered_functions_and_base_class_are_independent():
    original_function = _models.usermodel.f
    original_args = _models.usermodel.args.copy()

    def decay(x, ampl, rate):
        return ampl * np.exp(-rate * x)

    fit = scp.Optimize(usermodels={"LINEAR": _linear, "Decay": decay})
    line = fit.usermodels["linear"]
    exponential = fit.usermodels["decay"]
    assert line is not exponential
    assert line.args == ["ampl"]
    assert exponential.args == ["ampl", "rate"]
    np.testing.assert_array_equal(line().f(np.array([2.0]), 3.0), [6.0])
    np.testing.assert_allclose(exponential().f(np.array([2.0]), 3.0, 0.5), [3 / np.e])
    assert _models.usermodel.f is original_function
    assert _models.usermodel.args == original_args


def test_separate_estimators_and_reassignment_do_not_replace_existing_functions():
    first = scp.Optimize(usermodels={"shape": _linear})
    original = first.usermodels["shape"]
    second = scp.Optimize(usermodels={"shape": _quadratic})
    x = np.array([2.0])
    np.testing.assert_array_equal(first.usermodels["shape"]().f(x, 3.0), [6.0])
    np.testing.assert_array_equal(second.usermodels["shape"]().f(x, 3.0), [12.0])

    first.usermodels = {"shape": _quadratic}
    np.testing.assert_array_equal(original().f(x, 3.0), [6.0])


def test_two_user_functions_fit_together():
    x = np.linspace(1, 3, 81)
    data = scp.NDDataset(_linear(x, 2.0) + _quadratic(x, 0.3))
    data.x = scp.Coord(x)
    fit = scp.Optimize(
        usermodels={"linear": _linear, "quadratic": _quadratic},
        amplitude_mode="area",
        autobase=False,
    )
    fit.script = """
MODEL: line
shape: linear
    $ ampl: 1.0, 0.0, 10.0
MODEL: curve
shape: quadratic
    $ ampl: 1.0, 0.0, 10.0
"""
    fit.fit(data)
    assert fit.fp["ampl_line"] == pytest.approx(2.0, abs=1e-7)
    assert fit.fp["ampl_curve"] == pytest.approx(0.3, abs=1e-7)


def test_user_model_instances_keep_their_bound_state_and_argument_lists():
    class Shifted(_models.usermodel):
        args = ["ampl"]

        def __init__(self, offset):
            self.offset = offset

        def f(self, x, ampl):
            return ampl * x + self.offset

    left, right = Shifted(1.0), Shifted(5.0)
    fit = scp.Optimize(usermodels={"left": left, "right": right})
    x = np.array([2.0])
    np.testing.assert_array_equal(fit.usermodels["left"]().f(x, 3.0), [7.0])
    np.testing.assert_array_equal(fit.usermodels["right"]().f(x, 3.0), [11.0])
    assert fit.usermodels["left"].args is not left.args
    assert fit.usermodels["left"].args is not fit.usermodels["right"].args
