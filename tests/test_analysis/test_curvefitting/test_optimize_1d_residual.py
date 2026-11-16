"""Regression tests for the 1D residual and automatic baseline."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._models import gaussianmodel
from spectrochempy.analysis.curvefitting._models import usermodel


@pytest.mark.parametrize("baseline_first", [True, False])
def test_fit_masked_user_model_uses_full_axis(monkeypatch, baseline_first):
    coord = scp.Coord(np.linspace(2.0, -2.0, 65))
    x = coord.data
    mask = np.zeros(x.size, dtype=bool)
    mask[[5, 32, 60]] = True
    calls = []

    def profile(axis, ampl):
        calls.append(axis.copy())
        return ampl / (1.0 + (axis / 0.12) ** 2)

    monkeypatch.setattr(usermodel, "args", usermodel.args)
    monkeypatch.setattr(usermodel, "f", vars(usermodel)["f"])
    peak = """
MODEL: peak
shape: customprofile
    $ ampl: 1.0, 0.0, 4.0
"""
    baseline = """
MODEL: baseline1
shape: gaussianmodel
    * ampl: 0.15
    * pos: 0.0
    * width: 1.1
"""
    baseline_data = gaussianmodel().f(x, 0.15, 0.0, 1.1, normalized=False)
    y = 2.4 / (1.0 + (x / 0.12) ** 2) + baseline_data
    dataset = scp.NDDataset(y, coordset=[coord], mask=mask)
    opt = scp.Optimize(
        usermodels={"customprofile": profile},
        method="least_squares",
        amplitude_mode="height",
        autobase=False,
        autoampl=False,
        dry=False,
    )
    opt.script = baseline + peak if baseline_first else peak + baseline
    opt.fit(dataset)

    assert opt.result.diagnostics["success"]
    assert_allclose(opt.fp["ampl_peak"], 2.4, rtol=1e-8)
    fitted = opt.result.fitted
    assert_array_equal(fitted.mask.squeeze(), mask)
    assert_allclose(fitted.data.squeeze()[~mask], y[~mask], atol=1e-8)
    assert len(calls) > 2
    for axis in calls:
        assert_array_equal(axis, x)


def test_ampbas_preserves_ordered_sums_for_near_singular_model():
    x = np.arange(129, dtype=np.float64) / 64 - 1
    observed = 0.25 + 3 * x + 0.75 * x**2
    model = 1 + 2 * x + 2.0**-20 * x**2
    expected = (791731.3965183753, -1583459.7911025146, -791731.1489361703)

    assert_array_equal(scp.Optimize._ampbas(x, observed, model), expected)


def test_ampbas_keeps_warning_and_zero_coefficients_for_singular_model():
    x = np.linspace(-1.0, 1.0, 129)
    observed = 0.25 + 3 * x + 0.75 * x**2

    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        assert_array_equal(
            scp.Optimize._ampbas(x, observed, np.zeros_like(x)), (0, 0, 0)
        )


def test_autobase_residual_checks_scaling_of_cancelled_components(monkeypatch):
    x = np.linspace(-1.0, 1.0, 9)
    largest = np.finfo(float).max
    opt = scp.Optimize(autobase=True)
    profiles = {
        "large": np.full_like(x, largest),
        "opposite": np.full_like(x, -largest),
        "peak": x**2,
    }
    opt.script = "\n".join(
        f"MODEL: {label}\nshape: gaussianmodel\n* ampl: 1\n* pos: 0\n* width: 1"
        for label in profiles
    )
    monkeypatch.setattr(
        opt, "_evaluate_component", lambda _x, label, _spec: profiles[label]
    )

    with (
        np.errstate(over="raise"),
        pytest.raises(FloatingPointError, match="overflow encountered in multiply"),
    ):
        opt._get_modelsum(opt._model_spec, 3 * x**2, x, slice(None), x)


@pytest.mark.parametrize("over", ["warn", "raise"])
def test_streaming_residual_reports_reduction_overflow_after_all_profiles(
    monkeypatch, over
):
    x = np.arange(4, dtype=float)
    largest = np.finfo(float).max
    profiles = {
        "first": [largest, largest, 0, 0],
        "second": [largest, largest, 0, 0],
        "third": [0, 0, largest, largest],
        "fourth": [0, 0, largest, largest],
    }
    calls = []
    opt = scp.Optimize(autobase=False, usermodels={})
    opt.script = "\n".join(
        f"MODEL: {label}\nshape: gaussianmodel\n* ampl: 1\n* pos: 0\n* width: 1"
        for label in profiles
    )

    def evaluate(_x, label, _spec):
        calls.append(label)
        return profiles[label]

    monkeypatch.setattr(opt, "_evaluate_component", evaluate)
    message = "overflow encountered in reduce"
    with np.errstate(over=over):
        if over == "warn":
            with pytest.warns(RuntimeWarning, match=message) as recorded:
                total = opt._get_modelsum(opt._model_spec, x, x, slice(None), x)
            assert len(recorded) == 1
            assert np.isposinf(total).all()
        else:
            with pytest.raises(FloatingPointError, match=message):
                opt._get_modelsum(opt._model_spec, x, x, slice(None), x)

    assert calls == list(profiles)
