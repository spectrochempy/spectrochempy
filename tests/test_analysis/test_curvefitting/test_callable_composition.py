"""Private callable composition with explicit shared parameter definitions."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._models import gaussianmodel
from spectrochempy.analysis.curvefitting._models import lorentzianmodel
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model

SCRIPT = """
COMMON:
    $ gwidth: 0.7, 0.1, 2.0
MODEL: left
shape: gaussianmodel
    $ ampl: 1.0, 0.0, 5.0
    * pos: -1.0
    > width: gwidth
MODEL: right
shape: lorentzianmodel
    $ ampl: 0.5, 0.0, 5.0
    * pos: 1.2
    > width: gwidth
"""


def _gaussian(x, ampl=1.0, pos=-1.0, width=0.7):
    return gaussianmodel().f(x, ampl, pos, width)


def _lorentzian(x, ampl=0.5, pos=1.2, width=0.7):
    return lorentzianmodel().f(x, ampl, pos, width)


@pytest.mark.parametrize("mode", ["area", "height"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_two_functions_with_shared_width_match_the_dsl(mode, descending, masked):
    x = np.linspace(-4.0, 4.0, 129)
    if descending:
        x = x[::-1]
    observed = np.zeros_like(x)
    for function, ampl, pos in [(_gaussian, 2.0, -1.0), (_lorentzian, 1.2, 1.2)]:
        curve = function(x, ampl, pos, 0.6)
        if mode == "height":
            curve = ampl * curve / curve.max()
        observed += curve
    mask = np.zeros_like(x, dtype=bool)
    if masked:
        mask[[5, 50, 120]] = True
    data = scp.NDDataset(observed, coordset=[scp.Coord(x)], mask=mask)
    shared_width = _ParamSpec("width", vary=False, reference="gwidth")
    parameters = {
        label: {
            "ampl": _ParamSpec("ampl", ampl, bounds=(0.0, 5.0)),
            "pos": _ParamSpec("pos", pos, vary=False),
            "width": shared_width,
        }
        for label, ampl, pos in [("left", 1.0, -1.0), ("right", 0.5, 1.2)]
    }
    common = _ParamSpec("gwidth", 0.7, bounds=(0.1, 2.0))
    evaluations = []

    def left(x, ampl=1.0, pos=-1.0, width=0.7):
        evaluations.append(("left", width))
        return _gaussian(x, ampl, pos, width)

    def right(x, ampl=0.5, pos=1.2, width=0.7):
        evaluations.append(("right", width))
        return _lorentzian(x, ampl, pos, width)

    options = {
        "amplitude_mode": mode,
        "autobase": False,
        "autoampl": False,
        "method": "least_squares",
    }
    python_fit = scp.Optimize._from_callables(
        {"left": left, "right": right},
        parameters,
        common_params={"gwidth": common},
        **options,
    ).fit(data)
    script_fit = scp.Optimize(script=SCRIPT, **options).fit(data)
    np.testing.assert_array_equal(
        python_fit._model_spec.extract_varying_values(),
        script_fit._model_spec.extract_varying_values(),
    )
    for name in ("fitted", "components", "residuals"):
        actual = getattr(python_fit.result, name)
        expected = getattr(script_fit.result, name)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.mask, expected.mask)
    assert python_fit.result.n_varying_parameters == 3
    assert python_fit.result.degrees_of_freedom == int((~mask).sum()) - 3
    resolved = prepare_model(python_fit._model_spec)
    assert resolved.common_params["gwidth"].value == pytest.approx(0.6, abs=1e-5)
    for component in resolved.components:
        assert component.params["width"].value == resolved.common_params["gwidth"].value
    assert common.value == 0.7
    assert shared_width.reference == "gwidth"
    assert evaluations and len(evaluations) % 2 == 0
    for first, second in zip(evaluations[::2], evaluations[1::2], strict=True):
        assert first[0] == "left" and second[0] == "right"
        assert first[1] == second[1]


def test_reusing_parameter_metadata_does_not_implicitly_tie_components():
    def linear(x, ampl=1.0):
        return ampl * x

    def quadratic(x, ampl=1.0):
        return ampl * x**2

    initial = _ParamSpec("ampl", 1.0, bounds=(0.0, 5.0))
    estimator = scp.Optimize._from_callables(
        {"gaussianmodel": linear, "second_part": quadratic},
        {"gaussianmodel": {"ampl": initial}, "second_part": {"ampl": initial}},
        method="least_squares",
    )
    components = estimator._model_spec.components
    assert components[0].params["ampl"] is not components[1].params["ampl"]
    assert [component.label for component in components] == [
        "gaussianmodel",
        "second_part",
    ]
    assert components[0].model_name != components[1].model_name
    x = np.linspace(1.0, 3.0, 81)
    estimator.fit(
        scp.NDDataset(linear(x, 2.0) + quadratic(x, 0.3), coordset=[scp.Coord(x)])
    )
    assert estimator.fp["ampl_gaussianmodel"] == pytest.approx(2.0, abs=1e-6)
    assert estimator.fp["ampl_second_part"] == pytest.approx(0.3, abs=1e-6)
    assert initial.value == 1.0


def test_component_order_and_single_function_compatibility():
    composed = scp.Optimize._from_callables({"right": _lorentzian, "left": _gaussian})
    assert [component.label for component in composed._model_spec.components] == [
        "right",
        "left",
    ]
    single = scp.Optimize._from_callable(_gaussian, label="left")
    one_component = scp.Optimize._from_callables({"left": _gaussian})
    assert single.fp.data == one_component.fp.data
    assert single._model_spec.components == one_component._model_spec.components


@pytest.mark.parametrize(
    "functions, parameters, error",
    [
        ([], None, TypeError),
        ({}, None, ValueError),
        ({"left": _gaussian}, [], TypeError),
        ({"left": _gaussian}, {"missing": {}}, ValueError),
        ({"left": _gaussian}, {"left": []}, TypeError),
        ({"Upper": _gaussian}, None, ValueError),
        ({"left": None}, None, TypeError),
    ],
)
def test_invalid_composition_is_rejected(functions, parameters, error):
    with pytest.raises(error):
        scp.Optimize._from_callables(functions, parameters)


@pytest.mark.parametrize(
    "options", [{"usermodels": {}}, {"script": ""}, {"peaks": []}, {"warm_start": True}]
)
def test_conflicting_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        scp.Optimize._from_callables({"left": _gaussian}, **options)
