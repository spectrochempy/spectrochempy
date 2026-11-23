"""Parity of private Python model definitions and the existing fitting DSL."""

from copy import deepcopy
from importlib import import_module

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._models import gaussianmodel
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
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
shape: gaussianmodel
    $ ampl: 0.5, 0.0, 5.0
    * pos: 1.2
    > width: gwidth
"""


def _python_spec():
    return _FitModelSpec(
        common_params={"gwidth": _ParamSpec("gwidth", 0.7, bounds=(0.1, 2.0))},
        components=[
            _ComponentSpec(
                label,
                "gaussianmodel",
                {
                    "ampl": _ParamSpec("ampl", amplitude, bounds=(0.0, 5.0)),
                    "pos": _ParamSpec("pos", position, vary=False),
                    "width": _ParamSpec("width", vary=False, reference="gwidth"),
                },
            )
            for label, amplitude, position in [("left", 1.0, -1.0), ("right", 0.5, 1.2)]
        ],
    )


@pytest.mark.parametrize("mode", ["height", "area"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_python_and_script_definitions_fit_identically(mode, descending, masked):
    x = np.linspace(-4.0, 4.0, 129)
    if descending:
        x = x[::-1]
    observed = np.zeros_like(x)
    for amplitude, position in [(2.0, -1.0), (1.2, 1.2)]:
        curve = gaussianmodel().f(x, amplitude, position, 0.6)
        if mode == "height":
            curve = amplitude * curve / curve.max()
        observed += curve
    mask = np.zeros_like(x, dtype=bool)
    if masked:
        mask[[5, 50, 120]] = True
    data = scp.NDDataset(
        observed, coordset=[scp.Coord(x, units="cm^-1")], mask=mask, units="absorbance"
    )
    options = {
        "amplitude_mode": mode,
        "autobase": False,
        "autoampl": False,
        "method": "least_squares",
    }
    spec = _python_spec()
    before = deepcopy(spec)
    python_fit = scp.Optimize._from_model_spec(spec, **options).fit(data)
    script_fit = scp.Optimize(script=SCRIPT, **options).fit(data)

    np.testing.assert_array_equal(
        python_fit._model_spec.extract_varying_values(),
        script_fit._model_spec.extract_varying_values(),
    )
    for output in ("fitted", "components", "residuals"):
        actual = getattr(python_fit.result, output)
        expected = getattr(script_fit.result, output)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        assert actual.units == expected.units
    assert python_fit._model_spec.count_varying() == 3
    resolved = prepare_model(python_fit._model_spec)
    width = resolved.common_params["gwidth"].value
    assert width == pytest.approx(0.6, abs=1e-5)
    for component in resolved.components:
        assert component.params["width"].value == width
        assert component.params["pos"].vary is False
    assert spec.common_params == before.common_params
    assert spec.components == before.components


def test_structured_values_are_owned_without_parsing_rendered_text(monkeypatch):
    module = import_module("spectrochempy.analysis.curvefitting.optimize")
    parser = module._parse_script_to_modelspec
    calls = []

    def only_empty_script(script, *args, **kwargs):
        calls.append(script)
        assert script == ""
        return parser(script, *args, **kwargs)

    monkeypatch.setattr(module, "_parse_script_to_modelspec", only_empty_script)
    spec = _python_spec()
    value = 0.712345678901234
    spec.common_params["gwidth"].value = value
    estimator = scp.Optimize._from_model_spec(spec)
    assert estimator._model_spec.common_params["gwidth"].value == value
    assert estimator.fp["gwidth"] == value
    assert "0.7123" in estimator.script
    spec.common_params["gwidth"].value = 1.5
    assert estimator.fp["gwidth"] == value
    assert estimator._model_spec.common_params["gwidth"].value == value
    assert all(script == "" for script in calls)


@pytest.mark.parametrize(
    "options",
    [
        {"script": SCRIPT},
        {"peaks": []},
        {"shared_parameters": {"width": 1.0}},
        {"warm_start": True},
    ],
)
def test_structured_definition_rejects_competing_model_configuration(options):
    with pytest.raises(ValueError, match="cannot be combined"):
        scp.Optimize._from_model_spec(_python_spec(), **options)


def test_structured_definition_requires_canonical_spec():
    with pytest.raises(TypeError, match="_FitModelSpec"):
        scp.Optimize._from_model_spec({})


def test_warm_start_is_rejected_before_loading_saved_configuration():
    class NoConstruction(scp.Optimize):
        def __init__(self, **kwargs):
            raise AssertionError("Saved configuration must not be loaded")

    with pytest.raises(ValueError, match="warm_start=True"):
        NoConstruction._from_model_spec(_python_spec(), warm_start=True)


def test_constraint_name_validation_uses_structured_definition():
    constraint = {
        "type": "max_connections",
        "limit": 2,
        "parameters": ["ampl_left", "ampl_right"],
    }
    estimator = scp.Optimize._from_model_spec(_python_spec(), constraints=constraint)
    assert estimator.validate_constraints() == []
    with pytest.raises(ValueError, match="unknown"):
        scp.Optimize._from_model_spec(
            _python_spec(), constraints={**constraint, "parameters": ["unknown"]}
        )


def test_python_function_fits_from_structured_definition_without_model_script():
    def decay(time, ampl, rate, offset):
        return ampl * np.exp(-rate * time) + offset

    spec = _FitModelSpec(
        components=[
            _ComponentSpec(
                "signal",
                "decay",
                {
                    "ampl": _ParamSpec("ampl", 2.0, bounds=(0.0, 5.0)),
                    "rate": _ParamSpec("rate", 0.5, bounds=(0.1, 2.0)),
                    "offset": _ParamSpec("offset", 0.2, vary=False),
                },
            )
        ]
    )
    time = np.linspace(0.0, 5.0, 101)
    data = scp.NDDataset(decay(time, 3.0, 0.8, 0.2), coordset=[scp.Coord(time)])
    estimator = scp.Optimize._from_model_spec(
        spec,
        usermodels={"decay": decay},
        amplitude_mode="area",
        autobase=False,
        autoampl=False,
        method="least_squares",
    ).fit(data)
    assert estimator.fp["ampl_signal"] == pytest.approx(3.0, abs=1e-6)
    assert estimator.fp["rate_signal"] == pytest.approx(0.8, abs=1e-6)
    assert estimator.fp["offset_signal"] == 0.2
    assert estimator.result.diagnostics["success"]
