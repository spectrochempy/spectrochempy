"""Private preparation of simple Python functions for canonical 1D fitting."""

import inspect
import re
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np

from spectrochempy.analysis.curvefitting._models import usermodel
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec


def _callable_model_spec(
    function, parameters=None, *, label="model", common_params=None
):
    """Prepare a simple lowercase positional model for the canonical 1D path."""
    if not callable(function):
        raise TypeError("function must be callable")
    signature = list(inspect.signature(function).parameters.values())
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if not signature or any(
        parameter.kind not in positional for parameter in signature
    ):
        raise ValueError("A callable model requires an axis and positional parameters")
    names = [parameter.name for parameter in signature[1:]]
    if any(re.fullmatch(r"[a-z][a-z0-9]*", name) is None for name in names):
        raise ValueError("Parameter names must use lowercase letters and digits")
    if not isinstance(label, str) or re.fullmatch(r"[a-z][a-z0-9_]*", label) is None:
        raise ValueError("label must be a lowercase model identifier")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of _ParamSpec definitions")
    definitions = dict(parameters or {})
    unknown = set(definitions) - set(names)
    if unknown:
        raise ValueError(f"Unknown callable parameters: {sorted(unknown)}")
    prepared = {}
    for parameter in signature[1:]:
        name = parameter.name
        if name in definitions:
            definition = definitions[name]
            if not isinstance(definition, _ParamSpec) or definition.name != name:
                raise ValueError(f"Parameter {name!r} requires a matching _ParamSpec")
        else:
            value = parameter.default
            if value is inspect.Parameter.empty:
                raise ValueError(f"Parameter {name!r} requires an initial definition")
            if not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(
                    f"Parameter {name!r} needs a finite real scalar default"
                )
            definition = _ParamSpec(name, value)
        prepared[name] = definition
    if common_params is not None and not isinstance(common_params, Mapping):
        raise TypeError("common_params must be a mapping of _ParamSpec definitions")
    common = dict(common_params or {})
    for name, definition in common.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z]+[0-9]*", name) is None
            or not isinstance(definition, _ParamSpec)
            or definition.name != name
        ):
            raise ValueError(
                "COMMON parameters require matching named _ParamSpec definitions"
            )
    model = usermodel()
    model.f = function
    model.args = names
    spec = _FitModelSpec(
        components=[_ComponentSpec(label, "pythonmodel", prepared)],
        common_params=common,
    )
    return spec, {"pythonmodel": model}


def _callable_models_spec(functions, parameters=None, *, common_params=None):
    """Combine named callable definitions with one shared COMMON parameter block."""
    if not isinstance(functions, Mapping):
        raise TypeError("functions must be a mapping of component labels to callables")
    if not functions:
        raise ValueError("At least one callable component is required")
    if parameters is not None and not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of component definitions")
    definitions = dict(parameters or {})
    unknown = [label for label in definitions if label not in functions]
    if unknown:
        raise ValueError(f"Unknown component parameter definitions: {unknown}")
    components = []
    registry = {}
    common = {}
    for index, (label, function) in enumerate(functions.items()):
        spec, model = _callable_model_spec(
            function, definitions.get(label), label=label, common_params=common_params
        )
        component = deepcopy(spec.components[0])
        component.model_name = "pythonmodel" if index == 0 else f"pythonmodel{index}"
        components.append(component)
        registry[component.model_name] = model["pythonmodel"]
        if index == 0:
            common = spec.common_params
    return _FitModelSpec(components=components, common_params=common), registry
