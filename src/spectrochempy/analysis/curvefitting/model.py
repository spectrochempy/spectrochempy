"""Composable Python declarations for the canonical one-dimensional fitting path."""

import ast
import inspect
import operator
import re
from collections.abc import MutableMapping
from copy import deepcopy
from numbers import Real

import numpy as np

from spectrochempy.analysis.curvefitting._callable_spec import _callable_models_spec
from spectrochempy.analysis.curvefitting._models import gaussianmodel
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model

__all__ = ["Model", "Gaussian", "Linear", "Parameter", "Parameters"]

_UNSET = object()
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_OPERATORS = {
    ast.Add: ("+", operator.add),
    ast.Sub: ("-", operator.sub),
    ast.Mult: ("*", operator.mul),
    ast.Div: ("/", operator.truediv),
    ast.Pow: ("**", operator.pow),
}


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar magnitude")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class Parameter:
    """
    Initial scalar configuration, separate from fitted values and uncertainties.

    Parameters
    ----------
    value : float, optional
        Initial physical magnitude.
    min, max : float or None, optional
        Inclusive physical bounds. None means unbounded.
    vary : bool, optional
        Whether the parameter is estimated. An expression implies False.
    expr : str or None, optional
        Arithmetic expression using full parameter names.
    """

    def __init__(self, value=0.0, *, min=None, max=None, vary=True, expr=None):
        self.value, self.min, self.max = value, min, max
        self.vary, self.expr = vary, expr
        if expr is not None:
            self.vary = False
        self._validate()

    def _validate(self):
        self.value = _real(self.value, "value")
        self.min = None if self.min is None else _real(self.min, "min")
        self.max = None if self.max is None else _real(self.max, "max")
        if not isinstance(self.vary, (bool, np.bool_)):
            raise ValueError("vary must be boolean")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(
                "min must be strictly smaller than max; use vary=False to fix"
            )
        if self.expr is not None:
            if not isinstance(self.expr, str) or not self.expr.strip():
                raise ValueError("expr must be a nonempty expression or None")
            if self.vary or self.min is not None or self.max is not None:
                raise ValueError("Expression parameters cannot vary or have bounds")
        elif (self.min is not None and self.value < self.min) or (
            self.max is not None and self.value > self.max
        ):
            raise ValueError("Initial value lies outside its bounds")

    def set(
        self,
        *,
        value=_UNSET,
        min=_UNSET,
        max=_UNSET,
        vary=_UNSET,
        expr=_UNSET,
        scale=_UNSET,
    ):
        """
        Update initial configuration atomically and return this parameter.

        ``expr=None`` removes a link and makes the parameter free unless vary
        is supplied. An expression replaces independent bounds on the target.
        ``scale='log'`` is not implemented by the 1D backend.
        """
        if scale is not _UNSET and scale != "lin":
            raise NotImplementedError(
                "Log scale requires a separate 1D backend evolution"
            )
        candidate = deepcopy(self)
        for name, supplied in (
            ("value", value),
            ("min", min),
            ("max", max),
            ("vary", vary),
            ("expr", expr),
        ):
            if supplied is not _UNSET:
                setattr(candidate, name, supplied)
        if expr is not _UNSET and vary is _UNSET:
            candidate.vary = expr is None
        if expr is not _UNSET and expr is not None:
            if min is _UNSET:
                candidate.min = None
            if max is _UNSET:
                candidate.max = None
        candidate._validate()
        self.__dict__.update(candidate.__dict__)
        return self


class Parameters(MutableMapping):
    """
    Owned mapping of named initial Parameter configurations.

    Assignment copies a Parameter; reusing an object never creates a tie.
    Use ``expr`` to declare a relationship explicitly. Values are numerical
    magnitudes; unit conversion and logarithmic optimization are not provided
    by this first 1D facade.
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str) or _NAME.fullmatch(key) is None:
            raise ValueError("Parameter names must be lowercase Python identifiers")
        if not isinstance(value, Parameter):
            raise TypeError("Parameters entries must be Parameter objects")
        owned = deepcopy(value)
        owned._validate()
        self._data[key] = owned

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def add(self, name, value=0.0, **options):
        """Add an independent or auxiliary parameter, rejecting duplicate names."""
        if name in self:
            raise ValueError(f"Duplicate parameter {name!r}")
        scale = options.pop("scale", "lin")
        if scale != "lin":
            raise NotImplementedError(
                "Log scale requires a separate 1D backend evolution"
            )
        self[name] = Parameter(value, **options)
        return self[name]

    def copy(self):
        """Return an independent configuration copy."""
        return deepcopy(self)


class Model:
    """
    Wrap a scalar 1D function for Optimize and additive composition.

    Parameters
    ----------
    function : callable
        Receives a numerical coordinate vector followed by scalar positional
        parameters. Return a real vector on that grid, without normalization
        by Optimize. Parameter names must be lowercase Python identifiers;
        keyword-only and variadic arguments are rejected. Finite scalar defaults
        supply initial values; required parameters need make_params overrides.
    prefix : str, optional
        Empty, or a lowercase identifier ending in an underscore. Public names
        are prefix + function argument name. Each component needs a unique prefix.
    """

    def __init__(self, function, *, prefix=""):
        if not callable(function):
            raise TypeError("function must be callable")
        if not isinstance(prefix, str) or (
            prefix and (_NAME.fullmatch(prefix) is None or not prefix.endswith("_"))
        ):
            raise ValueError(
                "prefix must be empty or a lowercase identifier ending in '_'"
            )
        signature = list(inspect.signature(function).parameters.values())
        allowed = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        if len(signature) < 2 or any(p.kind not in allowed for p in signature):
            raise ValueError(
                "A Model requires an axis and scalar positional parameters"
            )
        if any(_NAME.fullmatch(p.name) is None for p in signature[1:]):
            raise ValueError(
                "Function parameter names must be lowercase Python identifiers"
            )
        defaults = {}
        for p in signature[1:]:
            defaults[p.name] = (
                None if p.default is inspect.Parameter.empty else Parameter(p.default)
            )
        self._parts = [(function, prefix, defaults)]

    def __add__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        result = deepcopy(self)
        result._parts += deepcopy(other._parts)
        result._names()
        return result

    def _names(self):
        prefixes = [prefix for _, prefix, _ in self._parts]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("Component prefixes must be unique; no implicit sharing")
        names = [
            prefix + name for _, prefix, defaults in self._parts for name in defaults
        ]
        if len(names) != len(set(names)):
            raise ValueError("Component parameter names collide")
        return names

    def make_params(self, **values):
        """Create independent parameters; overrides accept scalars or Parameter objects."""
        self._names()
        unknown = set(values) - set(self._names())
        if unknown:
            raise ValueError(f"Unknown model parameters: {sorted(unknown)}")
        params = Parameters()
        for _, prefix, defaults in self._parts:
            for local, default in defaults.items():
                name = prefix + local
                definition = values.get(name, default)
                if definition is None:
                    raise ValueError(f"Parameter {name!r} requires an initial value")
                if isinstance(definition, Parameter):
                    params[name] = definition
                else:
                    configured = (
                        deepcopy(default) if default is not None else Parameter()
                    )
                    params[name] = configured.set(value=definition)
        return params


def _gaussian(x, amplitude=1.0, center=0.0, width=1.0):
    if width <= 0:
        raise ValueError("Gaussian width must be strictly positive")
    return gaussianmodel().f(x, amplitude, center, width)


class Gaussian(Model):
    """
    Gaussian with the exact existing gaussianmodel normalization.

    ``center`` and FWHM ``width`` use coordinate magnitudes. ``amplitude`` is
    the existing ``ampl``: the normalized density is multiplied by abs(dx),
    so its sample sum approximates amplitude on an extended uniform grid.
    It is neither peak height nor a grid-independent integrated area.
    The sigma conversion remains width / 2.3548. Width must remain positive.
    """

    def __init__(self, *, prefix=""):
        super().__init__(_gaussian, prefix=prefix)
        self._parts[0][2]["width"].min = 1e-12


def _linear(x, slope=0.0, intercept=0.0):
    return slope * x + intercept


class Linear(Model):
    """
    Additive slope*x + intercept on the absolute coordinate magnitudes.

    Unlike polynomialbaseline, the coordinates are not centered. This is an
    explicitly fitted component, without automatic intensity rescaling or
    baseline subtraction. Intercept uses intensity magnitudes, slope uses
    intensity per coordinate magnitude.
    """

    def __init__(self, *, prefix=""):
        super().__init__(_linear, prefix=prefix)


def _expression_plan(params, names):
    trees, dependencies = {}, {}
    for name, param in params.items():
        param._validate()
        if param.expr is None:
            continue
        if len(param.expr) > 4096:
            raise ValueError(f"Expression for {name!r} is too long")
        try:
            tree = ast.parse(param.expr, mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"Invalid expression for {name!r}") from error
        trees[name] = tree
        dependencies[name] = set()

        def render(node, owner=name):
            if isinstance(node, ast.Name):
                if node.id not in params:
                    raise ValueError(f"Unknown expression target {node.id!r}")
                dependencies[owner].add(node.id)
                return f"({names[node.id]})"
            if isinstance(node, ast.Constant):
                return f"({_real(node.value, 'Expression constant')!r})"
            if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
                return f"({render(node.left)} {_OPERATORS[type(node.op)][0]} {render(node.right)})"
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                return f"({'+' if isinstance(node.op, ast.UAdd) else '-'}{render(node.operand)})"
            raise ValueError(
                "Expressions accept names, numbers, +, -, *, /, ** and parentheses only"
            )

        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError(f"Expression for {name!r} is too complex")
        trees[name] = render(tree)
    done, active = set(), set()

    def visit(name):
        if name in active:
            raise ValueError(f"Cyclic parameter expression involving {name!r}")
        if name not in done:
            active.add(name)
            for dependency in dependencies.get(name, ()):
                visit(dependency)
            active.remove(name)
            done.add(name)

    for name in params:
        visit(name)
    return trees, dependencies


def _prepare_public_model(model, params):
    model = deepcopy(model)
    if params is None:
        params = model.make_params()
    if not isinstance(params, Parameters):
        raise TypeError("params must be a Parameters object")
    params = params.copy()
    missing = set(model._names()) - set(params)
    if missing:
        raise ValueError(f"Missing model parameters: {sorted(missing)}")
    names = {}
    for index, name in enumerate(sorted(params)):
        digits = []
        number = index
        for _ in range(6):
            digits.append(chr(ord("a") + number % 26))
            number //= 26
        names[name] = "param" + "".join(reversed(digits))
    expressions, dependencies = _expression_plan(params, names)
    used = set(model._names())
    pending = list(used)
    while pending:
        for name in dependencies.get(pending.pop(), set()) - used:
            used.add(name)
            pending.append(name)
    if set(params) - used:
        raise ValueError(f"Unused auxiliary parameters: {sorted(set(params) - used)}")
    common = {
        names[name]: _ParamSpec(
            names[name], p.value, p.vary, (p.min, p.max), expressions.get(name)
        )
        for name, p in params.items()
    }
    functions, definitions = {}, {}
    for index, (function, prefix, defaults) in enumerate(model._parts):
        local_names = list(defaults)

        def adapter(x, *values, function=function):
            values = np.asarray(function(x, *values))
            if (
                values.shape != x.shape
                or not np.issubdtype(values.dtype, np.number)
                or not np.isrealobj(values)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    "Model functions must return a finite real vector on the coordinate grid"
                )
            return values

        arguments = [f"a{i}" for i in range(len(local_names))]
        adapter.__signature__ = inspect.Signature(
            [
                inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for n in ["x", *arguments]
            ]
        )
        label = f"component{index}"
        functions[label] = adapter
        definitions[label] = {
            argument: _ParamSpec(argument, vary=False, reference=names[prefix + name])
            for argument, name in zip(arguments, local_names, strict=True)
        }
    spec, registry = _callable_models_spec(functions, definitions, common_params=common)
    try:
        resolved = prepare_model(spec)
        for name, p in resolved.common_params.items():
            _real(p.value, f"Initial expression {name}")
        for function, prefix, _ in model._parts:
            if function is _gaussian:
                width = params[prefix + "width"]
                if resolved.common_params[names[prefix + "width"]].value <= 0:
                    raise ValueError("Gaussian width must be strictly positive")
                if width.vary and (width.min is None or width.min <= 0):
                    raise ValueError("A varying Gaussian width requires min > 0")
    except (ArithmeticError, TypeError) as error:
        raise ValueError("Invalid initial parameter expression") from error
    return model, params, names, spec, registry
