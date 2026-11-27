"""
Private, fit-local preparation of scalar 1D parameters and expressions.

The qualified language is numeric constants, COMMON names, parentheses and
unary/binary +, -, *, / and powers with preserved signed-literal precedence.
Other powers, numpy calls/constants, component-key references
and other legacy Python syntax retain prepare_model's historical evaluator.
In particular, textual expansion of forward references is not equivalent to
evaluating their values first: this adapter preserves historical precedence.
ND declarations retain their existing preparation and numeric scale semantics.
"""

import ast
import copy
import re
from dataclasses import dataclass
from dataclasses import field
from graphlib import CycleError
from graphlib import TopologicalSorter
from numbers import Real
from types import CodeType

from spectrochempy.analysis.curvefitting._modelspec import _RE_KEYWORD
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model
from spectrochempy.analysis.curvefitting._parameter_space import _prepared_parameters


@dataclass(frozen=True)
class _ExpressionStep:
    name: str
    symbol: str
    source: str
    code: CodeType = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self, "code", compile(self.source, "<fit expression>", "eval")
        )

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (self.name, self.symbol, self.source)


@dataclass(frozen=True)
class _ParameterPlan:
    """Immutable declaration snapshot; runtime magnitudes belong to the spec."""

    parameters: tuple
    sources: tuple
    components: tuple
    expressions: tuple
    dependencies: tuple
    fallback: bool = False
    coordinate_transform: str = "legacy_bounds"
    normalization_scale: float = 1.0

    @property
    def free_names(self):
        return tuple(p.name for p in self.parameters if not p.fixed)

    def resolve(self, spec):
        """Resolve current physical values without retaining mutable fit state."""
        if self.fallback:
            return prepare_model(spec)
        owners = {component.label: component.params for component in spec.components}
        values, namespace = {}, {}
        for key, symbol, identifier, *_ in self.sources:
            parameter = (
                spec.common_params[identifier[1]]
                if identifier[0] == "common"
                else owners[identifier[1]][identifier[2]]
            )
            if parameter.reference is None:
                values[key] = parameter.value
                namespace[symbol] = float(parameter.value)
        for step in self.expressions:
            value = float(eval(step.code, {"__builtins__": {}}, namespace))  # noqa: S307
            values[step.name] = namespace[step.symbol] = value
        common, components = {}, {label: {} for label, _ in self.components}
        for key, _, identifier, name, vary, bounds, linked in self.sources:
            parameter = _ParamSpec(name, values[key], vary and not linked, bounds)
            if identifier[0] == "common":
                common[identifier[1]] = parameter
            else:
                components[identifier[1]][identifier[2]] = parameter
        return _FitModelSpec(
            [
                _ComponentSpec(label, model, components[label])
                for label, model in self.components
            ],
            common,
        )


def _arithmetic_tree(expression):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Name,
        ast.Load,
        ast.Constant,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            return None
    return tree


def _preserves_signed_literals(source, tree):
    """Keep textual negative-base semantics when replacing values by symbols."""
    if "**" not in source:
        return True
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for name in names:

        class Negate(ast.NodeTransformer):
            def visit_Name(self, node, name=name):
                return ast.UnaryOp(ast.USub(), node) if node.id == name else node

        expected = Negate().visit(copy.deepcopy(tree))
        actual = ast.parse(re.sub(rf"\b{name}\b", f"-{name}", source), mode="eval")
        if ast.dump(expected) != ast.dump(actual):
            return False
    return True


def _compile_parameter_plan(spec):
    """Compile once per fit; never broaden the public expression language."""
    parameters = tuple(
        copy.deepcopy(parameter) for parameter in _prepared_parameters(spec)
    )
    declarations = [
        (name, ("common", name), parameter)
        for name, parameter in spec.common_params.items()
    ]
    declarations.extend(
        (f"{name}_{component.label}", ("components", component.label, name), parameter)
        for component in spec.components
        for name, parameter in component.params.items()
    )
    components = tuple((c.label, c.model_name) for c in spec.components)
    fallback = _ParameterPlan(parameters, (), components, (), (), True)
    keys = [key for key, _, _ in declarations]
    if (
        spec.expvars
        or spec.expnumber != 1
        or spec.constraints is not None
        or len(set(keys)) != len(keys)
        or len({label for label, _ in components}) != len(components)
    ):
        return fallback
    for _, _, parameter in declarations:
        if not isinstance(parameter.value, Real) or any(
            bound is not None and not isinstance(bound, Real)
            for bound in parameter.bounds
        ):
            return fallback
    references = {
        key: str(p.reference) for key, _, p in declarations if p.reference is not None
    }
    dependencies = {}
    for key, expression in references.items():
        tree = _arithmetic_tree(expression)
        if tree is None:
            return fallback
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        if any(_RE_KEYWORD.fullmatch(name) is None for name in names):
            return fallback
        if names - set(keys):
            return fallback
        if any(a != b and a in b for a in names for b in names):
            return fallback
        dependencies[key] = tuple(sorted(names))
    try:
        tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError(f"Cyclic parameter expression: {error.args[1]}") from error
    symbols = {key: f"V{index}" for index, key in enumerate(keys)}
    lookup = {key: references.get(key, symbols[key]) for key in keys}
    steps = []
    for key, expression in references.items():
        current = expression
        for _ in range(len(declarations) + 1):
            resolved = current
            for name in _RE_KEYWORD.findall(current):
                if name not in lookup:
                    return fallback
                resolved = resolved.replace(name, lookup[name])
            if resolved == current:
                break
            current = resolved
            if len(current) > 65536:
                return fallback
        else:
            return fallback
        tree = _arithmetic_tree(current)
        if tree is None:
            return fallback
        if any(
            node.id not in symbols.values()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        ):
            return fallback
        try:
            preserves_signs = _preserves_signed_literals(current, tree)
        except RecursionError:
            # Qualification is optional: deep legacy expressions remain valid
            # when copying/visiting their AST exceeds the interpreter limit.
            return fallback
        if not preserves_signs:
            return fallback
        steps.append(_ExpressionStep(key, symbols[key], current))
        lookup[key] = symbols[key]
    sources = tuple(
        (
            key,
            symbols[key],
            identifier,
            p.name,
            p.vary,
            tuple(p.bounds),
            p.reference is not None,
        )
        for key, identifier, p in declarations
    )
    return _ParameterPlan(
        parameters, sources, components, tuple(steps), tuple(dependencies.items())
    )
