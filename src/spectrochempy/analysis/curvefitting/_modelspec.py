# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Private structured model-definition object for the Optimize script DSL.

Provides a clean, structured in-memory representation of what the
Optimize script parser already knows, separating model topology,
parameter definitions, and parameter values.

This module is **private**.  It is not a public API.
"""

import copy
import re
import sys
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np

from spectrochempy.analysis.curvefitting._parameters import FitParameters

# Sentinel bound thresholds (same as FitParameters.__str__)
_STR_NEG_THRESH = -0.1 / sys.float_info.epsilon
_STR_POS_THRESH = +0.1 / sys.float_info.epsilon


def _unbound(val):
    """Convert sentinel bound values to ``None``."""
    if val is None:
        return None
    if val <= _STR_NEG_THRESH or val >= _STR_POS_THRESH:
        return None
    return val


def _bound_to_legacy_sentinel(val, *, lower):
    """Convert canonical ``None`` bounds to the historical FitParameters sentinel."""
    if val is None:
        return -1.0 / sys.float_info.epsilon if lower else +1.0 / sys.float_info.epsilon
    return val


# ======================================================================================
@dataclass
class _ParamSpec:
    """Parameter definition with value, bounds, and flags."""

    name: str
    value: float | list[float] = 0.0
    vary: bool = True
    bounds: tuple[float | None, float | None] = (None, None)
    reference: str | None = None  # name of referenced COMMON param, or None


# ======================================================================================
@dataclass
class _ComponentSpec:
    """A single model component with its parameters."""

    label: str
    model_name: str = ""
    params: dict[str, _ParamSpec] = field(default_factory=dict)


# ======================================================================================
class _FitModelSpec:
    """
    Structured model-definition object.

    Separates model topology (components), parameter definitions
    (names, vary, bounds, references), and parameter values into a
    clean representation.

    Unlike ``FitParameters``, which uses parallel dictionaries and
    implicit ``{param}_{label}`` key encoding, this object stores
    parameters as structured attributes of their owning component.
    """

    def __init__(
        self,
        components: list[_ComponentSpec] | None = None,
        common_params: dict[str, _ParamSpec] | None = None,
        constraints: Any = None,
        expvars: list[str] | None = None,
        expnumber: int = 1,
    ):
        self.components = list(components or [])
        self.common_params = dict(common_params or {})
        self.constraints = constraints  # reserved, not yet used
        self.expvars = list(expvars or [])
        self.expnumber = expnumber

    # ------------------------------------------------------------------
    @classmethod
    def from_fitparameters(cls, fp):
        """
        Build a ``_FitModelSpec`` from a parsed ``FitParameters`` instance.

        This is the primary way to construct a ``_FitModelSpec`` from
        user-provided script content.
        """
        common_params = {}

        # Extract COMMON parameters from fp.common
        # fp.common is keyed by raw parameter name (no model suffix)
        for raw_name, is_common in fp.common.items():
            if not is_common:
                continue
            if raw_name not in fp:
                continue

            is_ref = fp.reference.get(raw_name, False)
            if is_ref:
                common_params[raw_name] = _ParamSpec(
                    name=raw_name,
                    value=0.0,
                    vary=False,
                    bounds=(None, None),
                    reference=str(fp[raw_name]),
                )
            else:
                if raw_name in fp.expvars:
                    value = list(fp[raw_name])
                else:
                    value = float(fp[raw_name])
                common_params[raw_name] = _ParamSpec(
                    name=raw_name,
                    value=value,
                    vary=not fp.fixed.get(raw_name, False),
                    bounds=(
                        _unbound(fp.lob.get(raw_name)),
                        _unbound(fp.upb.get(raw_name)),
                    ),
                )

        # Extract model components
        components = []
        for model_label in fp.models:
            model_name = fp.model.get(model_label, "")
            model_params = {}

            for full_key in sorted(fp):
                # Each model param key is "{param}_{label}".
                # The first underscore separates name from model label.
                keyspl = full_key.split("_")
                if len(keyspl) < 2:
                    continue  # COMMON param has no suffix
                param_raw = keyspl[0]
                param_model = "_".join(keyspl[1:])
                if param_model != model_label:
                    continue

                is_ref = fp.reference.get(full_key, False)
                if is_ref:
                    model_params[param_raw] = _ParamSpec(
                        name=param_raw,
                        value=0.0,
                        vary=False,
                        bounds=(None, None),
                        reference=str(fp[full_key]),
                    )
                else:
                    if param_raw in fp.expvars:
                        value = list(fp[full_key])
                    else:
                        value = float(fp[full_key])
                    model_params[param_raw] = _ParamSpec(
                        name=param_raw,
                        value=value,
                        vary=not fp.fixed.get(full_key, False),
                        bounds=(
                            _unbound(fp.lob.get(full_key)),
                            _unbound(fp.upb.get(full_key)),
                        ),
                    )

            components.append(
                _ComponentSpec(
                    label=model_label,
                    model_name=model_name,
                    params=model_params,
                )
            )

        return cls(
            components=components,
            common_params=common_params,
            expvars=list(fp.expvars),
            expnumber=fp.expnumber,
        )

    # ------------------------------------------------------------------
    def to_script(self) -> str:
        """
        Serialize back to a script string accepted by the parser.

        The output is semantically equivalent to what the parser
        expects.  Whitespace and formatting are not guaranteed
        identical to the original script, but the re-parsed result
        will match ``from_fitparameters`` semantically.
        """
        lines = []

        # COMMON section
        if self.common_params:
            lines.append("COMMON:")
            for name in sorted(self.common_params):
                ps = self.common_params[name]
                if ps.reference is not None:
                    lines.append(f"    > {name}:{ps.reference}")
                else:
                    prefix = "*" if not ps.vary else "$"
                    lob = _format_bound(ps.bounds[0])
                    upb = _format_bound(ps.bounds[1])
                    lines.append(f"    {prefix} {name}: {ps.value:.4f}, {lob}, {upb}")
            lines.append("")

        # Model components
        for comp in self.components:
            lines.append(f"MODEL: {comp.label}")
            if comp.model_name:
                lines.append(f"shape: {comp.model_name}")
            for name in sorted(comp.params):
                ps = comp.params[name]
                if ps.reference is not None:
                    lines.append(f"    > {name}:{ps.reference}")
                else:
                    prefix = "*" if not ps.vary else "$"
                    lob = _format_bound(ps.bounds[0])
                    upb = _format_bound(ps.bounds[1])
                    lines.append(f"    {prefix} {name}: {ps.value:.4f}, {lob}, {upb}")

        result = "\n".join(lines).strip()
        if result:
            result += "\n"
        return result

    # ------------------------------------------------------------------
    def to_fitparameters(self):
        """
        Build a legacy ``FitParameters`` compatibility view from this model spec.

        This keeps the parser target architecture canonical while preserving
        historical ``Optimize.fp`` and ``_validate_script_content()`` behavior.
        """
        fp = FitParameters()
        fp.expvars = list(self.expvars)
        fp.expnumber = self.expnumber

        for name, ps in self.common_params.items():
            fp.common[name] = True
            fp.reference[name] = ps.reference is not None
            if ps.reference is not None:
                fp[name] = ps.reference
                continue
            fp[name] = (
                ps.value,
                _bound_to_legacy_sentinel(ps.bounds[0], lower=True),
                _bound_to_legacy_sentinel(ps.bounds[1], lower=False),
                not ps.vary,
            )

        for comp in self.components:
            fp.models.append(comp.label)
            fp.model[comp.label] = comp.model_name
            for name, ps in comp.params.items():
                fp.common[name] = False
                key = f"{name}_{comp.label}"
                fp.reference[key] = ps.reference is not None
                if ps.reference is not None:
                    fp[key] = ps.reference
                    continue
                fp[key] = (
                    ps.value,
                    _bound_to_legacy_sentinel(ps.bounds[0], lower=True),
                    _bound_to_legacy_sentinel(ps.bounds[1], lower=False),
                    not ps.vary,
                )

        return fp

    # ------------------------------------------------------------------
    def _iter_varying(self):
        """
        Yield ``(sorted_key, param_spec)`` for varying parameters.

        Iteration order matches ``sorted(fp.keys())`` from
        ``FitParameters`` for consistent behavior.
        """
        items: list[tuple[str, _ParamSpec]] = []

        for name, ps in self.common_params.items():
            items.append((name, ps))

        for comp in self.components:
            for name, ps in comp.params.items():
                key = f"{name}_{comp.label}"
                items.append((key, ps))

        items.sort(key=lambda x: x[0])

        for key, ps in items:
            if not ps.vary:
                continue
            yield key, ps

    # ------------------------------------------------------------------
    def count_varying(self) -> int:
        """
        Count free parameters using the same rule as the optimizer.

        Returns
        -------
        int
            Number of varying parameters.
        """
        n_varying = 0
        for _key, ps in self._iter_varying():
            if ps.name in self.expvars:
                n_varying += self.expnumber
            else:
                n_varying += 1
        return n_varying

    # ------------------------------------------------------------------
    def extract_varying_values(self):
        """
        Return fitted varying-parameter values in optimizer order.

        Returns
        -------
        ndarray or None
            Frozen 1-D array of varying parameter values, or ``None``
            when the model has no varying parameters.
        """
        values: list[float] = []
        for _key, ps in self._iter_varying():
            if ps.name in self.expvars:
                values.extend(
                    np.asarray(ps.value, dtype=np.float64).reshape(-1).tolist(),
                )
            else:
                values.append(float(ps.value))

        array = np.array(values, dtype=np.float64)
        array.flags.writeable = False
        return array

    # ------------------------------------------------------------------
    def component_view(self, label: str):
        """
        Return a ``_ComponentParamsView`` for the component identified by *label*.

        Parameters
        ----------
        label : str
            Component label to look up.

        Returns
        -------
        _ComponentParamsView
            Lightweight view providing ``.model`` dict and ``__getitem__``
            compatible with :func:`getmodel`.
        """
        for comp in self.components:
            if comp.label == label:
                return _ComponentParamsView(comp, self.common_params)
        raise KeyError(f"Component '{label}' not found in spec")


def _format_bound(val: float | None) -> str:
    """Format a bound value for script output."""
    if val is None:
        return "none"
    # Use integer format if exact integer
    if val == int(val):
        return f"{int(val)}"
    return f"{val:.4f}"


# ------------------------------------------------------------------
def _build_flat_lookup(spec: _FitModelSpec) -> dict[str, Any]:
    """
    Build a flat name-to-value-or-expression lookup dict from a spec.

    For resolved parameters the dict holds the float value; for unresolved
    reference parameters it holds the reference expression string.
    """
    lookup = {}
    for name, ps in spec.common_params.items():
        lookup[name] = ps.reference if ps.reference is not None else float(ps.value)
    for comp in spec.components:
        for name, ps in comp.params.items():
            key = f"{name}_{comp.label}"
            lookup[key] = ps.reference if ps.reference is not None else float(ps.value)
    return lookup


_RE_KEYWORD = re.compile(r"\b([a-z]+[0-9]*)\b")


def _resolve_expression(expr: str, lookup: dict[str, Any]) -> float:
    """
    Resolve a reference expression against a name-to-value lookup.

    Iteratively substitutes recognised tokens (parameter names or numpy
    identifiers) until the expression stabilises, then evaluates it.
    """
    current = str(expr)
    while True:
        tokens = _RE_KEYWORD.findall(current)
        if not tokens:
            break
        resolved = current
        for kw in tokens:
            if kw in lookup:
                resolved = resolved.replace(kw, str(lookup[kw]))
            elif kw in np.__dict__:
                resolved = resolved.replace(kw, f"np.{kw}")
        if resolved == current:
            break
        current = resolved
    return float(eval(str(current)))  # noqa: S307


def prepare_model(spec: _FitModelSpec) -> _FitModelSpec:
    """
    Resolve all reference parameters in a ``_FitModelSpec``.

    Returns a new spec in which every parameter that was defined as a
    reference (``_ParamSpec.reference`` is not ``None``) has been replaced
    by its resolved numeric value and marked as fixed (``vary=False``).
    The original spec is **not** modified.

    Parameters
    ----------
    spec : _FitModelSpec
        The model spec to prepare.

    Returns
    -------
    _FitModelSpec
        A fully resolved copy of the input spec.
    """
    result = copy.deepcopy(spec)
    lookup = _build_flat_lookup(result)

    # Resolve COMMON references first so component references can
    # depend on them.
    for name, ps in result.common_params.items():
        if ps.reference is not None:
            value = _resolve_expression(ps.reference, lookup)
            ps.value = value
            ps.reference = None
            ps.vary = False
            lookup[name] = value

    # Resolve component references.
    for comp in result.components:
        for name, ps in comp.params.items():
            if ps.reference is not None:
                value = _resolve_expression(ps.reference, lookup)
                ps.value = value
                ps.reference = None
                ps.vary = False
                key = f"{name}_{comp.label}"
                lookup[key] = value

    return result


# ======================================================================================
class _ComponentParamsView:
    """
    Lightweight parameter-view for a single component.

    Provides the two operations that :func:`getmodel` needs from its
    ``par`` argument:

    * ``.model[component_label]`` → model shape name
    * ``par[f"{param}_{label}"]`` → parameter value

    This allows :func:`getmodel` to operate on structured model data
    without depending on the ``{param}_{label}`` parser convention.
    """

    def __init__(
        self,
        component: _ComponentSpec,
        common_params: dict[str, _ParamSpec] | None = None,
    ):
        self._component = component
        self._common = dict(common_params or {})

        # .model dict for getmodel: {label: model_name}
        self.model = {component.label: component.model_name}

    # ------------------------------------------------------------------
    def __getitem__(self, key: str):
        # Key is f"{param}_{label}" — strip the known label suffix
        suffix = f"_{self._component.label}"
        raw_name = key[: -len(suffix)] if key.endswith(suffix) else key

        # Look up in component params first
        if raw_name in self._component.params:
            return float(self._component.params[raw_name].value)

        # Fall back to common params
        if raw_name in self._common:
            return float(self._common[raw_name].value)

        raise KeyError(key)

    # ------------------------------------------------------------------
    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False
