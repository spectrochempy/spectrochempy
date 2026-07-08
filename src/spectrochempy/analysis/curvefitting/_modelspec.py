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

import sys
from dataclasses import dataclass
from dataclasses import field
from typing import Any

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


# ======================================================================================
@dataclass
class _ParamSpec:
    """Parameter definition with value, bounds, and flags."""

    name: str
    value: float = 0.0
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
    ):
        self.components = list(components or [])
        self.common_params = dict(common_params or {})
        self.constraints = constraints  # reserved, not yet used

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
                common_params[raw_name] = _ParamSpec(
                    name=raw_name,
                    value=float(fp[raw_name]),
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
                    model_params[param_raw] = _ParamSpec(
                        name=param_raw,
                        value=float(fp[full_key]),
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

        return cls(components=components, common_params=common_params)

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


def _format_bound(val: float | None) -> str:
    """Format a bound value for script output."""
    if val is None:
        return "none"
    # Use integer format if exact integer
    if val == int(val):
        return f"{int(val)}"
    return f"{val:.4f}"
