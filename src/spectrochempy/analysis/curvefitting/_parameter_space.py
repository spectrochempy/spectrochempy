# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Adapters between fitting definitions and bounded optimizer vectors.

This preparation layer preserves the canonical and legacy parameter layouts.
It does not parse scripts, evaluate profiles, or select an optimization method.
"""

from dataclasses import dataclass

import numpy as np

from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._parameter_transform import _LOB_THRESHOLD
from spectrochempy.analysis.curvefitting._parameter_transform import _UPB_THRESHOLD
from spectrochempy.analysis.curvefitting._parameter_transform import (
    _external_transform_state,
)
from spectrochempy.analysis.curvefitting._parameter_transform import _to_external
from spectrochempy.analysis.curvefitting._parameter_transform import _to_internal
from spectrochempy.analysis.curvefitting._prepared_parameter import _PreparedParameter


@dataclass(frozen=True)
class _ScalarParameterSpace:
    """Bind a qualified scalar plan to transformed or physical solver coordinates."""

    slots: tuple

    @classmethod
    def from_plan(cls, plan):
        """Prepare owner positions once, without retaining mutable model objects."""
        if plan.fallback:
            raise ValueError("Scalar parameter vectors require a qualified plan")
        indices = {label: index for index, (label, _) in enumerate(plan.components)}
        return cls(
            tuple(
                (
                    None if p.identifier[0] == "common" else indices[p.identifier[1]],
                    p.identifier[-1],
                    p,
                )
                for p in plan.parameters
                if not p.fixed
            )
        )

    @staticmethod
    def _entry(definition, slot):
        index, key, _ = slot
        owner = (
            definition.common_params
            if index is None
            else definition.components[index].params
        )
        return owner[key]

    def internal_vector(self, definition):
        """Read current magnitudes using the snapshotted free order and bounds."""
        values = [
            _to_internal(self._entry(definition, slot).value, *slot[2].bounds)
            for slot in self.slots
        ]
        return values, [p.name for _, _, p in self.slots]

    def restore(self, definition, values, *, physical=False):
        """Restore a complete vector in place, leaving fixed and linked entries intact."""
        if len(values) != len(self.slots):
            raise ValueError(
                "Parameter vector length does not match its prepared space"
            )
        for slot, value in zip(self.slots, values, strict=True):
            self._entry(definition, slot).value = (
                float(value) if physical else _to_external(value, *slot[2].bounds)
            )
        return definition

    def physical_values(self, definition):
        """Read the current physical free vector in solver order."""
        return np.array(
            [self._entry(definition, slot).value for slot in self.slots], dtype=float
        )

    @property
    def physical_bounds(self):
        """Apply the existing sentinel-bound interpretation for physical probes."""
        return [
            (
                low if low is not None and low > _LOB_THRESHOLD else -np.inf,
                high if high is not None and high < _UPB_THRESHOLD else np.inf,
            )
            for _, _, parameter in self.slots
            for low, high in (parameter.bounds,)
        ]

    def transform_artifacts(self, values):
        """Evaluate signed transform derivatives in the prepared solver order."""
        return _artifacts_from_parameters((p for _, _, p in self.slots), values)


def _modelspec_parameter_count(spec):
    """Return the FitParameters-compatible total parameter count for a spec."""
    return len(spec.common_params) + sum(len(comp.params) for comp in spec.components)


def _restore_modelspec_external(spec, values, keys):
    """Restore external parameter values in a model spec from optimizer values."""
    key_index = {key: idx for idx, key in enumerate(keys)}
    for key, ps in spec._iter_varying():
        lob, upb = ps.bounds
        if ps.name in spec.expvars:
            restored = []
            for i in range(spec.expnumber):
                exp_key = f"{key}_exp{i}"
                if exp_key not in key_index:
                    break
                restored.append(_to_external(values[key_index[exp_key]], lob, upb))
            if restored:
                ps.value = restored
        else:
            if key not in key_index:
                continue
            ps.value = _to_external(values[key_index[key]], lob, upb)
    return spec


def _modelspec_internal_vector(spec):
    """Return internal optimizer values and keys for varying spec parameters."""
    return _vector_from_prepared(_prepared_parameters(spec))


def _prepared_parameters(definition):
    """Snapshot 1D declarations using stable identifiers and historical ordering."""
    declarations = []
    if isinstance(definition, _FitModelSpec):
        for name, parameter in definition.common_params.items():
            declarations.append(
                (
                    name,
                    ("common", name),
                    parameter.name,
                    parameter.value,
                    parameter.bounds,
                    not parameter.vary,
                )
            )
        for component in definition.components:
            for name, parameter in component.params.items():
                declarations.append(
                    (
                        f"{name}_{component.label}",
                        ("components", component.label, name),
                        parameter.name,
                        parameter.value,
                        parameter.bounds,
                        not parameter.vary,
                    )
                )
        declarations.sort(key=lambda entry: entry[0])
    else:
        for key in sorted(definition.keys()):
            declarations.append(
                (
                    key,
                    ("legacy", key),
                    key.split("_")[0],
                    definition.data[key],
                    (definition.lob.get(key), definition.upb.get(key))
                    if definition.fixed[key]
                    else (definition.lob[key], definition.upb[key]),
                    definition.fixed[key],
                )
            )

    parameters = []
    for key, identifier, source_name, value, bounds, fixed in declarations:
        if not fixed and source_name in definition.expvars:
            experiment_values = [value] if np.ndim(value) == 0 else value
            for index in range(definition.expnumber):
                parameters.append(
                    _PreparedParameter(
                        (*identifier, "experiments", index),
                        f"{key}_exp{index}",
                        experiment_values[index],
                        bounds,
                    )
                )
        else:
            parameters.append(
                _PreparedParameter(
                    identifier,
                    key,
                    value.copy() if isinstance(value, list) else value,
                    bounds,
                    fixed,
                )
            )
    return tuple(parameters)


def _vector_from_prepared(parameters):
    """Apply the historical 1D bound transform to free physical parameters."""
    values, keys = [], []
    for parameter in parameters:
        if not parameter.fixed:
            values.append(_to_internal(parameter.value, *parameter.bounds))
            keys.append(parameter.name)
    return values, keys


def _internal_vector(fp):
    """Return optimizer values and keys in historical parameter order."""
    return _vector_from_prepared(_prepared_parameters(fp))


def _physical_vector(definition):
    """Prepare finite physical free values and inclusive native bounds."""
    parameters = [p for p in _prepared_parameters(definition) if not p.fixed]
    legacy = not isinstance(definition, _FitModelSpec)
    values = np.array([p.value for p in parameters], dtype=float)
    bounds = np.array(
        [
            (
                -np.inf
                if p.bounds[0] is None or (legacy and p.bounds[0] <= _LOB_THRESHOLD)
                else p.bounds[0],
                np.inf
                if p.bounds[1] is None or (legacy and p.bounds[1] >= _UPB_THRESHOLD)
                else p.bounds[1],
            )
            for p in parameters
        ],
        dtype=float,
    ).reshape(-1, 2)
    if (
        values.shape != (len(parameters),)
        or not np.isfinite(values).all()
        or np.isnan(bounds).any()
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(values < bounds[:, 0])
        or np.any(values > bounds[:, 1])
    ):
        raise ValueError(
            "Native bounds require finite initial values inside ordered bounds"
        )
    return values, [p.name for p in parameters], bounds


def _transform_artifacts(definition, values, keys):
    """Snapshot physical-coordinate derivatives in the actual solver order."""
    parameters = {p.name: p for p in _prepared_parameters(definition) if not p.fixed}
    return _artifacts_from_parameters((parameters[key] for key in keys), values)


def _artifacts_from_parameters(parameters, values):
    """Preserve bound qualification for prepared or historical parameter orders."""
    states = [
        _external_transform_state(value, *parameter.bounds)
        for parameter, value in zip(parameters, values, strict=True)
    ]
    return {
        "parameter_transform_derivatives": tuple(float(state[0]) for state in states),
        "near_parameter_bounds": any(state[1] for state in states),
    }


def _restore_physical(definition, values, keys):
    """Assign physical free magnitudes without a round trip through solver space."""
    magnitudes = dict(zip(keys, values, strict=True))
    components = (
        {component.label: component for component in definition.components}
        if isinstance(definition, _FitModelSpec)
        else {}
    )
    for parameter in _prepared_parameters(definition):
        if parameter.fixed or parameter.name not in magnitudes:
            continue
        path = parameter.identifier
        if path[0] == "legacy":
            owner, name = definition.data, path[1]
        elif path[0] == "common":
            owner, name = definition.common_params, path[1]
        else:
            owner, name = components[path[1]].params, path[2]
        entry = owner[name]
        value = entry if path[0] == "legacy" else entry.value
        if len(path) > 3 and path[-2] == "experiments":
            was_scalar = np.ndim(value) == 0
            value = [value] if was_scalar else list(value)
            value[path[-1]] = float(magnitudes[parameter.name])
            if path[0] == "legacy" and was_scalar:
                value = value[0]
        else:
            value = float(magnitudes[parameter.name])
        if path[0] == "legacy":
            owner[name] = value
        else:
            entry.value = value
    return definition


def _restore_native(definition, values, keys):
    """Restore native values with the historical one-experiment scalar layout."""
    _restore_physical(definition, values, keys)
    if not isinstance(definition, _FitModelSpec) and definition.expnumber == 1:
        for parameter in _prepared_parameters(definition):
            path = parameter.identifier
            if not parameter.fixed and len(path) > 3 and path[-2] == "experiments":
                value = definition.data[path[1]]
                if isinstance(value, list) and len(value) == 1:
                    definition.data[path[1]] = value[0]
    return definition


def _restore_external(fp, p, keys):
    """Restore physical values in place using the supplied optimizer keys."""
    if isinstance(fp, _FitModelSpec):
        return _restore_modelspec_external(fp, p, keys)

    for key in list(fp.keys()):
        keysp = key.split("_")
        if keysp[0] in fp.expvars:
            ps = []
            for i in range(fp.expnumber):
                ks = f"{key}_exp{i}"
                if ks not in keys:
                    break
                k = keys.index(ks)
                ps.append(p[k])
            if ps:
                fp.data[key] = _to_external(
                    ps,
                    fp.lob[key],
                    fp.upb[key],
                )
        else:
            if key not in keys:
                continue
            k = keys.index(key)
            fp.data[key] = _to_external(
                p[k],
                fp.lob[key],
                fp.upb[key],
            )
    return fp
