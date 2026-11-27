"""Opt-in private affine origins around the existing linear/log coordinate map."""

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from numbers import Integral

import numpy as np

from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


def _magnitude(value, *, finite=True):
    if hasattr(value, "units"):
        raise ValueError("Centered coordinates require preconverted real magnitudes")
    integer = isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
    original = value if integer else np.asarray(value)
    if not integer and (original.shape != () or original.dtype.kind not in "iuf"):
        raise ValueError("Centered coordinates require real scalar magnitudes")
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.longdouble(original)
    except OverflowError as exc:
        raise ValueError("Centered magnitude exceeds the platform range") from exc
    if np.isnan(result) or (finite and not np.isfinite(result)):
        raise ValueError("Centered magnitude must be finite")
    if integer or original.dtype.kind in "iu":
        if not np.isfinite(result) or int(result) != int(original):
            raise ValueError("Integer magnitude loses precision on this platform")
    elif result != original:
        raise ValueError("Magnitude loses precision on this platform")
    return result


def _offset(value, origin, *, finite=True):
    value = _magnitude(value, finite=finite)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        offset = value - origin
    if np.isfinite(value) and not np.isfinite(offset):
        raise ValueError("Physical offset exceeds the platform range")
    return offset


@dataclass(frozen=True)
class _CenteredParameterCoordinates:
    """
    Preserve explicit linear origins before conversion to float64 solver values.

    ``origins`` follows the free order, with None for uncentered parameters.
    A centered linear parameter uses z=(p-origin)/normalization_scale; a log
    parameter must remain uncentered and keeps its positive explicit reference.
    All magnitudes and origins are already converted to their parameter units.

    The existing coordinate adapter owns log mapping, conditioning, range
    checks and Jacobian/covariance conversion. This wrapper snapshots the full
    physical layout and retains exact initial/bound anchors. When any origin
    is supplied, physical output uses platform longdouble; solver output is
    float64. Origins/anchors unrepresentable on that platform are rejected.
    Other proposals reconstruct the physically representable point, which
    need not be a bijection at sub-ULP offsets. A fitting backend still needs
    centered evaluations and per-binding reconstruction for its derivatives.
    No existing estimator selects this adapter yet.
    """

    layout: _ParameterLayout
    origins: tuple
    transforms: tuple | None = None
    references: tuple | None = None
    initial: tuple = field(init=False)
    bounds: tuple = field(init=False)
    _coordinates: _ParameterCoordinates = field(init=False, repr=False)
    _anchors: tuple = field(init=False, repr=False)

    def __post_init__(self):
        layout = deepcopy(self.layout)
        n = len(layout.free_parameters)
        origins = tuple(self.origins)
        if len(origins) != n:
            raise ValueError("Origins must match the free parameter count")
        transforms = ("lin",) * n if self.transforms is None else tuple(self.transforms)
        if len(transforms) != n:
            raise ValueError("Coordinates must match the free parameter count")
        origins = tuple(None if o is None else _magnitude(o) for o in origins)
        prepared = []
        index = 0
        for parameter in layout.parameters:
            if parameter.fixed:
                prepared.append(parameter)
                continue
            origin = origins[index]
            if origin is not None:
                if transforms[index] != "lin":
                    raise ValueError("Only linear coordinates may have an origin")
                parameter = replace(
                    parameter,
                    value=_offset(parameter.value, origin),
                    bounds=tuple(
                        None if b is None else _offset(b, origin, finite=False)
                        for b in parameter.bounds
                    ),
                )
            prepared.append(parameter)
            index += 1
        coordinates = _ParameterCoordinates(
            _ParameterLayout(tuple(prepared), layout.free_scales),
            transforms,
            self.references,
        )
        anchors = []
        for index, parameter in enumerate(layout.free_parameters):
            mapping = {}
            if origins[index] is not None:
                for coordinate, physical in zip(
                    (coordinates.initial[index], *coordinates.bounds[index]),
                    (parameter.value, *parameter.bounds),
                    strict=True,
                ):
                    if physical is None:
                        continue
                    physical = _magnitude(physical, finite=False)
                    if not np.isfinite(physical):
                        continue
                    if coordinate in mapping and mapping[coordinate] != physical:
                        raise ValueError(
                            "Distinct physical anchors collapse in coordinates"
                        )
                    mapping[coordinate] = physical
            anchors.append(tuple(mapping.items()))
        for name, value in (
            ("layout", layout),
            ("origins", origins),
            ("transforms", coordinates.transforms),
            ("references", coordinates.references),
            ("initial", coordinates.initial),
            ("bounds", coordinates.bounds),
            ("_coordinates", coordinates),
            ("_anchors", tuple(anchors)),
        ):
            object.__setattr__(self, name, value)

    def to_solver(self, values):
        """Subtract origins in retained precision, then encode float64 coordinates."""
        values = np.asarray(values, dtype=object)
        if values.shape != (len(self.origins),):
            raise ValueError("Physical values must match the free parameter count")
        shifted = [
            _magnitude(value) if origin is None else _offset(value, origin)
            for value, origin in zip(values, self.origins, strict=True)
        ]
        return self._coordinates.to_solver(shifted)

    def to_physical(self, values):
        """Reconstruct physical magnitudes, retaining exact declared anchors."""
        shifted = self.to_offsets(values)
        solver = np.asarray(values, dtype=float)
        if all(origin is None for origin in self.origins):
            return shifted
        result = shifted.astype(np.longdouble)
        for index, origin in enumerate(self.origins):
            if origin is not None:
                anchors = dict(self._anchors[index])
                with np.errstate(over="ignore", invalid="ignore"):
                    result[index] = anchors.get(solver[index], origin + result[index])
        if not np.isfinite(result).all():
            raise ValueError("Solver values overflow centered physical magnitudes")
        return result

    def to_offsets(self, values):
        """Return centered linear differences and ordinary uncentered magnitudes."""
        return self._coordinates.to_physical(values)

    def derivatives(self, values):
        """Return dp/dz; an additive linear origin does not change this factor."""
        self.to_physical(values)
        return self._coordinates.derivatives(values)

    def physical_jacobian(self, matrix, values):
        """Preserve the common residual weighting and physical column order."""
        self.to_physical(values)
        return self._coordinates.physical_jacobian(matrix, values)

    def physical_covariance(self, matrix, values):
        """Use the common diagonal transformation and representability guards."""
        self.to_physical(values)
        return self._coordinates.physical_covariance(matrix, values)
