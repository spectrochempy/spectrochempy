"""
Private linear/log coordinates on the common physical parameter layout.

The composable Python 1D facade selects this adapter for explicit log
declarations; public ND fitting entries do not select it yet. The private
unprojected grid-callback backend can receive an explicitly prepared adapter.
For a log parameter q=ln(p/reference); otherwise q=p. The solver coordinate
is z=q/normalization_scale. References and physical magnitudes must already
use the same unit: unit conversion and public reference selection belong to
preparation, not to this numerical layer. Native bounds are mapped to z;
the historical unbounded bound transforms are not composed with this map.
"""

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


def _real_array(values, shape, name, *, finite=True):
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real array with shape {shape}")
    original = array
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        array = original.astype(float)
    if (finite and not np.isfinite(array).all()) or np.isnan(array).any():
        raise ValueError(f"{name} contains non-finite magnitudes")
    if np.any(np.isfinite(original) & ~np.isfinite(array)):
        raise ValueError(f"{name} overflows float64 magnitudes")
    if np.any((original != 0) & (array == 0)):
        raise ValueError(f"{name} underflows float64 magnitudes")
    return array


@dataclass(frozen=True)
class _ParameterCoordinates:
    """
    Snapshot mixed coordinates for independent free float64 magnitudes.

    ``transforms`` and ``references`` follow ``layout.free_names``; fixed and
    dependent records have no solver coordinate. Every log reference is
    explicit, positive and finite. A log domain requires a finite, strictly
    positive lower bound; its upper bound may be infinite. Initial values
    may equal a physical bound. This does not qualify any solver's stopping
    policy or enable endpoint starts through an existing public facade.

    Methods return new arrays, reject unrepresentable results, and never
    clip a value. Covariance conversion is algebraic only: identifiability,
    active-bound and variance-factor qualifications remain the caller's job.
    Exact encoded initial values and finite endpoints retain their physical
    anchors, avoiding an artificial inverse overflow from logarithm rounding.
    Distinct physical anchors collapsing to one coordinate are rejected.
    """

    layout: _ParameterLayout
    transforms: tuple | None = None
    references: tuple | None = None
    initial: tuple = field(init=False)
    bounds: tuple = field(init=False)
    _scales: tuple = field(init=False, repr=False)
    _anchors: tuple = field(init=False, repr=False)

    def __post_init__(self):
        n = len(self.layout.free_parameters)
        transforms = ("lin",) * n if self.transforms is None else tuple(self.transforms)
        references = (None,) * n if self.references is None else tuple(self.references)
        if len(transforms) != n or len(references) != n:
            raise ValueError("Coordinates must match the free parameter count")
        if any(t not in ("lin", "log") for t in transforms):
            raise ValueError("Only lin and log coordinates are qualified")
        scales = _real_array(self.layout.free_scales, (n,), "Conditioning scales")
        if np.any(scales <= 0):
            raise ValueError("Conditioning scales must be strictly positive")
        object.__setattr__(self, "_scales", tuple(scales))
        for transform, reference in zip(transforms, references, strict=True):
            if transform == "lin":
                if reference is not None:
                    raise ValueError("Linear coordinates do not use a log reference")
            elif reference is None or _real_array(reference, (), "Log reference") <= 0:
                raise ValueError(
                    "Log coordinates require a positive explicit reference"
                )
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(
            self,
            "references",
            tuple(None if r is None else float(r) for r in references),
        )
        physical_bounds = _real_array(
            [
                (
                    low if low is not None else -np.inf,
                    high if high is not None else np.inf,
                )
                for low, high in self.layout.free_bounds
            ],
            (n, 2) if n else (0,),
            "Physical bounds",
            finite=False,
        ).reshape(n, 2)
        if np.any(physical_bounds[:, 0] >= physical_bounds[:, 1]):
            raise ValueError("Physical bounds must be strictly ordered")
        for index, transform in enumerate(transforms):
            if transform == "log" and (
                not np.isfinite(physical_bounds[index, 0])
                or physical_bounds[index, 0] <= 0
            ):
                raise ValueError(
                    "Log coordinates require a positive finite lower bound"
                )
        values = _real_array(self.layout.free_values, (n,), "Initial physical values")
        if np.any(values < physical_bounds[:, 0]) or np.any(
            values > physical_bounds[:, 1]
        ):
            raise ValueError("Initial physical values must lie inside their bounds")
        object.__setattr__(self, "initial", tuple(self.to_solver(values)))
        low = self._encode(physical_bounds[:, 0], finite=False)
        high = self._encode(physical_bounds[:, 1], finite=False)
        if np.any(low >= high):
            raise ValueError("Physical bounds collapse in solver coordinates")
        object.__setattr__(self, "bounds", tuple(zip(low, high, strict=True)))
        anchors = []
        for index, initial in enumerate(self.initial):
            mapping = {initial: values[index]}
            for coordinate, physical in (
                (low[index], physical_bounds[index, 0]),
                (high[index], physical_bounds[index, 1]),
            ):
                if np.isfinite(physical):
                    if coordinate in mapping and mapping[coordinate] != physical:
                        raise ValueError(
                            "Distinct physical anchors collapse in solver coordinates"
                        )
                    mapping[coordinate] = physical
            anchors.append(tuple(mapping.items()))
        object.__setattr__(self, "_anchors", tuple(anchors))

    def _encode(self, values, *, finite=True):
        n = len(self.layout.free_parameters)
        result = _real_array(values, (n,), "Physical values", finite=finite).copy()
        original = result.copy()
        with np.errstate(
            over="ignore", under="ignore", invalid="ignore", divide="ignore"
        ):
            for index, transform in enumerate(self.transforms):
                if transform == "log":
                    if result[index] <= 0:
                        raise ValueError(
                            "Log physical values must be strictly positive"
                        )
                    if np.isinf(result[index]):
                        continue
                    reference = self.references[index]
                    relative = (result[index] - reference) / reference
                    if abs(relative) < 0.5:
                        result[index] = np.log1p(relative)
                    else:
                        value, power = np.frexp(result[index])
                        reference, reference_power = np.frexp(reference)
                        result[index] = np.log(value / reference) + (
                            power - reference_power
                        ) * np.log(2.0)
            coordinates = result.copy()
            result /= np.asarray(self._scales)
        if (
            np.isnan(result).any()
            or np.any(np.isfinite(original) & ~np.isfinite(result))
            or np.any((coordinates != 0) & (result == 0))
        ):
            raise ValueError(
                "Physical values are not representable in solver coordinates"
            )
        return result

    def to_solver(self, values):
        """Map finite physical magnitudes to z without enforcing declared bounds."""
        return self._encode(values)

    def to_physical(self, values):
        """Map z to finite physical magnitudes, rejecting log overflow/zero."""
        n = len(self.layout.free_parameters)
        original = _real_array(values, (n,), "Solver values")
        result = original.copy()
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result *= np.asarray(self._scales)
            for index, transform in enumerate(self.transforms):
                anchors = dict(self._anchors[index])
                if original[index] in anchors:
                    result[index] = anchors[original[index]]
                    continue
                if transform == "log":
                    reference = self.references[index]
                    result[index] = (
                        reference * np.exp(result[index])
                        if abs(result[index]) <= 700
                        else np.exp(np.log(reference) + result[index])
                    )
                    if result[index] == 0:
                        raise ValueError("Log physical values underflow to zero")
                elif original[index] != 0 and result[index] == 0:
                    raise ValueError("Linear physical values underflow to zero")
        if not np.isfinite(result).all():
            raise ValueError("Solver values overflow physical magnitudes")
        return result

    def derivatives(self, values):
        """Return diagonal dp/dz, including the separate conditioning scale."""
        physical = self.to_physical(values)
        result = np.array(self._scales, dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for index, transform in enumerate(self.transforms):
                if transform == "log":
                    result[index] *= physical[index]
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("Physical coordinate derivatives are not representable")
        return result

    def physical_jacobian(self, solver_jacobian, values):
        """Convert Jr,z to Jr,p once; preserve the caller's residual weighting."""
        derivatives = self.derivatives(values)
        matrix = np.asarray(solver_jacobian)
        if matrix.ndim != 2:
            raise ValueError("Residual Jacobian must be two dimensional")
        matrix = _real_array(matrix, (matrix.shape[0], len(derivatives)), "Jacobian")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = matrix / derivatives
        if not np.isfinite(result).all() or np.any((matrix != 0) & (result == 0)):
            raise ValueError("Physical Jacobian is not representable")
        return result

    def physical_covariance(self, solver_covariance, values):
        """Apply diag(dp/dz) Cov(z) diag(dp/dz), retaining free order."""
        derivatives = self.derivatives(values)
        n = len(derivatives)
        matrix = _real_array(solver_covariance, (n, n), "Covariance")
        factors, powers = np.frexp(derivatives)
        mantissas, exponents = np.frexp(matrix)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.ldexp(
                factors[:, None] * mantissas * factors[None, :],
                powers[:, None] + exponents + powers[None, :],
            )
        if not np.isfinite(result).all() or np.any((matrix != 0) & (result == 0)):
            raise ValueError("Physical covariance is not representable")
        return result
