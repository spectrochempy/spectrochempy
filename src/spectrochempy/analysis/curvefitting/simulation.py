# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Simulate sums of separable spectral peaks on a coordinate grid."""

from collections.abc import Mapping
from collections.abc import Sequence
from numbers import Real

import numpy as np
from pint import Quantity as PintQuantity
from pint.errors import PintError

from spectrochempy.analysis.curvefitting._models import _GAUSSIAN_WIDTH_PER_SIGMA
from spectrochempy.analysis.curvefitting._models import gaussian
from spectrochempy.analysis.curvefitting._models import lorentzian
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity
from spectrochempy.core.units._conversion import _import_quantity

__all__ = ["simulate_peaks"]
__dataset_methods__ = ["simulate_peaks"]

_MODELS = ("gaussian", "lorentzian")
_GAUSSIAN_MODEL_WIDTH = _GAUSSIAN_WIDTH_PER_SIGMA / np.sqrt(8 * np.log(2))


def _interval_unit(unit, quantity_type=Quantity):
    """Return the corresponding difference unit for an affine unit."""
    return (quantity_type(1, unit) - quantity_type(0, unit)).units


def _parameter_value(value, unit, name, *, interval=False):
    """Return a finite real magnitude in the target unit."""
    quantity = value if isinstance(value, PintQuantity) else None
    magnitude = quantity.magnitude if quantity is not None else value
    if isinstance(magnitude, (bool, np.bool_, np.timedelta64)) or not isinstance(
        magnitude, Real
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        if quantity is not None:
            if unit is None:
                raise ValueError(f"{name} requires units on the template")
            if interval:
                quantity_type = type(quantity)
                quantity = quantity_type(
                    magnitude, _interval_unit(quantity.units, quantity_type)
                )
                unit = _interval_unit(unit)
            if not isinstance(quantity, Quantity):
                quantity = _import_quantity(quantity, unit)
            magnitude = quantity.to(unit).magnitude
        numeric = (
            magnitude
            if isinstance(magnitude, np.longdouble) and name != "height"
            else float(magnitude)
        )
    except PintError as exc:
        raise ValueError(f"Incompatible units for {name}") from exc
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real scalar")
    if isinstance(magnitude, (int, np.integer)):
        return int(magnitude)
    return magnitude if isinstance(magnitude, np.longdouble) else numeric


def _scaled_offsets(values, center, width):
    """Compute centered distances without rounding either operand before subtraction."""
    dtype = np.result_type(
        values.dtype,
        type(center) if isinstance(center, np.floating) else np.float64,
        type(width) if isinstance(width, np.floating) else np.float64,
    )
    center_value = dtype.type(center)
    width_value = dtype.type(width)
    if values.dtype.kind in "iu":
        whole_center = int(center)
        displacement = (values.astype(object) - whole_center).astype(dtype)
        if not isinstance(center, int):
            displacement -= center_value - dtype.type(whole_center)
    else:
        displacement = values.astype(dtype) - center_value
        if isinstance(center, int):
            displacement -= dtype.type(center - int(center_value))
    offsets = displacement / width_value
    overflow = np.isinf(displacement)
    offsets[overflow] = values[overflow] / width_value - center_value / width_value
    displacement[overflow] = values[overflow] * 0.5 - center_value * 0.5
    return offsets, displacement, overflow


def _profile_factors(model, profile, offsets, displacement, overflow, width):
    """Keep profile tails as a mantissa and a binary exponent."""
    mantissa, exponent = np.frexp(profile)
    tiny = profile < np.finfo(float).tiny
    if np.any(tiny):
        if model == "lorentzian":
            distance, power = np.frexp(np.abs(displacement[tiny]))
            span, span_power = np.frexp(
                float(width) if isinstance(width, int) else width
            )
            mantissa[tiny] = (0.5 * span / distance) ** 2
            exponent[tiny] = 2 * (span_power - power - overflow[tiny])
        else:
            log2_profile = np.maximum(-4 * offsets[tiny] ** 2, -4096)
            power = np.floor(log2_profile).astype(int)
            mantissa[tiny] = np.exp2(log2_profile - power)
            exponent[tiny] = power
    return mantissa, exponent


def _history_magnitude(value):
    """Serialize extended precision parameters without rounding them to float64."""
    return str(value) if isinstance(value, np.longdouble) else value


def _evaluate_separable_peak(shape, coordinates, peak):
    """Evaluate one normalized peak using a broadcast vector for each axis."""
    height = peak["height"]
    contribution = np.full(shape, height, dtype=float)
    if height == 0:
        return contribution

    profiles = []
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        for axis, (dim, values) in enumerate(coordinates.items()):
            params = peak["axes"][dim]
            model = params["model"]
            offsets, displacement, overflow = _scaled_offsets(
                values, params["center"], params["width"]
            )
            if model == "lorentzian":
                tail = np.abs(offsets) > 1
                reciprocal = 0.5 / offsets[tail]
                arguments = offsets.copy()
                arguments[tail] = 0.5 * reciprocal
                profile = lorentzian(arguments, normalized=False)
                profile[tail] *= reciprocal**2
            else:
                profile = gaussian(
                    offsets, width=_GAUSSIAN_MODEL_WIDTH, normalized=False
                )
            axis_shape = [1] * len(shape)
            axis_shape[axis] = values.size
            profiles.append(
                (model, profile, offsets, displacement, overflow, params["width"])
            )
            contribution *= profile.reshape(axis_shape)

        tiny = np.finfo(float).tiny
        recover = np.abs(contribution) < tiny
        for axis, (_, profile, *_rest) in enumerate(profiles):
            axis_shape = [1] * len(shape)
            axis_shape[axis] = profile.size
            recover |= (profile < tiny).reshape(axis_shape)

        if np.any(recover):
            mantissa, exponent = np.frexp(float(height))
            for axis, indices in enumerate(np.nonzero(recover)):
                model, *vectors, width = profiles[axis]
                factor, power = _profile_factors(
                    model, *(vector[indices] for vector in vectors), width
                )
                mantissa = mantissa * factor
                exponent = exponent + power
            contribution[recover] = np.ldexp(mantissa, exponent)
    return contribution


def simulate_peaks(template, peaks):
    """
    Simulate an additive spectrum of separable peaks on a coordinate grid.

    Parameters
    ----------
    template : NDDataset
        Nonempty dataset supplying the shape, coordinates, mask and intensity
        units. Each axis must have a finite, real numeric coordinate. Where
        multiple coordinates exist, use the default one. Sample intensities
        in the template are ignored.
    peaks : sequence of mapping
        Each mapping has a signed scalar ``height`` and an ``axes`` mapping
        keyed by all template dimension names. For each axis specify
        ``model`` ("gaussian" or "lorentzian"), ``center`` and ``width``
        (positive full width at half maximum). Scalars use the corresponding
        template units. Pint quantities are converted, with widths treated
        as intervals. An empty sequence gives a zero spectrum.

    Returns
    -------
    NDDataset
        Simulated intensities with detached coordinates, mask and provenance.

    Notes
    -----
    Each contribution is its height times the product of unit-height axis
    profiles, summed in the supplied peak order. The height is defined at the
    continuous center and may differ from the sampled maximum. Masked locations
    are simulated as well and remain masked in the output.

    Examples
    --------
    >>> import spectrochempy as scp
    >>> grid = scp.NDDataset([0, 0, 0], coordset=[scp.Coord([-1, 0, 1])])
    >>> peak = {
    ...     "height": -2,
    ...     "axes": {"x": {"model": "lorentzian", "center": 0, "width": 2}},
    ... }
    >>> scp.simulate_peaks(grid, [peak]).data.tolist()
    [-1.0, -2.0, -1.0]
    """
    if template.ndim == 0 or template.size == 0:
        raise ValueError("simulate_peaks requires a nonempty coordinate grid")
    if not isinstance(peaks, Sequence) or isinstance(peaks, (str, bytes)):
        raise TypeError("peaks must be an ordered sequence of mappings")

    coordinates = {}
    units = {}
    for axis, dim in enumerate(template.dims):
        coord = getattr(template, dim, None)
        if isinstance(coord, CoordSet):
            coord = coord.default
        if not isinstance(coord, Coord) or not coord.has_data:
            raise ValueError(f"Dimension {dim!r} requires a numeric coordinate")
        values = np.asarray(coord._data)
        if (
            values.shape != (template.shape[axis],)
            or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Dimension {dim!r} requires a finite numeric coordinate")
        coordinates[dim] = values
        units[dim] = coord.units

    parameters = []
    for peak in peaks:
        if not isinstance(peak, Mapping) or set(peak) != {"height", "axes"}:
            raise ValueError("Each peak requires height and axes")
        axes = peak["axes"]
        if not isinstance(axes, Mapping) or set(axes) != set(coordinates):
            raise ValueError("Each peak must specify every template dimension")
        height = _parameter_value(peak["height"], template.units, "height")
        resolved = {}
        for dim in coordinates:
            spec = axes[dim]
            if not isinstance(spec, Mapping) or set(spec) != {
                "model",
                "center",
                "width",
            }:
                raise ValueError(f"Axis {dim!r} requires model, center and width")
            model = spec["model"]
            if not isinstance(model, str) or model not in _MODELS:
                raise ValueError(f"Unsupported model on axis {dim!r}: {model!r}")
            center = _parameter_value(spec["center"], units[dim], f"{dim} center")
            width = _parameter_value(
                spec["width"], units[dim], f"{dim} width", interval=True
            )
            if width <= 0:
                raise ValueError(f"{dim} width must be strictly positive")
            resolved[dim] = {"model": model, "center": center, "width": width}
        parameters.append({"height": height, "axes": resolved})

    data = np.zeros(template.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        for peak in parameters:
            data += _evaluate_separable_peak(template.shape, coordinates, peak)
    if not np.isfinite(data).all():
        raise ValueError("Simulated intensity exceeds the finite floating-point range")

    result = NDDataset(
        data,
        dims=list(template.dims),
        coordset=template.coordset.copy(),
        mask=np.array(template.mask, copy=True),
        units=template.units,
        title="simulated peak intensity",
        history=template.history_entries,
    )
    result._append_history_entry(
        operation="simulate_peaks",
        parameters={
            "dims": list(template.dims),
            "axis_units": {
                dim: str(unit) if unit is not None else None
                for dim, unit in units.items()
            },
            "intensity_units": str(template.units)
            if template.units is not None
            else None,
            "amplitude": "signed_height_at_continuous_center",
            "peaks": [
                {
                    "height": _history_magnitude(peak["height"]),
                    "axes": {
                        dim: {
                            key: _history_magnitude(value)
                            for key, value in spec.items()
                        }
                        for dim, spec in peak["axes"].items()
                    },
                }
                for peak in parameters
            ],
        },
        message=f"Simulated {len(parameters)} separable peak(s)",
    )
    return result
