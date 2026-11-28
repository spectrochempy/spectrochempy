"""Shared copied context and physical inference helpers for centered ND fits."""

from copy import deepcopy

import numpy as np

from spectrochempy.analysis.curvefitting._centered_coordinates import (
    _CenteredParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._centered_coordinates import _magnitude
from spectrochempy.core.units import ur


def _bind_centered_coordinates(layout, coordinates, *, kind="profile"):
    """Copy an adapter only after matching every physical declaration, fixes included."""
    if not isinstance(coordinates, _CenteredParameterCoordinates):
        raise TypeError(
            f"{kind.capitalize()} coordinates require a prepared centered coordinate space"
        )
    if len(coordinates.layout.parameters) != len(layout.parameters):
        raise ValueError(
            f"Coordinate declarations must match the complete {kind} layout"
        )
    for supplied, expected in zip(
        coordinates.layout.parameters, layout.parameters, strict=True
    ):
        metadata = (supplied.identifier, supplied.name, supplied.fixed, supplied.unit)
        if metadata != (
            expected.identifier,
            expected.name,
            expected.fixed,
            expected.unit,
        ):
            raise ValueError(
                f"Coordinate declarations must match the complete {kind} layout"
            )
        if not isinstance(supplied.fixed, (bool, np.bool_)):
            raise ValueError("Coordinate fixed flags must be boolean")
        for a, b in zip(
            (supplied.value, *supplied.bounds),
            (expected.value, *expected.bounds),
            strict=True,
        ):
            if _magnitude(a, finite=False) != _magnitude(b, finite=False):
                raise ValueError(
                    f"Coordinate declarations must match the complete {kind} layout"
                )
    for record, transform in zip(
        layout.free_parameters, coordinates.transforms, strict=True
    ):
        if (
            transform == "log"
            and record.unit is not None
            and ur.Quantity(0, record.unit).to_base_units().magnitude != 0
        ):
            raise ValueError("Log coordinates require multiplicative parameter units")
    return deepcopy(coordinates)


def _coordinate_config(coordinates):
    """Keep exact initial magnitudes separate from physical fitted values."""
    return {
        p.name: {
            "coordinate_transform": transform,
            "normalization_scale": float(scale),
            "log_reference": reference,
            "origin": origin,
            "initial_value": deepcopy(p.value),
            "bounds": [
                deepcopy(b) if b is not None and np.isfinite(b) else None
                for b in p.bounds
            ],
            "unit": str(p.unit) if p.unit is not None else None,
        }
        for p, transform, scale, reference, origin in zip(
            coordinates.layout.free_parameters,
            coordinates.transforms,
            coordinates.layout.free_scales,
            coordinates.references,
            coordinates.origins,
            strict=True,
        )
    }


def _scaled_physical_jacobian(matrix, scales, intensity):
    """Scale physical columns without an overflowing intermediate product."""
    mantissa, power = np.frexp(matrix)
    scale, scale_power = np.frexp(scales)
    amplitude, amplitude_power = np.frexp(intensity)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.asarray(
            np.ldexp(
                mantissa * scale / amplitude, power + scale_power - amplitude_power
            ),
            dtype=float,
        )
