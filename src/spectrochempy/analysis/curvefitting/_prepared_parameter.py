"""Private physical parameter records and ordered execution layouts."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class _PreparedParameter:
    """Describe a prepared parameter independently of model topology and solver."""

    identifier: tuple[str | int, ...]
    name: str
    value: Any
    bounds: tuple[Any, Any] = (None, None)
    fixed: bool = False
    unit: Any = None

    def as_legacy_tuple(self):
        """Return the historical ND parser tuple without converting its values."""
        return self.value, *self.bounds, self.fixed


@dataclass(frozen=True)
class _ParameterLayout:
    """
    Preserve normalized declaration order without choosing solver coordinates.

    Physical records remain separate from numerical conditioning scales.
    This layout performs no sorting, magnitude conversion or expression evaluation.
    """

    parameters: tuple[_PreparedParameter, ...]
    normalization_scales: tuple | None = None
    free_parameters: tuple[_PreparedParameter, ...] = field(init=False)
    free_names: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        parameters = tuple(self.parameters)
        if len({p.name for p in parameters}) != len(parameters):
            raise ValueError("Prepared parameter names must be unique")
        if len({p.identifier for p in parameters}) != len(parameters):
            raise ValueError("Prepared parameter identifiers must be unique")
        free = tuple(p for p in parameters if not p.fixed)
        scales = (
            (1.0,) * len(free)
            if self.normalization_scales is None
            else tuple(self.normalization_scales)
        )
        if len(scales) != len(free):
            raise ValueError("Conditioning scales must match the free parameter count")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "normalization_scales", scales)
        object.__setattr__(self, "free_parameters", free)
        object.__setattr__(self, "free_names", tuple(p.name for p in free))

    @property
    def free_values(self):
        """Return physical magnitudes without coercing their numeric precision."""
        return tuple(p.value for p in self.free_parameters)

    @property
    def free_bounds(self):
        """Return the adapter's physical bounds without sentinel conversion."""
        return tuple(p.bounds for p in self.free_parameters)

    @property
    def free_units(self):
        """Return units in the same order as the free magnitudes."""
        return tuple(p.unit for p in self.free_parameters)

    @property
    def free_scales(self):
        """Return numerical conditioning scales, independently of transforms."""
        return self.normalization_scales
