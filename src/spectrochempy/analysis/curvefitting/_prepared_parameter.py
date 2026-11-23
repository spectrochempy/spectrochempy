"""Private physical parameter records shared by fitting preparation adapters."""

from dataclasses import dataclass
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
