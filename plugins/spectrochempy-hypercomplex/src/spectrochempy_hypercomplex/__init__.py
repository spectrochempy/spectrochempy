# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""Hypercomplex / quaternion support plugin for SpectroChemPy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.utils.print import _format_array_values

# Public quaternion utilities — safe for other plugins to import
from ._quaternion import _HAS_QUATERNION as is_available  # noqa: F401, N811
from ._quaternion import as_float_array  # noqa: F401
from ._quaternion import as_quat_array  # noqa: F401
from ._quaternion import as_quaternion  # noqa: F401


class HyperAccessor:
    """
    Dataset accessor for hypercomplex / quaternion operations.

    Accessed via ``dataset.hyper``.
    """

    def __init__(self, dataset) -> None:
        self._dataset = dataset

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_quaternion(self) -> bool:
        """True if the data array is hypercomplex (quaternion)."""
        from ._quaternion import typequaternion  # noqa: PLC0415

        return (
            typequaternion is not None and self._dataset._data.dtype == typequaternion
        )

    @property
    def RR(self):
        """Real-real component of a 2D hypercomplex array."""
        if not self.is_quaternion:
            raise TypeError("Not a hypercomplex array")
        return self.component("RR")

    @property
    def RI(self):
        """Real-imaginary component of a 2D hypercomplex array."""
        if not self.is_quaternion:
            raise TypeError("Not a hypercomplex array")
        return self.component("RI")

    @property
    def IR(self):
        """Imaginary-real component of a 2D hypercomplex array."""
        if not self.is_quaternion:
            raise TypeError("Not a hypercomplex array")
        return self.component("IR")

    @property
    def II(self):
        """Imaginary-imaginary component of a 2D hypercomplex array."""
        if not self.is_quaternion:
            raise TypeError("Not a hypercomplex array")
        return self.component("II")

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def set_quaternion(self, inplace=False):
        """Convert the dataset data to quaternion (hypercomplex) type."""
        from ._quaternion import typequaternion  # noqa: PLC0415

        if typequaternion is None:
            raise RuntimeError(
                "numpy-quaternion is required for hypercomplex support. "
                "Install it with: pip install numpy-quaternion"
            )

        new = self._dataset if inplace else self._dataset.copy()
        dtype = new._data.dtype if hasattr(new._data, "dtype") else new.dtype
        if dtype != typequaternion:
            new._data = _make_quaternion(new.data)
        if not inplace:
            return new
        return None

    set_hypercomplex = set_quaternion
    set_hypercomplex.__doc__ = "Alias of set_quaternion."

    def component(self, select="REAL"):
        """
        Extract a named component from a hypercomplex array.

        Parameters
        ----------
        select : str
            Component selector: ``"RR"``, ``"RI"``, ``"IR"``, ``"II"``,
            or ``"REAL"`` / ``"R"`` / ``"I"`` for 1D complex data.
        """
        from ._quaternion import get_component  # noqa: PLC0415

        return get_component(self._dataset.data, select)


def _make_quaternion(data):
    """Convert real/interleaved data to quaternion dtype."""
    from ._quaternion import typequaternion  # noqa: PLC0415

    if data.ndim == 0:
        return data
    if typequaternion is not None and data.dtype == typequaternion:
        return data
    if data.ndim == 1:
        r, i = data[::2], data[1::2]
        return as_quaternion(r, i)
    if data.ndim == 2:
        return as_quaternion(data[:, ::2], data[:, 1::2])
    msg = (
        "An array of data to be transformed to quaternion must be 2D or 1D, "
        f"got {data.ndim}D."
    )
    raise ValueError(msg)


# ------------------------------------------------------------------
# NDMath handler implementations
# ------------------------------------------------------------------


# Ufunc names that numpy-quaternion handles natively (no decomposition needed).
# Everything else must be decomposed into complex arrays, executed,
# and recomposed.
_NATIVE_QUATERNION_UFUNCS = frozenset(
    [
        "add",
        "iadd",
        "sub",
        "isub",
        "subtract",
        "mul",
        "imul",
        "multiply",
        "div",
        "idiv",
        "divide",
        "true_divide",
        "truediv",
        "power",
        "negative",
        "positive",
        "sign",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "isnan",
        "isinf",
        "isfinite",
        "copysign",
        "nextafter",
        "spacing",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        "absolute",
        "fabs",
        "conj",
        "conjugate",
    ]
)


def _hyper_ndmath_branch(fname: str, data: np.ndarray, args: list) -> str | None:
    """Return ``"quaternion"`` when the operands require quaternion execution."""
    from ._quaternion import typequaternion  # noqa: PLC0415

    if typequaternion is None:
        return None

    is_quaternion = data.dtype == typequaternion
    for arg in args:
        if hasattr(arg, "dtype") and arg.dtype == typequaternion:
            is_quaternion = True
            break

    if not is_quaternion:
        return None

    # numpy-quaternion handles these ufuncs natively — no need to decompose.
    if fname in _NATIVE_QUATERNION_UFUNCS:
        return None

    return "quaternion"


def _hyper_ndmath_execute(branch: str, f, d, args):
    """Execute *f* on quaternion data *d* by decomposing into complex arrays."""
    if branch != "quaternion":
        return None
    from ._quaternion import quat_as_complex_array  # noqa: PLC0415

    dr, di = quat_as_complex_array(d)
    datar = f(dr, *args)
    datai = f(di, *args)
    return as_quaternion(datar, datai)


# ------------------------------------------------------------------
# Numpy-method overrides for quaternion data
# ------------------------------------------------------------------


def _is_quaternion_dataset(dataset) -> bool:
    """Return True if *dataset* carries quaternion data."""
    from ._quaternion import typequaternion  # noqa: PLC0415

    return (
        typequaternion is not None
        and hasattr(dataset, "_data")
        and hasattr(dataset._data, "dtype")
        and dataset._data.dtype == typequaternion
    )


def _hyper_numpy_abs(dataset, *args, **kwargs):
    """Quaternion-aware absolute value."""
    if not _is_quaternion_dataset(dataset):
        return None

    w, x, y, z = as_float_array(dataset.data).T
    data = np.ma.sqrt(w**2 + x**2 + y**2 + z**2)
    dataset._data = data.data
    dataset._mask = getattr(data, "mask", False)
    return dataset


def _hyper_numpy_conjugate(dataset, *args, **kwargs):
    """Quaternion-aware conjugate."""
    if not _is_quaternion_dataset(dataset):
        return None
    dim = kwargs.get("dim", "x")
    axis, _ = dataset.get_axis(dim, allows_none=True)
    dataset = dataset.swapdims(axis, -1)
    dataset[..., 1::2] = -dataset[..., 1::2]
    return dataset.swapdims(axis, -1)


def _hyper_numpy_max(dataset, *args, **kwargs):
    """Quaternion-aware max — operates on the real (w) component."""
    if not _is_quaternion_dataset(dataset):
        return None

    data = dataset.data
    w = as_float_array(data)[..., 0]
    axis = kwargs.get("axis")
    keepdims = kwargs.get("keepdims", False)
    m = np.ma.max(w, axis=axis, keepdims=keepdims)
    if np.isscalar(m) or (m.size == 1 and not keepdims):
        if not np.isscalar(m):
            m = m[()]
        if dataset.units is not None:
            from spectrochempy.core.units import Quantity  # noqa: PLC0415

            return Quantity(m, dataset.units)
        return m
    dataset._data = m.data
    dataset._mask = getattr(m, "mask", False)
    return dataset


def _hyper_numpy_min(dataset, *args, **kwargs):
    """Quaternion-aware min — operates on the real (w) component."""
    if not _is_quaternion_dataset(dataset):
        return None

    data = dataset.data
    w = as_float_array(data)[..., 0]
    axis = kwargs.get("axis")
    keepdims = kwargs.get("keepdims", False)
    m = np.ma.min(w, axis=axis, keepdims=keepdims)
    if np.isscalar(m) or (m.size == 1 and not keepdims):
        if not np.isscalar(m):
            m = m[()]
        if dataset.units is not None:
            from spectrochempy.core.units import Quantity  # noqa: PLC0415

            return Quantity(m, dataset.units)
        return m
    dataset._data = m.data
    dataset._mask = getattr(m, "mask", False)
    return dataset


# ------------------------------------------------------------------
# Display handlers for quaternion data
# ------------------------------------------------------------------


def _hyper_display_array_values(dataset, *, sep="\n", ufmt=" {:~P}"):
    """Return RR/RI/IR/II display blocks for quaternion datasets."""
    if not _is_quaternion_dataset(dataset):
        return None

    hyper = getattr(dataset, "hyper", None)
    if hyper is None or not hyper.is_quaternion:
        return None

    units = ufmt.format(dataset.units) if dataset.has_units else ""
    parts = []
    for pref in ("RR", "RI", "IR", "II"):
        data = np.asarray(getattr(hyper, pref))
        parts.append(
            _format_array_values(
                data,
                is_masked=False,
                dtype=data.dtype,
                sep=sep,
                prefix=pref,
                units=units,
            )
        )
    return sep.join(parts)


def _hyper_display_complex_dim_flags(dataset):
    """Mark quaternion dataset dimensions as complex in detailed display."""
    if not _is_quaternion_dataset(dataset):
        return None
    return [True] * dataset.ndim


# ------------------------------------------------------------------
# Plugin class
# ------------------------------------------------------------------


class HyperComplexPlugin(SpectroChemPyPlugin):
    """Hypercomplex / quaternion support for SpectroChemPy."""

    name = "hypercomplex"
    version = "0.1.5"
    description = "Hypercomplex / quaternion data support"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.ACCESSOR]

    def register_accessors(self) -> list[dict]:
        """Register the ``hyper`` dataset accessor."""
        return [
            {
                "name": "hyper",
                "func": HyperAccessor,
                "description": "Hypercomplex / quaternion dataset operations",
            },
        ]

    def register_handlers(self) -> dict[str, Callable]:
        """Register NDMath execution handlers for quaternion dispatch."""
        return {
            "ndmath.execution_branch": _hyper_ndmath_branch,
            "ndmath.execute": _hyper_ndmath_execute,
            "ndmath.numpy_method.absolute": _hyper_numpy_abs,
            "ndmath.numpy_method.conjugate": _hyper_numpy_conjugate,
            "ndmath.numpy_method.amax": _hyper_numpy_max,
            "ndmath.numpy_method.amin": _hyper_numpy_min,
            "display.array_values": _hyper_display_array_values,
            "display.complex_dim_flags": _hyper_display_complex_dim_flags,
        }
