# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Public MCRALS constraint classes.

This module introduces the **public constraint API** for ``MCRALS``.

Each class represents a single piece of scientific prior knowledge about
the concentration (``"C"``) or spectral (``"St"``) profiles that
``MCRALS`` estimates. The classes are deliberately *declarative*: they
describe **what is known**, not **how** the knowledge is incorporated
into the ALS optimisation.

.. important::

    This is a **skeleton** API. The classes in this module are data
    containers and validators only. They are **not yet connected** to
    the internal constraint engine used by :class:`MCRALS`, and using
    them does not change the behaviour of ``MCRALS.fit``. Connecting
    the public API to the internal engine, the legacy traitlet
    converter, and the actual enforcement implementations are the
    subject of subsequent PRs.

The canonical profile identifiers are the strings ``"C"`` (concentrations)
and ``"St"`` (spectra). Every constraint validates its ``profile``
argument against this set. The public vocabulary is deliberately
generic: a profile is either a concentration or a spectrum, and each
constraint class is the same regardless of which side it targets — the
``profile`` argument alone identifies the constrained object. There is
therefore a single :class:`ReferenceProfile` for both concentration and
spectral references, and a single :class:`ModelProfile` for both
concentration and spectral model-based (profile-generator) constraints.

Example::

    from spectrochempy import NonNegative, Closure, ReferenceProfile, ProfileModel

    constraints = [
        NonNegative("C"),
        Closure("C"),
        ReferenceProfile(
            "St",
            component=0,
            data=reference_spectrum,
        ),
        ModelProfile(
            "C",
            components=[0, 1],
            model=my_model,
        ),
    ]

See the project RFC (``spectrochempy_maintainer/rfcs/``) for the full
design rationale and the planned migration path.
"""

# DEVNOTE:
# API methods accessible as  scp.method or scp.class must be defined in __all__
# Configurable class (requires a configuration file) must be declared in
# __configurables__. None of the constraint classes are configurable.

__all__ = [
    "Constraint",
    "NonNegative",
    "Closure",
    "Unimodal",
    "Monotonic",
    "ZeroRegion",
    "Selectivity",
    "FixedValues",
    "ReferenceProfile",
    "ModelProfile",
]

import numpy as np

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

#: Canonical profile identifiers accepted by every constraint.
_PROFILES = ("C", "St")

#: Admissible monotonic directions.
_DIRECTIONS = ("increasing", "decreasing")

#: Admissible unimodal modalities. ``"strict"`` enforces a single
#: maximum, ``"smooth"`` allows a flat-topped region (mirrors the
#: historical ``unimodConcMod`` / ``unimodSpecMod`` traitlets).
_UNIMODAL_MODS = ("strict", "smooth")


def _validate_profile(profile):
    """
    Validate the ``profile`` argument.

    Parameters
    ----------
    profile : str
        Canonical profile identifier. Must be ``"C"`` (concentrations) or
        ``"St"`` (spectra).

    Raises
    ------
    TypeError
        If ``profile`` is not a string.
    ValueError
        If ``profile`` is not one of the canonical identifiers.

    Returns
    -------
    str
        The validated profile identifier.
    """
    if not isinstance(profile, str):
        raise TypeError(f"profile must be a string, got {type(profile).__name__!r}.")
    if profile not in _PROFILES:
        raise ValueError(f"profile must be one of {_PROFILES!r}, got {profile!r}.")
    return profile


def _validate_components(components, *, name="components", allow_none=True):
    """
    Validate a component-selection list.

    A component selection is either ``None`` (meaning "all components",
    the default for most constraints) or a list of non-negative integer
    indices. Indices may be in any order and need not be contiguous.

    Parameters
    ----------
    components : None or list[int]
        Component selection.
    name : str, optional
        Argument name used in error messages.
    allow_none : bool, optional
        If ``True`` (default), ``None`` is accepted and means "all".

    Raises
    ------
    TypeError
        If the selection is not a list or contains non-integer entries.
    ValueError
        If the list is empty or contains negative indices.

    Returns
    -------
    list[int] or None
        The validated selection as a list of ints, or ``None``.
    """
    if components is None:
        if not allow_none:
            raise ValueError(f"{name} must be a non-empty list of integers.")
        return None
    if isinstance(components, (int,)):
        raise TypeError(
            f"{name} must be a list of integers, got a single int. "
            f"Use [{components}] to select a single component."
        )
    if not isinstance(components, (list, tuple)):
        raise TypeError(
            f"{name} must be a list of integers, got {type(components).__name__!r}."
        )
    out = []
    for idx in components:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(
                f"{name} entries must be integers, got "
                f"{type(idx).__name__!r} ({idx!r})."
            )
        if idx < 0:
            raise ValueError(f"{name} entries must be non-negative, got {idx!r}.")
        out.append(idx)
    if not out:
        raise ValueError(f"{name} must not be empty.")
    return out


def _validate_component(component, *, name="component"):
    """
    Validate a single component index.

    Parameters
    ----------
    component : int
        Non-negative integer index.
    name : str, optional
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``component`` is not an integer.
    ValueError
        If ``component`` is negative.

    Returns
    -------
    int
        The validated component index.
    """
    if isinstance(component, bool) or not isinstance(component, int):
        raise TypeError(f"{name} must be an integer, got {type(component).__name__!r}.")
    if component < 0:
        raise ValueError(f"{name} must be non-negative, got {component!r}.")
    return component


def _validate_tolerance(tolerance, *, name="tolerance"):
    """
    Validate a tolerance parameter.

    Tolerances are expected to be real numbers ``>= 1.0``. This mirrors
    the historical ``monoIncTol`` / ``monoDecTol`` semantics where a
    value of ``1.0`` makes the constraint a strict bound and a value
    above ``1.0`` allows small local violations.

    Parameters
    ----------
    tolerance : float
        Tolerance value.
    name : str, optional
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``tolerance`` is not a real number.
    ValueError
        If ``tolerance`` is less than ``1.0``.

    Returns
    -------
    float
        The validated tolerance as a float.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError(
            f"{name} must be a real number, got {type(tolerance).__name__!r}."
        )
    tol = float(tolerance)
    if tol < 1.0:
        raise ValueError(f"{name} must be >= 1.0 to remain admissible, got {tol!r}.")
    return tol


def _validate_target(target, *, name="target"):
    """
    Validate a closure target.

    The target is either:

    * a positive scalar number that the selected components should sum to
      at every row (e.g. ``1.0`` for unit-sum closure), or
    * an array-like of per-row target values.

    A zero scalar target would imply all selected components are
    identically zero, which is not a meaningful closure and is rejected.
    Array-like targets are validated only for being an admissible
    array-like — shape and positivity checks are deferred to the
    enforcement engine.

    Parameters
    ----------
    target : float or array-like
        Closure target.  A scalar must be strictly positive.  An
        array-like is passed through without copying or shape validation.
    name : str, optional
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``target`` is a boolean, a string, ``None``, or another
        non-numeric, non-iterable object.
    ValueError
        If ``target`` is a scalar that is not strictly positive.

    Returns
    -------
    float or object
        The validated target.  Scalars are returned as ``float``;
        array-likes are returned unchanged (no copy).
    """
    if isinstance(target, bool):
        raise TypeError(f"{name} must be a numeric scalar or array-like, got bool.")
    # Scalar case: must be a positive number.
    if isinstance(target, (int, float)):
        t = float(target)
        if t <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {t!r}.")
        return t
    # Array-like case: delegate to the generic array-like validator.
    # None, strings, and scalars are already rejected above.
    return _validate_array_like(target, name=name)


def _targets_equal(a, b):
    """
    Compare two closure-target values, handling array-likes correctly.

    Scalars are compared with ``==``.  Array-likes are compared via
    :func:`numpy.array_equal` so that element-wise comparison yields a
    single boolean (rather than a per-element array that would confuse
    tuple comparison in the base-class ``__eq__``).

    Parameters
    ----------
    a : float or array-like
    b : float or array-like

    Returns
    -------
    bool
    """
    a_is_scalar = isinstance(a, (int, float))
    b_is_scalar = isinstance(b, (int, float))
    if a_is_scalar and b_is_scalar:
        return a == b
    if a_is_scalar != b_is_scalar:
        return False
    # Both are array-like.
    try:
        return bool(np.array_equal(a, b))
    except Exception:  # noqa: BLE001
        return False


def _validate_region(region, *, name="region"):
    """
    Validate a region (index range) selection.

    A region is a 2-tuple ``(start, stop)`` of non-negative integers
    with ``stop > start``. It denotes the contiguous range of indices
    (e.g. wavelengths or observations) over which the constraint
    applies.

    Parameters
    ----------
    region : tuple[int, int]
        ``(start, stop)`` of the region, half-open.
    name : str, optional
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``region`` is not a sequence or contains non-integer entries.
    ValueError
        If the region does not have exactly two non-negative entries
        with ``stop > start``.

    Returns
    -------
    tuple[int, int]
        The validated ``(start, stop)`` pair.
    """
    if region is None:
        raise ValueError(f"{name} must be a (start, stop) pair, got None.")
    if isinstance(region, (list, tuple)):
        if len(region) != 2:
            raise ValueError(
                f"{name} must have exactly two entries (start, stop), "
                f"got {len(region)}."
            )
        out = []
        for v in region:
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(
                    f"{name} entries must be integers, got "
                    f"{type(v).__name__!r} ({v!r})."
                )
            out.append(v)
        start, stop = out
        if start < 0:
            raise ValueError(f"{name} start must be non-negative, got {start!r}.")
        if stop <= start:
            raise ValueError(
                f"{name} stop must be greater than start, got ({start!r}, {stop!r})."
            )
        return start, stop
    raise TypeError(
        f"{name} must be a list or tuple of two integers, got "
        f"{type(region).__name__!r}."
    )


def _validate_unimodal_mod(mod, *, name="mod"):
    """
    Validate the unimodal modality.

    Parameters
    ----------
    mod : str
        Either ``"strict"`` (single maximum) or ``"smooth"`` (allows a
        flat-topped region).
    name : str, optional
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``mod`` is not a string.
    ValueError
        If ``mod`` is not one of the admissible values.

    Returns
    -------
    str
        The validated modality.
    """
    if not isinstance(mod, str):
        raise TypeError(f"{name} must be a string, got {type(mod).__name__!r}.")
    if mod not in _UNIMODAL_MODS:
        raise ValueError(f"{name} must be one of {_UNIMODAL_MODS!r}, got {mod!r}.")
    return mod


def _validate_direction(direction, *, name="direction"):
    """
    Validate the monotonicity direction.

    Parameters
    ----------
    direction : str
        Either ``"increasing"`` or ``"decreasing"``.
    name : str, optional
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``direction`` is not a string.
    ValueError
        If ``direction`` is not one of the admissible values.

    Returns
    -------
    str
        The validated direction.
    """
    if not isinstance(direction, str):
        raise TypeError(f"{name} must be a string, got {type(direction).__name__!r}.")
    if direction not in _DIRECTIONS:
        raise ValueError(f"{name} must be one of {_DIRECTIONS!r}, got {direction!r}.")
    return direction


def _validate_array_like(values, *, name):
    """
    Validate that an object can be interpreted as an array-like.

    Accepted inputs are any object that carries profile-shaped numeric
    data without being a scalar. The following are accepted in
    particular:

    - ``numpy.ndarray``
    - ``spectrochempy.NDDataset``
    - ``spectrochempy.Coord``
    - ``list``
    - ``tuple``
    - any other object exposing ``__iter__`` (a generator is consumed
      only far enough to be recognised as iterable — see Notes).

    No copy is made and no particular numeric container is enforced.
    Shape compatibility with the constrained profile subset is
    validated lazily, at enforcement time, by the ALS engine; the
    skeleton API does not know the number of components or the data
    shape and so deliberately does not check it here.

    Parameters
    ----------
    values : object
        Candidate array-like.
    name : str
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``values`` is ``None``, a scalar (``bool`` / ``int`` /
        ``float``), or not iterable.

    Returns
    -------
    object
        The passed object, unchanged. No copy is made; the enforcement
        engine will later perform any necessary conversion.

    Notes
    -----
    Strings are rejected even though they are technically iterable,
    because they are a common mistake (e.g. passing a column name
    instead of the data) and never a meaningful profile.
    """
    if values is None:
        raise TypeError(f"{name} must be an array-like, got None.")
    if isinstance(values, str):
        raise TypeError(f"{name} must be an array-like of numbers, got a string.")
    if isinstance(values, (bool, int, float)):
        raise TypeError(
            f"{name} must be an array-like, got a scalar ({type(values).__name__!r})."
        )
    # Explicit allow-list of the common array-likes consumed by users.
    # NDDataset / Coord are imported lazily to avoid a hard dependency
    # cycle at module import time.
    try:
        from spectrochempy.core.dataset.nddataset import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        if isinstance(values, (Coord, NDDataset)):
            return values
    except ImportError:  # pragma: no cover - core is always importable
        pass
    if isinstance(values, (list, tuple, np.ndarray)):
        return values
    # Fall back to a generic iterable check: any object exposing
    # ``__iter__`` (e.g. a generator, a pandas Series, a range) is
    # accepted. Iterating just to check would be wasteful, so we rely
    # on ``__iter__`` presence rather than calling ``iter(values)``.
    if hasattr(values, "__iter__"):
        return values
    raise TypeError(
        f"{name} must be an array-like (list, tuple, numpy array, "
        f"NDDataset, Coord, or any iterable of numbers), got "
        f"{type(values).__name__!r}."
    )


def _validate_callable(model, *, name="model"):
    """
    Validate that an object is callable.

    Parameters
    ----------
    model : object
        Candidate callable.
    name : str, optional
        Argument name used in error messages.

    Raises
    ------
    TypeError
        If ``model`` is not callable.

    Returns
    -------
    object
        The passed callable.
    """
    if not callable(model):
        raise TypeError(f"{name} must be callable, got {type(model).__name__!r}.")
    return model


# --------------------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------------------


class Constraint:
    """
    Abstract base class for MCRALS constraints.

    A constraint represents a single piece of scientific prior knowledge
    about the concentration (``"C"``) or spectral (``"St"``) profiles
    estimated by :class:`spectrochempy.MCRALS`.

    Every public constraint subclasses this base and accepts ``profile``
    as its first positional argument. The base class validates the
    profile identifier and provides a uniform ``repr`` and equality
    protocol so that constraints can be compared and inspected
    consistently.

    This base is **not** itself instantiable as a usable constraint: it
    carries no scientific meaning. Subclass it to define a new
    constraint family; do not subclass ``Constraint`` directly when
    defining a new scientific concept — pick or introduce a dedicated
    subclass instead.

    Parameters
    ----------
    profile : str
        Canonical profile identifier. Must be ``"C"`` (concentrations)
        or ``"St"`` (spectra).

    Raises
    ------
    TypeError
        If ``profile`` is not a string.
    ValueError
        If ``profile`` is not one of the canonical identifiers.

    Notes
    -----
    Constraint objects are *declarative*. They store the user intent
    and validate it; they do not perform any numerical computation. The
    actual enforcement (projection, regularised least squares, profile
    generation, ...) is the responsibility of the internal ALS engine
    and is added in subsequent PRs.

    See Also
    --------
    NonNegative : Non-negativity constraint.
    Closure : Closure (constant sum) constraint.
    Unimodal : Unimodality constraint.
    Monotonic : Monotonicity constraint.
    """

    def __init__(self, profile):
        self._profile = _validate_profile(profile)

    # -- properties ------------------------------------------------------
    @property
    def profile(self):
        """str: Canonical profile identifier (``"C"`` or ``"St"``)."""
        return self._profile

    # -- representation --------------------------------------------------
    def _repr_params(self):
        """
        Return a list of ``(name, value)`` pairs for ``__repr__``.

        Subclasses override this to expose their *public* parameters
        (never leading underscores). The base implementation exposes
        only ``profile``.
        """
        return [("profile", self._profile)]

    def __repr__(self):
        params = ", ".join(f"{name}={value!r}" for name, value in self._repr_params())
        return f"{type(self).__name__}({params})"

    # -- equality --------------------------------------------------------
    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self._public_state() == other._public_state()

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    # __eq__ is defined, so __hash__ defaults to None (unhashable). This
    # is intentional: constraint objects hold mutable collections and are
    # not meant to participate in dicts/sets at the skeleton stage.

    def _public_state(self):
        """
        Return a tuple of the public parameter values, in canonical order.

        Used by ``__eq__``. The default returns the values from
        ``_repr_params`` in the same order. Subclasses with non-hashable
        parameters (numpy arrays) should override this if they need a
        comparison that acknowledges array shape but not exact float
        equality (the default implementation below handles arrays via
        ``numpy.array_equal`` through ``__eq__`` on the underlying
        object — which is fine for the skeleton).
        """
        return tuple(value for _, value in self._repr_params())

    # -- introspection ---------------------------------------------------
    @property
    def name(self):
        """str: Short human-readable name of the constraint family."""
        return type(self).__name__


# --------------------------------------------------------------------------------------
# Projection constraints
# --------------------------------------------------------------------------------------


class NonNegative(Constraint):
    """
    Non-negativity constraint.

    States that the selected components of a profile must remain
    non-negative. This is the most common MCR-ALS constraint for both
    concentrations (positive absorptivities / amounts) and spectra
    (positive absorptivities).

    Parameters
    ----------
    profile : str
        ``"C"`` (concentrations) or ``"St"`` (spectra).
    components : list[int], optional
        Component indices to which the constraint applies. ``None``
        (default) means "all components".

    Examples
    --------
    >>> from spectrochempy import NonNegative
    >>> NonNegative("C")
    NonNegative(profile='C', components=None)
    >>> NonNegative("St", components=[0, 2])
    NonNegative(profile='St', components=[0, 2])
    """

    def __init__(self, profile, components=None):
        super().__init__(profile)
        self._components = _validate_components(components)

    def _repr_params(self):
        return [("profile", self._profile), ("components", self._components)]

    @property
    def components(self):
        """list[int] or None: Component selection (``None`` means "all")."""
        return self._components


class Closure(Constraint):
    """
    Closure (constant sum) constraint.

    States that the selected components of a profile must sum to a
    target value across the constrained axis (typically the
    concentration rows summing to ``1.0``).

    Parameters
    ----------
    profile : str
        ``"C"`` (concentrations) or ``"St"`` (spectra).
    components : list[int], optional
        Component indices included in the closure. ``None`` (default)
        means "all components".
    target : float or array-like, optional
        Target sum for the selected components.  A scalar is applied
        to every row; an array-like supplies one target per row.
        Default is ``1.0``.

    Examples
    --------
    >>> from spectrochempy import Closure
    >>> Closure("C")
    Closure(profile='C', components=None, target=1.0)
    >>> Closure("C", components=[0, 1], target=100.0)
    Closure(profile='C', components=[0, 1], target=100.0)
    >>> import numpy as np
    >>> Closure("C", target=np.array([1.0, 1.0, 1.0]))
    Closure(profile='C', components=None, target=array([1., 1., 1.]))
    """

    def __init__(self, profile, components=None, target=1.0):
        super().__init__(profile)
        self._components = _validate_components(components)
        self._target = _validate_target(target)

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("components", self._components),
            ("target", self._target),
        ]

    @property
    def components(self):
        """list[int] or None: Components in the closure (``None`` = all)."""
        return self._components

    @property
    def target(self):
        """Float or array-like: Target sum for the selected components."""
        return self._target

    # -- equality --------------------------------------------------------
    # Override base-class ``__eq__`` so that array-valued targets are
    # compared element-wise via ``np.array_equal`` rather than through
    # the default tuple-element comparison (which would fail when one
    # element is a numpy array).

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        if self._profile != other._profile:
            return False
        if self._components != other._components:
            return False
        return _targets_equal(self._target, other._target)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


class Unimodal(Constraint):
    """
    Unimodality constraint.

    States that each selected component has a single maximum along the
    constrained axis. Useful for chromatographic / kinetic profiles
    where one peak is expected per chemical species.

    Parameters
    ----------
    profile : str
        ``"C"`` (concentrations) or ``"St"`` (spectra).
    components : list[int], optional
        Component indices to which the constraint applies. ``None``
        (default) means "all components".
    mod : str, optional
        Unimodal modality: ``"strict"`` (single maximum, default) or
        ``"smooth"`` (allow a flat-topped region).

    Examples
    --------
    >>> from spectrochempy import Unimodal
    >>> Unimodal("C")
    Unimodal(profile='C', components=None, mod='strict')
    >>> Unimodal("St", components=[0], mod="smooth")
    Unimodal(profile='St', components=[0], mod='smooth')
    """

    def __init__(self, profile, components=None, mod="strict"):
        super().__init__(profile)
        self._components = _validate_components(components)
        self._mod = _validate_unimodal_mod(mod)

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("components", self._components),
            ("mod", self._mod),
        ]

    @property
    def components(self):
        """list[int] or None: Component selection (``None`` means "all")."""
        return self._components

    @property
    def mod(self):
        """str: Unimodal modality (``"strict"`` or ``"smooth"``)."""
        return self._mod


class Monotonic(Constraint):
    """
    Monotonicity constraint.

    States that each selected component must be monotonically increasing
    or decreasing along the constrained axis (typically the
    observation/time axis for concentrations).

    Parameters
    ----------
    profile : str
        ``"C"`` (concentrations) or ``"St"`` (spectra).
    direction : str
        Either ``"increasing"`` or ``"decreasing"``.
    components : list[int], optional
        Component indices to which the constraint applies. ``None``
        (default) means "all components".
    tolerance : float, optional
        Admissibility tolerance. Must be ``>= 1.0``. ``1.0`` makes the
        constraint strict; values above ``1.0`` allow small local
        violations (mirrors the historical ``monoIncTol`` / ``monoDecTol``
        semantics). Default is ``1.1``.

    Examples
    --------
    >>> from spectrochempy import Monotonic
    >>> Monotonic("C", "increasing")
    Monotonic(profile='C', direction='increasing', components=None, tolerance=1.1)
    >>> Monotonic("C", "decreasing", components=[0], tolerance=1.0)
    Monotonic(profile='C', direction='decreasing', components=[0], tolerance=1.0)
    """

    def __init__(
        self,
        profile,
        direction,
        components=None,
        tolerance=1.1,
    ):
        super().__init__(profile)
        self._direction = _validate_direction(direction)
        self._components = _validate_components(components)
        self._tolerance = _validate_tolerance(tolerance)

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("direction", self._direction),
            ("components", self._components),
            ("tolerance", self._tolerance),
        ]

    @property
    def direction(self):
        """str: Monotonicity direction (``"increasing"`` or ``"decreasing"``)."""
        return self._direction

    @property
    def components(self):
        """list[int] or None: Component selection (``None`` means "all")."""
        return self._components

    @property
    def tolerance(self):
        """float: Admissibility tolerance (``>= 1.0``)."""
        return self._tolerance


class ZeroRegion(Constraint):
    """
    Zero-region constraint.

    States that the selected components must be exactly zero on a given
    contiguous region of the constrained axis (e.g. a component known to
    be absent before a given observation index or outside a given
    spectral window).

    Parameters
    ----------
    profile : str
        ``"C"`` (concentrations) or ``"St"`` (spectra).
    region : tuple[int, int]
        ``(start, stop)`` of the half-open region over which the
        selected components must be zero. ``stop`` must be strictly
        greater than ``start``.
    components : list[int], optional
        Component indices to which the constraint applies. ``None``
        (default) means "all components".

    Examples
    --------
    >>> from spectrochempy import ZeroRegion
    >>> ZeroRegion("C", region=(0, 5))
    ZeroRegion(profile='C', region=(0, 5), components=None)
    >>> ZeroRegion("St", region=(40, 60), components=[1])
    ZeroRegion(profile='St', region=(40, 60), components=[1])
    """

    def __init__(self, profile, region, components=None):
        super().__init__(profile)
        self._region = _validate_region(region)
        self._components = _validate_components(components)

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("region", self._region),
            ("components", self._components),
        ]

    @property
    def region(self):
        """tuple[int, int]: Half-open ``(start, stop)`` region."""
        return self._region

    @property
    def components(self):
        """list[int] or None: Component selection (``None`` means "all")."""
        return self._components


class Selectivity(Constraint):
    """
    Selectivity constraint.

    States that each selected component is the only one present on a
    given region of the constrained axis. This encodes the
    *unimodality-of-presence* assumption used in many hard-modeling
    workflows (e.g. one species dominates a portion of the elution
    window).

    Parameters
    ----------
    profile : str
        ``"C"`` (concentrations) or ``"St"`` (spectra).
    region : tuple[int, int]
        ``(start, stop)`` of the half-open region on which the
        constraint applies. ``stop`` must be strictly greater than
        ``start``.
    component : int
        Index of the single component that must be the *sole* component
        on the region.

    Examples
    --------
    >>> from spectrochempy import Selectivity
    >>> Selectivity("C", region=(0, 5), component=0)
    Selectivity(profile='C', region=(0, 5), component=0)
    """

    def __init__(self, profile, region, component):
        super().__init__(profile)
        self._region = _validate_region(region)
        self._component = _validate_component(component)

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("region", self._region),
            ("component", self._component),
        ]

    @property
    def region(self):
        """tuple[int, int]: Half-open ``(start, stop)`` region."""
        return self._region

    @property
    def component(self):
        """int: Index of the sole component present on the region."""
        return self._component


# --------------------------------------------------------------------------------------
# Reference constraints
# --------------------------------------------------------------------------------------


class FixedValues(Constraint):
    """
    Fixed-values constraint.

    States that selected components of a profile must take pre-specified
    fixed values (e.g. a measured spectrum known to be noise-free, or a
    known concentration profile on a subset of observations).

    Parameters
    ----------
    profile : str
        ``"C"`` (concentrations) or ``"St"`` (spectra).
    values : array-like
        The fixed values to impose. The shape must be compatible with
        the constrained profile subset at enforcement time (validated
        lazily; the skeleton API does not know the number of components
        or the data shape).
    components : list[int], optional
        Component indices to which the fixed values apply. ``None``
        (default) means "all components".

    Examples
    --------
    >>> from spectrochempy import FixedValues
    >>> FixedValues("St", values=[[0.1, 0.2], [0.3, 0.4]])
    FixedValues(profile='St', values=[[0.1, 0.2], [0.3, 0.4]], components=None)
    """

    def __init__(self, profile, values, components=None):
        super().__init__(profile)
        self._values = _validate_array_like(values, name="values")
        self._components = _validate_components(components)

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("values", self._values),
            ("components", self._components),
        ]

    @property
    def values(self):
        """array-like: The fixed values to impose."""
        return self._values

    @property
    def components(self):
        """list[int] or None: Component selection (``None`` means "all")."""
        return self._components


class ReferenceProfile(Constraint):
    """
    Reference-profile constraint.

    States that a selected component profile must remain close to
    a known reference (e.g. a measured kinetic trace for concentrations,
    or a pure-component spectrum measured independently). The actual
    enforcement (projection vs. regularised least squares) is chosen by
    the engine; this object only carries the intent.

    Parameters
    ----------
    profile : str
        Must be ``"C"`` (concentrations) or ``"St"`` (spectra).
    component : int
        Index of the component the reference applies to.
    data : array-like
        The reference profile, as a 1-D array-like.

    Examples
    --------
    >>> from spectrochempy import ReferenceProfile
    >>> ReferenceProfile("C", component=0, data=[0.1, 0.2, 0.7, 0.5])
    ReferenceProfile(profile='C', component=0, data=[0.1, 0.2, 0.7, 0.5])
    >>> ReferenceProfile("St", component=1, data=[0.1, 0.9, 0.5, 0.2])
    ReferenceProfile(profile='St', component=1, data=[0.1, 0.9, 0.5, 0.2])
    """

    def __init__(self, profile, component, data):
        super().__init__(profile)
        self._component = _validate_component(component)
        self._data = _validate_array_like(data, name="data")

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("component", self._component),
            ("data", self._data),
        ]

    @property
    def component(self):
        """int: Index of the component the reference applies to."""
        return self._component

    @property
    def data(self):
        """array-like: The reference profile (1-D)."""
        return self._data


# --------------------------------------------------------------------------------------
# Profile generators (model-based constraints)
# --------------------------------------------------------------------------------------


class ModelProfile(Constraint):
    """
    Profile generator constraint.

    States that a profile must lie in the family of profiles generated
    by a user-supplied model. The model is a callable fitted at each ALS
    iteration on the current least-squares profile and used to regenerate
    the constrained profile. This generalises the historical ``getConc``
    and ``getSpec`` mechanisms of :class:`MCRALS`.

    Parameters
    ----------
    profile : str
        Must be ``"C"`` (concentrations) or ``"St"`` (spectra).
    components : list[int], optional
        Component indices to which the model applies. ``None`` (default)
        means "all components".
    model : callable
        A callable that, given the current ALS profile for the selected
        side (``"C"`` or ``"St"``), returns the model-constrained profile.
        Validation is limited to checking that it is callable; signature
        enforcement is deferred to the enforcement engine.

    Examples
    --------
    >>> from spectrochempy import ProfileModel
    >>> def my_model(C):
    ...     return C
    >>> ModelProfile("C", components=[0, 1], model=my_model)
    ModelProfile(profile='C', components=[0, 1], model=<function my_model at ...>)
    >>> ModelProfile("St", components=[0], model=my_model)
    ModelProfile(profile='St', components=[0], model=<function my_model at ...>)
    """

    def __init__(self, profile, components=None, model=None):
        super().__init__(profile)
        self._components = _validate_components(components)
        self._model = _validate_callable(model)

    def _repr_params(self):
        return [
            ("profile", self._profile),
            ("components", self._components),
            ("model", self._model),
        ]

    @property
    def components(self):
        """list[int] or None: Component selection (``None`` means "all")."""
        return self._components

    @property
    def model(self):
        """callable: Model callable used to regenerate the constrained profile."""
        return self._model
