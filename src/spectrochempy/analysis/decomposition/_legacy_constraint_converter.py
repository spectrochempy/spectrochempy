# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Private converter from legacy MCRALS traitlet configuration to public Constraint objects.

This module translates the existing (traitlet-based) estimator parameters of
:class:`~spectrochempy.analysis.decomposition.mcrals.MCRALS` into instances
of the public constraint classes defined in
:mod:`spectrochempy.analysis.decomposition.mcrals_constraints`.

The converter is the bridge between the old traitlet-based API and the new
``constraints=`` API.  It is called automatically by ``MCRALS._fit`` when
``self.constraints is None`` (i.e. the user has not provided explicit
constraint objects).  The resulting public constraint list is then processed
identically to a user-provided list through ``_public_to_internal``, so all
internal code paths are the same regardless of which API the user chose.

Typical usage::

    from spectrochempy.analysis.decomposition._legacy_constraint_converter import (
        legacy_to_constraints,
    )

    mcr = MCRALS(...)
    constraints = legacy_to_constraints(mcr)
"""

__all__ = ["legacy_to_constraints"]

from spectrochempy.analysis.decomposition.mcrals_constraints import Closure
from spectrochempy.analysis.decomposition.mcrals_constraints import ModelProfile
from spectrochempy.analysis.decomposition.mcrals_constraints import Monotonic
from spectrochempy.analysis.decomposition.mcrals_constraints import NonNegative
from spectrochempy.analysis.decomposition.mcrals_constraints import Unimodal

# ---------------------------------------------------------------------------------------
# Sentinel used to indicate "all components".
# ---------------------------------------------------------------------------------------
_ALL = object()


def legacy_to_constraints(estimator):
    """
    Convert legacy MCRALS traitlets to public Constraint objects.

    Parameters
    ----------
    estimator : MCRALS
        The MCRALS estimator instance whose traitlets will be read.

    Returns
    -------
    list[Constraint]
        List of public constraint objects representing the current
        estimator configuration.

    Notes
    -----
    The returned list is ordered by profile side (concentration first,
    then spectral).  Within each side the order matches the historical
    constraint application order used in the internal engine:
    non-negativity → unimodality → monotonicity → closure → model-based
    (hard-profile) constraints.

    Normalization (``normSpec``) is a joint operation on both ``C`` and
    ``St`` and has no corresponding public constraint class at this time;
    it is not emitted by this converter.
    """
    constraints = []
    constraints.extend(_conc_constraints(estimator))
    constraints.extend(_spec_constraints(estimator))
    return constraints


# ---------------------------------------------------------------------------------------
# Concentration-side converters
# ---------------------------------------------------------------------------------------


def _conc_constraints(estimator):
    """Build the list of public constraints for the concentration side."""
    result = []

    # nonnegConc -> NonNegative("C", ...)
    nn = _to_components(estimator.nonnegConc)
    if nn is _ALL:
        result.append(NonNegative("C"))
    elif nn is not None:
        result.append(NonNegative("C", components=nn))

    # unimodConc + unimodConcMod + unimodConcTol -> Unimodal("C", ...)
    um = _to_components(estimator.unimodConc)
    if um is _ALL:
        result.append(
            Unimodal(
                "C",
                mod=_str_or_default(estimator.unimodConcMod, "strict"),
                tolerance=estimator.unimodConcTol,
            )
        )
    elif um is not None:
        result.append(
            Unimodal(
                "C",
                components=um,
                mod=_str_or_default(estimator.unimodConcMod, "strict"),
                tolerance=estimator.unimodConcTol,
            )
        )

    # monoIncConc -> Monotonic("C", "increasing", ...)
    mi = _to_components(estimator.monoIncConc)
    if mi is _ALL:
        result.append(Monotonic("C", "increasing", tolerance=estimator.monoIncTol))
    elif mi is not None:
        result.append(
            Monotonic(
                "C",
                "increasing",
                components=mi,
                tolerance=estimator.monoIncTol,
            )
        )

    # monoDecConc -> Monotonic("C", "decreasing", ...)
    md = _to_components(estimator.monoDecConc)
    if md is _ALL:
        result.append(Monotonic("C", "decreasing", tolerance=estimator.monoDecTol))
    elif md is not None:
        result.append(
            Monotonic(
                "C",
                "decreasing",
                components=md,
                tolerance=estimator.monoDecTol,
            )
        )

    # closureConc -> Closure("C", ...)
    cc = _to_components(estimator.closureConc)
    closure_method = getattr(estimator, "closureMethod", "scaling")
    if cc is _ALL:
        result.append(
            Closure(
                "C", target=_extract_closure_target(estimator), method=closure_method
            )
        )
    elif cc is not None:
        result.append(
            Closure(
                "C",
                components=cc,
                target=_extract_closure_target(estimator),
                method=closure_method,
            )
        )

    # hardConc + getConc + argsGetConc + kwargsGetConc -> ModelProfile("C", ...)
    hc = _to_components(estimator.hardConc)
    if hc is not None and callable(estimator.getConc):
        mp_kw = {
            "model": estimator.getConc,
            "model_args": estimator.argsGetConc,
            "model_kwargs": estimator.kwargsGetConc,
        }
        if hc is not _ALL:
            mp_kw["components"] = hc
            comps_for_mapping = hc
        else:
            # hardConc=="all": infer component range from legacy mapping length
            legacy_map = estimator.getC_to_C_idx
            if legacy_map != "default" and legacy_map is not None:
                comps_for_mapping = list(range(len(legacy_map)))
            else:
                comps_for_mapping = []
        mp_kw["mapping"] = _convert_legacy_mapping(
            estimator.getC_to_C_idx, comps_for_mapping
        )
        result.append(ModelProfile("C", **mp_kw))

    return result


# ---------------------------------------------------------------------------------------
# Spectral-side converters
# ---------------------------------------------------------------------------------------


def _spec_constraints(estimator):
    """Build the list of public constraints for the spectral side."""
    result = []

    # nonnegSpec -> NonNegative("St", ...)
    nn = _to_components(estimator.nonnegSpec)
    if nn is _ALL:
        result.append(NonNegative("St"))
    elif nn is not None:
        result.append(NonNegative("St", components=nn))

    # unimodSpec + unimodSpecMod + unimodSpecTol -> Unimodal("St", ...)
    um = _to_components(estimator.unimodSpec)
    if um is _ALL:
        result.append(
            Unimodal(
                "St",
                mod=_str_or_default(estimator.unimodSpecMod, "strict"),
                tolerance=estimator.unimodSpecTol,
            )
        )
    elif um is not None:
        result.append(
            Unimodal(
                "St",
                components=um,
                mod=_str_or_default(estimator.unimodSpecMod, "strict"),
                tolerance=estimator.unimodSpecTol,
            )
        )

    # hardSpec + getSpec + argsGetSpec + kwargsGetSpec -> ModelProfile("St", ...)
    hs = _to_components(estimator.hardSpec)
    if hs is not None and callable(estimator.getSpec):
        mp_kw = {
            "model": estimator.getSpec,
            "model_args": estimator.argsGetSpec,
            "model_kwargs": estimator.kwargsGetSpec,
        }
        if hs is not _ALL:
            mp_kw["components"] = hs
            comps_for_mapping = hs
        else:
            legacy_map = estimator.getSt_to_St_idx
            if legacy_map != "default" and legacy_map is not None:
                comps_for_mapping = list(range(len(legacy_map)))
            else:
                comps_for_mapping = []
        mp_kw["mapping"] = _convert_legacy_mapping(
            estimator.getSt_to_St_idx, comps_for_mapping
        )
        result.append(ModelProfile("St", **mp_kw))

    return result


# ---------------------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------------------


def _to_components(value):
    """
    Normalise a legacy traitlet value.

    Parameters
    ----------
    value : str or list or ndarray or None
        The raw traitlet value.

    Returns
    -------
    _ALL, list[int], or None
        * ``_ALL`` means "the constraint applies to all components"
          (call the public constructor without ``components``).
        * ``list[int]`` means "the constraint applies to the listed components".
        * ``None`` means the constraint is inactive and should not be emitted.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _ALL if value == "all" else None
    # list, tuple, or ndarray
    try:
        lst = list(value)
    except TypeError:
        return None
    if not lst:
        return None
    return [int(i) for i in lst]


def _extract_closure_target(estimator):
    """
    Extract the closure target from the legacy ``closureTarget`` traitlet.

    The legacy ``closureTarget`` is either the string ``"default"`` (which
    resolves to an array of ones at fit time) or an array of per-observation
    target values.  The public :class:`Closure` constraint accepts both
    scalars and array-like targets, so this helper passes array-like values
    through directly.

    Parameters
    ----------
    estimator : MCRALS
        The estimator instance.

    Returns
    -------
    float or array-like
        The closure target.  ``"default"`` is mapped to the scalar ``1.0``.
    """
    target = estimator.closureTarget
    if isinstance(target, str):
        return 1.0  # "default"
    return target


def _convert_legacy_mapping(legacy_mapping, components):
    """
    Convert legacy ``getC_to_C_idx`` / ``getSt_to_St_idx`` to public mapping.

    Legacy format:
      ``legacy_mapping[i]`` → ALS component index that model output
      column/row *i* maps to, or ``None`` (model output is ignored).

    Public format:
      ``mapping[j]`` → model output column/row index assigned to ALS
      component ``components[j]``, or ``None`` (component keeps ALS
      estimate and is not replaced).

    Parameters
    ----------
    legacy_mapping : str or list or None
        ``"default"`` (identity), ``None`` (identity), or a list of
        int-or-``None`` entries.
    components : list[int]
        The resolved list of ALS component indices that the hard
        constraint applies to.

    Returns
    -------
    list or None
        Mapping in the public format, or ``None`` for identity.
    """
    if legacy_mapping == "default" or legacy_mapping is None:
        return None
    if not components:
        return None
    mapping = [None] * len(components)
    for model_col, als_idx in enumerate(legacy_mapping):
        if als_idx is not None and als_idx in components:
            j = components.index(als_idx)
            mapping[j] = model_col
    if all(m is None for m in mapping):
        return None
    if mapping == list(range(len(components))):
        return None
    return mapping


def _str_or_default(value, default):
    """
    Return ``value`` if it is a string, otherwise ``default``.

    Some traitlet validators (e.g. ``unimodConcMod``) are enums that are
    always a string.  This helper handles the general case where the
    value might be a non-string before the validator has run.

    Parameters
    ----------
    value : object
        The traitlet value.
    default : str
        Default to return if ``value`` is not a string.

    Returns
    -------
    str
    """
    if isinstance(value, str):
        return value
    return default
