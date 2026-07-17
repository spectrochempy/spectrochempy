# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module that implements the MCRALS class."""
# ruff: noqa: N815

# DEVNOTE:
# API methods accessible as  scp.method or scp.class must be defined in __all__
# Configurable class (which requires a configuration file)
# must be declared in __configurables__

__all__ = ["MCRALS"]
__configurables__ = ["MCRALS"]

import base64
import logging
import warnings
from dataclasses import dataclass
from dataclasses import field

import dill
import numpy as np
import scipy
import traitlets as tr

from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._analysisbase import _wrap_ndarray_output_to_nddataset
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis.decomposition.mcrals_constraints import Constraint
from spectrochempy.application.application import info_
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import signature_has_configurable_traits

# --------------------------------------------------------------------------------------
# Names of legacy traitlet-based constraint parameters.
# Used to detect mixed-API usage (legacy traitlets + new constraints= parameter).
# --------------------------------------------------------------------------------------
_UNSET = object()

_LEGACY_CONSTRAINT_TRAITS = frozenset(
    {
        "nonnegConc",
        "unimodConc",
        "unimodConcMod",
        "unimodConcTol",
        "monoDecConc",
        "monoDecTol",
        "monoIncConc",
        "monoIncTol",
        "closureConc",
        "closureTarget",
        "closureMethod",
        "hardConc",
        "getConc",
        "argsGetConc",
        "kwargsGetConc",
        "getC_to_C_idx",
        "nonnegSpec",
        "normSpec",
        "unimodSpec",
        "unimodSpecMod",
        "unimodSpecTol",
        "hardSpec",
        "getSpec",
        "argsGetSpec",
        "kwargsGetSpec",
        "getSt_to_St_idx",
    }
)


# --------------------------------------------------------------------------------------
# Internal iteration state
# --------------------------------------------------------------------------------------


@dataclass
class _AugmentedStructure:
    """
    Internal description of an augmented (multiset) data structure.

    Two modes are supported:

    * ``mode == "vertical"`` — all matrices share the same number of
      columns (spectral variables) but may have different numbers of rows.
      ``row_slices`` contains one slice per experiment; ``column_slices``
      contains a single slice covering all spectral variables.

    * ``mode == "horizontal"`` — all matrices share the same number of
      rows (observations) but may have different numbers of columns.
      ``row_slices`` contains a single slice covering all rows;
      ``column_slices`` contains one slice per spectral block.

    ``n_row_blocks`` is the number of concentration (C) blocks:
    ``len(row_slices)``.  In vertical mode this equals the number of
    input datasets; in horizontal mode it is always 1 (shared C).

    ``n_column_blocks`` is the number of spectral (St) blocks:
    ``len(column_slices)``.  In horizontal mode this equals the number of
    input datasets; in vertical mode it is always 1 (shared St).

    ``block_presence`` is an optional ``(n_blocks, n_components)`` boolean
    matrix describing which components are physically present in each block.
    It is populated by :class:`_ComponentPresenceConstraint` and consumed
    by inter-block constraints such as :class:`_TrilinearConstraint`.
    ``None`` means all components are present in all blocks.
    """

    mode: str
    row_slices: tuple[slice, ...]
    column_slices: tuple[slice, ...]
    input_shapes: tuple[tuple[int, int], ...]
    block_presence: list[list[bool]] | None = None

    def __post_init__(self):
        if self.mode not in ("vertical", "horizontal"):
            raise ValueError(
                f"Unknown augmentation mode {self.mode!r}. "
                f"Supported modes: 'vertical', 'horizontal'."
            )

    @property
    def n_row_blocks(self) -> int:
        """Number of concentration (C) blocks — ``len(row_slices)``."""
        return len(self.row_slices)

    @property
    def n_column_blocks(self) -> int:
        """Number of spectral (St) blocks — ``len(column_slices)``."""
        return len(self.column_slices)


@dataclass(frozen=True)
class _FactorMetadata:
    """Immutable physical metadata snapshot for one factor or data block."""

    title: str | None = None
    units: object | None = None


@dataclass(frozen=True)
class _FactorMetadataContext:
    """Initial-factor and input metadata captured before array conversion."""

    mode: str | None
    X_blocks: tuple[_FactorMetadata, ...]
    C: _FactorMetadata = _FactorMetadata()
    St: _FactorMetadata = _FactorMetadata()
    C_blocks: tuple[_FactorMetadata, ...] = ()
    St_blocks: tuple[_FactorMetadata, ...] = ()


@dataclass(frozen=True)
class _ResolvedFactorMetadata:
    """Physical metadata assigned to accepted MCRALS factor outputs."""

    C: _FactorMetadata
    St: _FactorMetadata
    C_blocks: tuple[_FactorMetadata, ...]
    St_blocks: tuple[_FactorMetadata, ...]
    scale_is_physically_preserved: bool


@dataclass
class _ALSState:
    """
    Internal iteration state for the MCRALS ALS loop.

    This is a private, purely transient container used by ``MCRALS._fit`` to
    pass arrays and counters between the iteration helpers and the internal
    constraint objects. It is never exposed outside the estimator and
    carries no behavior of its own. The public API still returns plain
    tuples; this object does not leak.
    """

    # mandatory inputs / counters, set once at initialization
    X: np.ndarray
    C: np.ndarray
    St: np.ndarray
    n_components: int
    stdev: float
    residual_change: float
    niter: int
    ndiv: int

    # convergence diagnostics (all changes/errors are relative fractions)
    reconstruction_error: float = np.inf
    profile_change: float = np.inf
    previous_C: np.ndarray | None = None
    previous_St: np.ndarray | None = None
    converged: bool = False
    convergence_reason: str | None = None

    # Augmented data structure (None for simple 2D)
    augmentation: _AugmentedStructure | None = None

    # per-iteration snapshots, refreshed at every ALS step
    C_constrained: np.ndarray | None = None
    C_ls: np.ndarray | None = None
    St_ls: np.ndarray | None = None

    # per-iteration extra outputs returned by external generators
    extra_output_conc: list = field(default_factory=list)
    extra_output_spec: list = field(default_factory=list)

    # optional history of per-iteration profiles (only used when
    # ``storeIterations`` is True)
    C_constrained_list: list = field(default_factory=list)
    C_ls_list: list = field(default_factory=list)
    St_constrained_list: list = field(default_factory=list)
    St_ls_list: list = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Internal constraint engine
#
# The classes in this section are entirely private: they are not exported in
# ``__all__``, not exposed through any ``MCRALS`` attribute, and not part of
# the public API. They simply encapsulate the algorithms previously inlined
# in ``_fit`` / the PR2 helper methods, so that ``_fit`` can iterate over an
# ordered list of constraint objects instead of hardcoding each constraint
# application by name.
#
# The architecture is:
#
#   user input (``constraints= [...]`` or legacy traitlets)
#       |
#       v
#   public ``Constraint`` objects
#       |  (``legacy_to_constraints()`` for the legacy path)
#       v
#   ``_public_to_internal()``
#       |
#       v
#   private ``_Constraint`` objects
#       |  (split by profile side in ``_build_from_public_constraints``)
#       v
#   ALS loop
#
# ``constraints=`` is the canonical public API. Legacy traitlets are
# converted to public ``Constraint`` objects by ``legacy_to_constraints``
# before they reach ``_public_to_internal``.  Internally, both paths produce
# identical ``_Constraint`` pipelines.
#
# ``apply`` returns the (possibly rebound) profile array so that callers can
# keep the constrained reference — this mirrors the historical semantics of
# the PR2 helpers, where ``_unimodal_2D`` could rebind its input to a
# transposed view. Constraints that only mutate in place still return the
# passed array for uniform dispatch (``profile = c.apply(profile, state)``).
#
# No constraint changes the algorithm: each ``apply`` body is a verbatim
# transcription of the corresponding PR2 helper, including guard conditions
# and the in-place vs. copy semantics. ``_ClosureConstraint`` preserves the
# PR1 truthiness fix for issue #911; ``_NormalizationConstraint`` preserves
# the PR1 zero-norm guard (B9).
# --------------------------------------------------------------------------------------


class _Constraint:
    """
    Base class for internal MCRALS constraints.

    Each subclass encapsulates a single constraint (non-negativity,
    unimodality, monotonicity, closure, normalization, profile generation)
    and exposes a uniform ``apply(values, state)`` entry point. ``values`` is
    the current profile array (``C`` for concentrations, ``St`` for
    spectra); ``state`` is the transient ``_ALSState`` for the current
    fit. Implementations return the constrained profile so that callers
    can follow array reassignments (see ``_UnimodalConstraint``).
    """

    #: Short human-readable name used in debug logs / future introspection.
    name: str = "constraint"

    #: Whether this constraint must be applied independently per block
    #: when the data is augmented. ``True`` means the pipeline slices
    #: ``values`` by block before calling ``apply``. ``False`` means the
    #: constraint sees the full concatenated array and manages blocks
    #: itself (e.g. ``_TrilinearConstraint``).
    is_block_local: bool = True

    #: Block indices to which this constraint applies (None = all blocks).
    _blocks: list[int] | None = None

    def apply(self, values, state):  # noqa: D401 - imperative form preferred here
        """
        Return ``values`` with the constraint applied.

        Subclasses must preserve the historical in-place vs. reassignment
        semantics of the corresponding PR2 helper: a constraint that
        historically mutated its input in place may still do so, and one
        that historically reassigned the array (e.g. ``_unimodal_2D``)
        must return the new reference.
        """
        raise NotImplementedError


class _NonNegativeConstraint(_Constraint):
    """
    Clip selected profiles to ``>= 0`` (non-negativity).

    Wraps the historical ``_apply_nonneg_conc`` / ``_apply_nonneg_spec``
    helpers. The selection is a list of profile indexes; ``axis`` selects
    which dimension the indexes address (``0`` for C columns, ``1`` for
    St rows), matching the historical helpers.

    The activation guard uses an explicit truthiness test on the
    selection list rather than ``np.any(...)``, so that selecting only
    component 0 (``[0]``) is honoured — ``np.any([0])`` evaluates to
    ``False`` and would silently disable the constraint. This mirrors the
    PR1 fix for issue #911 applied to ``_ClosureConstraint``.
    """

    name = "non-negative"

    def __init__(self, indices, axis, blocks=None):
        self._indices = indices
        self._axis = axis
        self._blocks = blocks

    def apply(self, values, state):
        if not self._indices:
            return values
        if self._axis == 0:
            values[:, self._indices] = values[:, self._indices].clip(min=0)
        else:
            values[self._indices, :] = values[self._indices, :].clip(min=0)
        return values


class _UnimodalConstraint(_Constraint):
    """
    Enforce unimodality on selected profiles.

    Wraps the historical ``_apply_unimod_conc`` / ``_apply_unimod_spec``
    helpers. ``_unimodal_2D`` may rebind its input to a transposed view,
    which is why ``apply`` returns the (possibly reassigned) array.

    The activation guard uses an explicit truthiness test on the
    selection list rather than ``np.any(...)``, so that selecting only
    component 0 (``[0]``) is honoured. Same fix family as PR1 #911.
    """

    name = "unimodal"

    def __init__(self, indices, axis, tol, mod, blocks=None):
        self._indices = indices
        self._axis = axis
        self._tol = tol
        self._mod = mod
        self._blocks = blocks

    def apply(self, values, state):
        if not self._indices:
            return values
        return _unimodal_2D(
            values,
            idxes=self._indices,
            axis=self._axis,
            tol=self._tol,
            mod=self._mod,
        )


class _MonotonicIncreaseConstraint(_Constraint):
    """
    Force monotonic increase on selected C columns.

    Wraps the historical ``_apply_monoinc_conc`` helper. Operates along
    the observation axis (axis 0 of ``C``); requires the number of
    observations ``ny`` carried on ``values.shape[0]``.

    The activation guard uses an explicit truthiness test on the
    selection list rather than ``np.any(...)``, so that selecting only
    component 0 (``[0]``) is honoured. Same fix family as PR1 #911.
    """

    name = "monotonic-increase"

    def __init__(self, indices, tol, blocks=None):
        self._indices = indices
        self._tol = tol
        self._blocks = blocks

    def apply(self, values, state):
        if not self._indices:
            return values
        ny = values.shape[0]
        for s in self._indices:
            for curid in np.arange(ny - 1):
                if values[curid + 1, s] < values[curid, s] / self._tol:
                    values[curid + 1, s] = values[curid, s]
        return values


class _MonotonicDecreaseConstraint(_Constraint):
    """
    Force monotonic decrease on selected C columns.

    Wraps the historical ``_apply_monodec_conc`` helper. Symmetric to
    ``_MonotonicIncreaseConstraint``.

    The activation guard uses an explicit truthiness test on the
    selection list rather than ``np.any(...)``, so that selecting only
    component 0 (``[0]``) is honoured. Same fix family as PR1 #911.
    """

    name = "monotonic-decrease"

    def __init__(self, indices, tol, blocks=None):
        self._indices = indices
        self._tol = tol
        self._blocks = blocks

    def apply(self, values, state):
        if not self._indices:
            return values
        ny = values.shape[0]
        for s in self._indices:
            for curid in np.arange(ny - 1):
                if values[curid + 1, s] > values[curid, s] * self._tol:
                    values[curid + 1, s] = values[curid, s]
        return values


class _ClosureConstraint(_Constraint):
    """
    Enforce closure on selected C columns.

    Wraps the historical ``_apply_closure_conc`` helper. Preserves the
    PR1 truthiness guard on the selection list so that a single
    selected component (e.g. ``[0]``) is honoured — matching the fix
    for issue #911 / ``closureConc="all"``. The builder only emits a
    ``_ClosureConstraint`` when ``closureConc`` is truthy, so the guard
    is technically redundant here; it is kept explicit so the constraint
    remains self-contained and safe to invoke out of pipeline order.
    """

    name = "closure"

    def __init__(self, indices, method, target, blocks=None):
        self._indices = indices
        self._method = method
        self._target = target
        self._blocks = blocks

    def apply(self, values, state):
        if not self._indices:
            return values
        if self._method == "scaling":
            Q = _lstsq(values[:, self._indices], self._target.T)
            values[:, self._indices] = np.dot(
                values[:, self._indices],
                np.diag(Q),
            )
        elif self._method == "constantSum":
            totalConc = np.sum(values[:, self._indices], axis=1)
            # guard against zero total concentration to avoid nan/inf
            totalConc = np.where(totalConc == 0, 1.0, totalConc)
            values[:, self._indices] = (
                values[:, self._indices] * self._target[:, None] / totalConc[:, None]
            )
        return values


class _NormalizationConstraint(_Constraint):
    """
    Jointly rescale ``St`` and ``C`` to normalize spectral rows.

    Wraps the historical ``_apply_normalization`` helper. Because it must
    mutate both ``state.St`` and ``state.C`` jointly (so that ``C @ St``
    is preserved), ``apply`` ignores its ``values`` argument and instead
    reads from / writes to ``state``. The builder emits at most one
    normalization constraint, which is applied after the spectral constraint
    pipeline — never as part of a generic per-profile pipeline.
    """

    name = "normalization"

    # Normalization is not block-local: it modifies both C and St jointly.
    is_block_local = False

    def __init__(self, method, n_components):
        self._method = method
        self._n_components = n_components

    def apply(self, values, state):
        if self._method == "max":
            alpha = np.max(state.St, axis=1).reshape(self._n_components, 1)
            # guard against zero-norm spectra to avoid nan/inf
            alpha = np.where(alpha == 0, 1.0, alpha)
            state.St = state.St / alpha
            state.C = state.C * alpha.T
        elif self._method == "euclid":
            alpha = np.linalg.norm(state.St, axis=1).reshape(self._n_components, 1)
            # guard against zero-norm spectra to avoid nan/inf
            alpha = np.where(alpha == 0, 1.0, alpha)
            state.St = state.St / alpha
            state.C = state.C * alpha.T
        return values


class _ComponentPresenceConstraint(_Constraint):
    """
    Force zero concentration profiles for absent components in each block.

    This constraint is NOT block-local: it manages block iteration
    internally using ``state.augmentation``. It must be applied late
    in the concentration pipeline so that earlier constraints do not
    reintroduce values in absent blocks.
    """

    name = "component-presence"
    is_block_local = False

    def __init__(self, presence, blocks=None):
        self._presence = presence  # list[list[bool]], shape (n_blocks, n_components)
        self._blocks = blocks

    def apply(self, values, state):
        aug = state.augmentation
        if aug is None:
            raise ValueError("ComponentPresence constraint requires augmented data.")
        n_blocks = len(aug.row_slices)
        if len(self._presence) != n_blocks:
            raise ValueError(
                f"Presence matrix has {len(self._presence)} rows but "
                f"data has {n_blocks} blocks."
            )
        selected_blocks = range(n_blocks) if self._blocks is None else self._blocks
        for bi in selected_blocks:
            if bi >= n_blocks:
                continue
            presence_row = self._presence[bi]
            n_comp = len(presence_row)
            if values.shape[1] != n_comp:
                raise ValueError(
                    f"Presence row {bi} has {n_comp} entries but "
                    f"profile has {values.shape[1]} components."
                )
            sl = aug.row_slices[bi]
            for j, present in enumerate(presence_row):
                if not present:
                    values[sl, j] = 0.0

        # Store presence info on the augmented structure for inter-block
        # constraints (e.g. Trilinear).
        if state.augmentation is not None:
            state.augmentation.block_presence = self._presence
        return values


class _TrilinearConstraint(_Constraint):
    """
    Enforce trilinearity on selected concentration profiles across blocks.

    This constraint is NOT block-local: it manages block iteration
    internally using ``state.augmentation``. It assembles profiles from
    selected blocks, projects each component onto rank-1 SVD, and writes
    the reconstruction back.

    If ``_ComponentPresenceConstraint`` is also active, blocks where the
    component is marked absent are excluded from the projection.
    """

    name = "trilinear"
    is_block_local = False

    def __init__(self, components, blocks=None, synchronization="none"):
        self._components = components
        self._blocks = blocks
        self._synchronization = synchronization
        # Runtime diagnostics
        self.amplitudes_ = None
        self.singular_values_ = None

    def apply(self, values, state):
        aug = state.augmentation
        if aug is None:
            raise ValueError("Trilinear constraint requires augmented data.")
        if self._synchronization != "none":
            raise NotImplementedError(
                f"synchronization={self._synchronization!r} is not implemented."
            )

        n_blocks = len(aug.row_slices)
        selected_blocks = (
            list(range(n_blocks)) if self._blocks is None else self._blocks
        )

        if len(selected_blocks) < 2:
            raise ValueError(
                "Trilinear constraint requires at least 2 blocks, "
                f"got {len(selected_blocks)}."
            )

        # Validate that selected blocks have the same number of points
        block_lengths = set()
        for bi in selected_blocks:
            sl = aug.row_slices[bi]
            block_lengths.add(sl.stop - sl.start)
        if len(block_lengths) > 1:
            raise ValueError(
                "Trilinear constraint requires all selected blocks to have "
                f"the same number of points, but got lengths {sorted(block_lengths)}."
            )

        components = self._components
        if components is None:
            components = list(range(values.shape[1]))

        n_points = (
            aug.row_slices[selected_blocks[0]].stop
            - aug.row_slices[selected_blocks[0]].start
        )
        n_selected = len(selected_blocks)

        # Per-component presence mask: which blocks are active for each component.
        # True = participate in rank-1 projection; False = left at zero.
        presence_mask = (
            state.augmentation.block_presence
            if state.augmentation is not None
            and state.augmentation.block_presence is not None
            else [[True] * values.shape[1]] * n_selected
        )

        estimated_amplitudes = np.full((len(components), n_blocks), np.nan)

        for ci, comp in enumerate(components):
            # Build profiles matrix: (n_points, n_active_blocks)
            active_blocks = [
                j
                for j, bi in enumerate(selected_blocks)
                if bi < len(presence_mask) and presence_mask[bi][comp]
            ]
            n_active = len(active_blocks)

            if n_active < 2:
                # Not enough active blocks for rank-1 projection
                continue

            profiles = np.zeros((n_points, n_active))
            for j, aj in enumerate(active_blocks):
                bi = selected_blocks[aj]
                sl = aug.row_slices[bi]
                profiles[:, j] = values[sl, comp]

            reconstruction, amplitudes = _project_rank_one_profiles(profiles)
            estimated_amplitudes[
                ci, [selected_blocks[aj] for aj in active_blocks]
            ] = amplitudes

            for j, aj in enumerate(active_blocks):
                bi = selected_blocks[aj]
                sl = aug.row_slices[bi]
                values[sl, comp] = reconstruction[:, j]

        self.amplitudes_ = estimated_amplitudes
        return values


class _ModelProfileConstraint(_Constraint):
    """
    Inject profiles generated by a model callable.

    Stores ``model``, ``components``, ``model_args``,
    ``model_kwargs`` directly (created from a ``ModelProfile``
    public constraint via ``_public_to_internal``).  The constraint owns
    its mutable state (``model_args``, ``extra_outputs``,
    ``last_generated_profile``, ``history``).

    The estimator back-reference is used only for helper methods such
    as ``_C_2_NDDataset``, ``_St_2_NDDataset``,
    ``_call_external_generator`` and ``_unpack_generator_output``.
    The constraint never reads legacy estimator traitlets directly — all
    configuration reaches it through the public ``Constraint`` API.
    """

    name = "model-profile"

    def __init__(
        self,
        estimator,
        side,
        model,
        components,
        model_args=(),
        model_kwargs=None,
        profile_mapping=None,
        blocks=None,
    ):
        self._estimator = estimator
        self._side = side  # "conc" or "spec"
        self._model = model
        self._components = components  # resolved list of component indices
        self._profile_mapping = profile_mapping  # None = identity
        self._blocks = blocks

        self.model_args = model_args  # mutable: replaced when model returns new_args
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}

        # Runtime state
        self.extra_outputs = []  # per-iteration extra outputs from the model
        self.last_generated_profile = None  # profile from the last iteration
        self.history = []  # (model_args, extra_output, profile) per iteration

    def apply(self, values, state):
        est = self._estimator
        if not self._components:
            return values

        extra = None
        if self._side == "conc":
            current = est._C_2_NDDataset(values)
            output = est._call_external_generator(
                self._model,
                current,
                self.model_args,
                self._model_kwargs,
            )
            profiles, new_args, extra = est._unpack_generator_output(output)
            if new_args is not _UNCHANGED:
                self.model_args = new_args
            if self._profile_mapping is not None:
                profiles_arr = _profile_asarray(profiles)
                n_outputs = profiles_arr.shape[1]
                for i, comp in enumerate(self._components):
                    col = self._profile_mapping[i]
                    if col is not None:
                        if col >= n_outputs:
                            raise IndexError(
                                f"mapping[{i}] = {col} is out of range for "
                                f"the model output which has {n_outputs} "
                                f"column{'s' if n_outputs != 1 else ''}."
                            )
                        values[:, comp] = profiles_arr[:, col]
            else:
                values[:, self._components] = _profile_asarray(profiles)
            state.extra_output_conc = extra
        else:
            current = est._St_2_NDDataset(values)
            output = est._call_external_generator(
                self._model,
                current,
                self.model_args,
                self._model_kwargs,
            )
            profiles, new_args, extra = est._unpack_generator_output(output)
            if new_args is not _UNCHANGED:
                self.model_args = new_args
            if self._profile_mapping is not None:
                profiles_arr = _profile_asarray(profiles)
                n_outputs = profiles_arr.shape[0]
                for i, comp in enumerate(self._components):
                    row = self._profile_mapping[i]
                    if row is not None:
                        if row >= n_outputs:
                            raise IndexError(
                                f"mapping[{i}] = {row} is out of range for "
                                f"the model output which has {n_outputs} "
                                f"row{'s' if n_outputs != 1 else ''}."
                            )
                        values[comp, :] = profiles_arr[row, :]
            else:
                values[self._components, :] = _profile_asarray(profiles)
            state.extra_output_spec = extra

        self.extra_outputs.append(extra if extra is not None else [])
        self.last_generated_profile = profiles
        self.history.append((self.model_args, extra, profiles))
        return values


# DEVNOTE:
# the following decorator allow to correct signature and docs of traitlets.HasTraits
# derived class
@signature_has_configurable_traits
class MCRALS(DecompositionAnalysis):
    """
    Multivariate Curve Resolution Alternating Least Squares (MCRALS).

    :term:`MCR-ALS` ( ``Multivariate Curve Resolution Alternating Least Squares`` )
    resolve's a set (or several sets) of spectra :math:`X` of an evolving mixture
    (or a set of mixtures) into the spectra :math:`S^t` of "pure" species and their
    concentration profiles :math:`C`.

    In terms of matrix equation:

    .. math:: X = C.S^t + E

    where :math:`E` is the matrix of residuals.

    Parameters
    ----------
    constraints : list of `Constraint`, optional, default: ``None``
        Public constraint objects describing scientific prior knowledge about
        the concentration (``"C"``) or spectral (``"St"``) profiles. When
        provided, each element must be an instance of a public constraint
        class (e.g., ``NonNegative``, ``Unimodal``, ``Closure``,
        ``Monotonic``, ``ModelProfile``). ``None`` selects the built-in
        default constraints; an empty list requests an unconstrained fit.

        ``constraints`` can be passed at construction time or assigned before
        calling ``fit``::

            from spectrochempy.analysis import constraints as mc

            mcr = MCRALS()
            mcr.constraints = [
                mc.NonNegative("C"),
                mc.Closure("C"),
            ]
            mcr.fit(X, guess)

        .. versionadded:: 0.7.0
    solver_C : ``'lstsq'`` | ``'nnls'`` | ``'pnnls'``, optional, default: ``'lstsq'``
        Solver used to estimate concentration profiles ``C`` from ``X`` and
        ``St``.
    solver_St : ``'lstsq'`` | ``'nnls'`` | ``'pnnls'``, optional, default: ``'lstsq'``
        Solver used to estimate spectral profiles ``St`` from ``X`` and ``C``.
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    warm_start : `bool`, optional, default: `False`
        When fitting repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        reuse the solution of the previous call to fit and add more components
        (if available) in a sequential manner.

        When `warm_start` is `True`, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to `fit`.

    Notes
    -----
    Three dimensionless stopping diagnostics are evaluated after each ALS
    iteration: ``reconstruction_error`` measures the current relative
    reconstruction error, ``residual_change`` measures the relative change
    in residual standard deviation since the preceding iteration, and
    ``profile_change`` measures the scale/sign-invariant relative change of
    the resolved factor profiles. Convergence is reached as soon as any
    enabled tolerance is satisfied. By default, only
    ``tol_residual_change=1e-3`` is enabled.

    With ``log_level="INFO"``, these three diagnostics are printed using the
    same relative, dimensionless convention as their tolerances. The final
    message identifies the stopping diagnostic, its value, and its tolerance.

    The bilinear model has an intrinsic scale ambiguity: without a calibrated,
    unit-bearing initial ``C0`` or ``St0``, the individual value units of the
    resolved factors are undefined. In that case ``C`` and ``St`` remain
    unitless and use conservative titles. A calibrated initial factor fixes the
    scale convention, allowing the complementary units to be derived from
    ``[X] = [C][St]``.

    Operations that explicitly reset or exchange factor scale — spectral
    normalization, closure, and model/hard profile replacement — clear the
    resolved factor units. For horizontal augmentation, ``St_blocks`` carries
    block-specific physical metadata; the heterogeneous concatenated ``St`` is
    deliberately unitless with a neutral title.

    See Also
    --------
    fit : Fit the MCRALS model on X.
    transform : Apply dimensionality reduction.
    fit_transform : Fit the model and apply dimensionality reduction.

    """

    # Developer notes
    # ----------------
    # Private traits with internal validation (see Traitlets library for more
    # information)
    # Notice that variable not defined this way lack this type validation, so they are
    # more prone to errors.

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # They will be written in a file from which the default can be modified)
    # Obviously, the parameters can also be modified at runtime as usual by assignment.
    # ----------------------------------------------------------------------------------

    tol_residual_change = tr.Float(
        1.0e-3,
        allow_none=True,
        min=0.0,
        help=(
            "Relative tolerance on the change in residual standard deviation. "
            "The dimensionless value is abs(sigma_k - sigma_(k-1)) / "
            "sigma_(k-1); 1e-3 therefore means 0.1%. Set to None to disable "
            "this stopping criterion."
        ),
    ).tag(config=True)

    tol_reconstruction_error = tr.Float(
        None,
        allow_none=True,
        min=0.0,
        help=(
            "Tolerance on the relative reconstruction error "
            "norm(X - C @ St) / norm(X). This dimensionless stopping "
            "criterion is disabled when set to None."
        ),
    ).tag(config=True)

    tol_profile_change = tr.Float(
        None,
        allow_none=True,
        min=0.0,
        help=(
            "Relative tolerance on the change of the resolved C and St "
            "profiles between successive iterations. Factor pairs are first "
            "normalized to remove their arbitrary reciprocal scale and sign; "
            "the larger relative Frobenius change of C and St is used. This "
            "dimensionless criterion is disabled when set to None."
        ),
    ).tag(config=True)

    # Deprecated compatibility alias. Historically this value was expressed
    # as a percentage, hence tol=0.1 is equivalent to the new default 1e-3.
    tol = tr.Float(
        0.1,
        help=("Deprecated alias for 100 * tol_residual_change (percent)."),
    ).tag(config=True)

    max_iter = tr.Integer(50, help="Maximum number of :term:`ALS` iteration.").tag(
        config=True,
    )

    maxdiv = tr.Integer(
        5,
        help="Maximum number of successive non-converging iterations.",
    ).tag(config=True)

    solverConc = tr.Enum(
        ["lstsq", "nnls", "pnnls"],
        default_value="lstsq",
        help=(
            r"""Solver used to get `C` from `X` and `St`.

- ``'lstsq'``\ : uses ordinary least squares with `~numpy.linalg.lstsq`
- ``'nnls'``\ : non-negative least squares (`~scipy.optimize.nnls`) are applied
  sequentially on all profiles
- ``'pnnls'``\ : bounded least squares constrains the component profiles indicated
  in `nonnegConc` to be non-negative while leaving other components unconstrained.
"""
        ),
    ).tag(config=True)

    nonnegConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Non-negativity constraint on concentrations.

- ``'all'``\ : all concentrations profiles are considered
  non-negative.
- `list` of indexes: the corresponding profiles are considered non-negative,
  not the others. For instance ``[0, 2]`` indicates that profile \#0 and \#2
  are non-negative while profile \#1 *can* be negative.
- ``[]``\ : all profiles can be negative."""
        ),
    ).tag(config=True)

    unimodConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Unimodality constraint on concentrations.

- ``'all'``\ : all concentrations profiles are considered unimodal.
- `list` of indexes: the corresponding profiles are considered unimodal, not the others.
  For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are unimodal while
  profile ``#1`` *can* be multimodal.
- ``[]``\ : all profiles can be multimodal."""
        ),
    ).tag(config=True)

    unimodConcMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help=(
            r"""Method to apply unimodality.

- ``'strict'``\ : values deviating from :term:`unimodality` are reset to the value of the
  previous point.
- ``'smooth'``\ : both values (deviating point and previous point) are modified to avoid
  steps in the concentration profile."""
        ),
    ).tag(config=True)

    unimodConcTol = tr.Float(
        default_value=1.1,
        help=(
            r"""Tolerance parameter for :term:`unimodality`.

Correction is applied only if:

- ``C[i,j] > C[i-1,j] * unimodConcTol`` on the decreasing branch of profile ``#j``,
- ``C[i,j] < C[i-1,j] * unimodConcTol`` on the increasing branch of profile ``#j``."""
        ),
    ).tag(config=True)

    monoDecConc = tr.List(
        default_value=[],
        help=(
            r"""Monotonic decrease constraint on concentrations.

- ``[]``\ : no constraint is applied.
- `list` of indexes: the corresponding profiles are considered to decrease
  monotonically, not the others. For instance ``[0, 2]`` indicates that profile ``#0``
  and ``#2`` are decreasing while profile ``#1`` *can* increase."""
        ),
    ).tag(config=True)

    monoDecTol = tr.Float(
        default_value=1.1,
        help=r"""Tolerance parameter for monotonic decrease.

Correction is applied only if: ``C[i,j] > C[i-1,j] * monoDecTol`` .""",
    ).tag(config=True)

    monoIncConc = tr.List(
        default_value=[],
        help=(
            r"""Monotonic increase constraint on concentrations.

- ``[]``\ : no constraint is applied.
- `list` of indexes: the corresponding profiles are considered to increase
  monotonically, not the others. For instance ``[0, 2]`` indicates that profile ``#0``
  and ``#2`` are increasing while profile ``#1`` *can* decrease."""
        ),
    ).tag(config=True)

    monoIncTol = tr.Float(
        default_value=1.1,
        help=r"""Tolerance parameter for monotonic increase.

Correction is applied only if ``C[i,j] < C[i-1,j] * monoIncTol`` along profile ``#j``.""",
    ).tag(config=True)

    closureConc = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help=(
            r"""Defines the concentration profiles subjected to closure constraint.

- ``[]``\ : no constraint is applied.
- ``'all'``\ : all profile are constrained so that their weighted sum equals the
  `closureTarget`
- `list` of indexes: the corresponding profiles are constrained so that their weighted sum
  equals `closureTarget`."""
        ),
    ).tag(config=True)

    closureTarget = tr.Union(
        (tr.Enum(["default"]), Array()),
        default_value="default",
        help=(
            r"""The value of the sum of concentrations profiles subjected to closure.

- ``'default'``\ : the total concentration is set to ``1.0`` for all observations.
- :term:`array-like` of size :term:`n_observations`: the values of concentration for
  each observation. Hence, ``np.ones(X.shape[0]``) would be equivalent to
  ``'default'``."""
        ),
    ).tag(config=True)

    closureMethod = tr.Enum(
        ["scaling", "constantSum"],
        default_value="scaling",
        help=(
            r"""The method used to enforce :term:`closure` (:cite:t:`omidikia:2018`).

- ``'scaling'`` recompute the concentration profiles using least squares:

  .. math::

     C \leftarrow C \cdot \textrm{diag} \left( C_L^{-1} c_t \right)

  where :math:`c_t` is the vector given by `closureTarget` and :math:`C_L^{-1}`
  is the left inverse of :math:`C`.
- ``'constantSum'`` normalize the sum of concentration profiles to `closureTarget`.
"""
        ),
    ).tag(config=True)

    hardConc = tr.List(
        default_value=[],
        help=(
            r"""Defines hard constraints on the concentration profiles.

- ``[]``\ : no constraint is applied.
- `list` of indexes: the corresponding profiles will set by `getConc`."""
        ),
    ).tag(config=True)

    getConc = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help=(
            r"""An external function that provide ``len(hardConc)`` concentration
profiles.

It should be using one of the following syntax:

- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC``
- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc``
- ``getConc(Ccurr, *argsGetConc, **kwargsGetConc) -> hardC, newArgsGetConc,
  extraOutputGetConc``

where:

- ``Ccurr`` is the current `C` dataset,
- ``argsGetConc`` are the parameters needed to completely specify the function.
- ``hardC`` is a `~numpy.ndarray` or `NDDataset` of shape
  (:term:`n_observations` , len(``hardConc``),
- ``newArgsGetConc`` are the updated parameters for the next iteration (can be `None`),
- ``extraOutputGetConc`` can be any other relevant output to be kept in
  ``extraOutputGetConc`` attribute, a list of ``extraOutputGetConc`` at each MCR ALS
  iteration.

.. note::
    ``getConc`` can be also a serialized function created using dill and base64
    python libraries. Normally not used directly, it is here for internal
    process."""
        ),
    ).tag(config=True)

    argsGetConc = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetConc = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    getC_to_C_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help=(
            r"""Correspondence of the profiles returned by `getConc`
and `C[:,hardConc]`.

- ``'default'``: the profiles correspond to those of `C[:,hardConc]`. This is equivalent
  to ``range(len(hardConc)``)
- `list` of indexes or of `None`. For instance ``[2, 1, 0]`` indicates that the
  third profile returned by `getC` (index ``2``) corresponds to the 1st profile of
  `C[:, hardConc]`, the 2nd returned profile (index ``1``) corresponds to
  second profile of `C[:, hardConc]`, etc..."""
        ),
    ).tag(config=True)

    solverSpec = tr.Enum(
        ["lstsq", "nnls", "pnnls"],
        default_value="lstsq",
        help=(
            r"""Solver used to get `St` from `X` and `C`.

- ``'lstsq'``\ : uses ordinary least squares with `~numpy.linalg.lstsq`
- ``'nnls'``\ : non-negative least squares (`~scipy.optimize.nnls`) are applied
  sequentially on all profiles
- ``'pnnls'``\ : bounded least squares constrains the component profiles indicated
  in `nonnegSpec` to be non-negative while leaving other components unconstrained."""
        ),
    ).tag(config=True)

    nonnegSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value="all",
        help=(
            r"""Non-negativity constraint on spectra.

- ``'all'``\ : all profiles are considered non-negative.
- `list` of indexes : the corresponding profiles are considered non-negative, not the
  others. For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are
  non-negative while profile ``#1`` *can* be negative.
- ``[]``\ : all profiles can be negative."""
        ),
    ).tag(config=True)

    normSpec = tr.Enum(
        [None, "euclid", "max"],
        default_value=None,
        help=(
            r"""Defines whether the spectral profiles should be normalized.

- `None`\ : no normalization is applied.
- ``'euclid'``\ : spectra are normalized with respect to their total area,
- ``'max'``\ : spectra are normalized with respect to their maximum value."""
        ),
    ).tag(config=True)

    unimodSpec = tr.Union(
        (tr.Enum(["all"]), tr.List()),
        default_value=[],
        help=(
            r"""Unimodality constraint on Spectra.

- ``[]``\ : all profiles can be multimodal.
- ``'all'``\ : all profiles are unimodal (equivalent to ``range(n_components)``).
- array of indexes : the corresponding profiles are considered unimodal, not the others.
  For instance ``[0, 2]`` indicates that profile ``#0`` and ``#2`` are unimodal while
  profile ``#1`` *can* be multimodal."""
        ),
    ).tag(config=True)

    unimodSpecMod = tr.Enum(
        ["strict", "smooth"],
        default_value="strict",
        help=(
            r"""Method used to apply unimodality.

- ``'strict'``\ : values deviating from unimodality are reset to the value of the previous
  point.
- ``'smooth'``\ : both values (deviating point and previous point) are modified to avoid
  steps in the concentration profile."""
        ),
    ).tag(config=True)

    unimodSpecTol = tr.Float(
        default_value=1.1,
        help=(
            r"""Tolerance parameter for unimodality.

Correction is applied only if the deviating point ``St[j, i]`` is larger than
``St[j, i-1] * unimodSpecTol`` on the decreasing branch of profile
``#j``, or lower than ``St[j, i-1] * unimodSpecTol`` on the increasing branch of
profile  ``#j``."""
        ),
    ).tag(config=True)

    hardSpec = tr.List(
        default_value=[],
        help=(
            r"""Defines hard constraints on the spectral profiles.

- ``[]``\ : no constraint is applied.
- `list` of indexes : the corresponding profiles will set by `getSpec`."""
        ),
    ).tag(config=True)

    getSpec = tr.Union(
        (tr.Callable(), tr.Unicode()),
        default_value=None,
        allow_none=True,
        help=(
            r"""An external function that will provide ``len(hardSpec)`` concentration
profiles.

It should be using one of the following syntax:

- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt``
- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec``
- ``getSpec(Stcurr, *argsGetSpec, **kwargsGetSpec) -> hardSt, newArgsGetSpec,
  extraOutputGetSpec``

with:

- ``Stcurr``\ : the current value of `St` in the :term:`ALS` loop,
- ``*argsGetSpec`` and ``**kwargsGetSpec``\ : the parameters needed to completely
  specify the function.
- ``hardSt``\ : `~numpy.ndarray` or `NDDataset` of shape
  ``(n_observations, len(hardSpec)``,
- ``newArgsGetSpec``\ : updated parameters for the next ALS iteration (can be None),
- ``extraOutputGetSpec``\ : any other relevant output to be kept in
  `extraOutputGetSpec` attribute, a list of ``extraOutputGetSpec`` at each iterations.

.. note::
    ``getSpec`` can be also a serialized function created using dill and base64
    python libraries. Normally not used directly, it is here for internal process.
"""
        ),
    ).tag(config=True)

    argsGetSpec = tr.Tuple(
        default_value=(),
        help="Supplementary positional arguments passed to the external function.",
    ).tag(config=True)

    kwargsGetSpec = tr.Dict(
        default_value={},
        help="Supplementary keyword arguments passed to the external function.",
    ).tag(config=True)

    getSt_to_St_idx = tr.Union(
        (tr.Enum(["default"]), tr.List()),
        default_value="default",
        help=(
            r"""Correspondence between the indexes of the spectra returned by `getSpec`
and `St`.

- ``'default'``\ : the indexes correspond to those of `St`. This is equivalent
  to ``range(len(hardSpec)``).
- `list` of indexes : corresponding indexes in `St`, i.e. ``[2, None, 0]`` indicates that the
  first returned profile corresponds to the third `St` profile (index ``2``), the 2nd
  returned profile does not correspond to any profile in `St`, the 3rd returned profile
  corresponds to the first `St` profile (index ``0`` )."""
        ),
    ).tag(config=True)

    storeIterations = tr.Bool(
        default_value=False,
        help=(r"""Whether to store the C and St generated at each iteration."""),
    ).tag(config=True)

    constraints = tr.Any(
        default_value=None,
        allow_none=True,
        help=(
            "Public constraint objects for MCR-ALS. "
            "``None`` selects the built-in defaults; an empty sequence means "
            "unconstrained. Otherwise, provide a list or tuple of ``Constraint`` "
            "instances defining the scientific constraints for the fit."
        ),
    )

    # ----------------------------------------------------------------------------------
    # Public property wrappers for deprecated solver traitlet names
    # ----------------------------------------------------------------------------------
    @property
    def solver_C(self):
        """Solver used to get `C` from `X` and `St`."""
        return self.solverConc

    @solver_C.setter
    def solver_C(self, value):
        self.solverConc = value

    @property
    def solver_St(self):
        """Solver used to get `St` from `X` and `C`."""
        return self.solverSpec

    @solver_St.setter
    def solver_St(self, value):
        self.solverSpec = value

    @property
    def warm_start(self):
        """Whether to reuse the previous solution on the next ``fit`` call."""
        return self._warm_start

    @warm_start.setter
    def warm_start(self, value):
        self._warm_start = value

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *args,
        log_level=logging.WARNING,
        warm_start=False,
        **kwargs,
    ):
        # Mark instance as not yet fully initialized so that __setattr__
        # does not emit deprecation warnings for trait defaults.
        object.__setattr__(self, "_init_done", False)

        if len(args) > 0:
            raise ValueError(
                "Passing arguments such as MCRALS(X, profile) "
                "is now deprecated. "
                "Instead, use MCRAL() followed by MCRALS.fit(X, profile). "
                "See the documentation and examples",
            )

        # ``tol`` used percentages historically. Keep accepting it, but make
        # the modern relative-valued parameter the single internal source of
        # truth.
        legacy_tol_passed = "tol" in kwargs
        if legacy_tol_passed:
            if "tol_residual_change" in kwargs:
                raise ValueError(
                    "Cannot specify both deprecated `tol` and " "`tol_residual_change`."
                )
            warnings.warn(
                "`tol` is deprecated; use `tol_residual_change` with a "
                "relative value instead (for example, tol=0.1 becomes "
                "tol_residual_change=1e-3).",
                FutureWarning,
                stacklevel=2,
            )
            kwargs["tol_residual_change"] = kwargs["tol"] / 100.0

        # Detect deprecated solver kwarg names and warn.
        if "solverConc" in kwargs:
            warnings.warn(
                "`solverConc` is deprecated; use `solver_C` instead.",
                FutureWarning,
                stacklevel=2,
            )
            if "solver_C" in kwargs:
                raise ValueError("Cannot specify both `solverConc` and `solver_C`.")
            kwargs["solver_C"] = kwargs.pop("solverConc")
        if "solverSpec" in kwargs:
            warnings.warn(
                "`solverSpec` is deprecated; use `solver_St` instead.",
                FutureWarning,
                stacklevel=2,
            )
            if "solver_St" in kwargs:
                raise ValueError("Cannot specify both `solverSpec` and `solver_St`.")
            kwargs["solver_St"] = kwargs.pop("solverSpec")

        # Detect legacy constraint traitlet kwargs and warn.
        _legacy_in_kwargs = _LEGACY_CONSTRAINT_TRAITS & set(kwargs)
        if _legacy_in_kwargs:
            warnings.warn(
                "Legacy MCR-ALS constraint parameters are deprecated; "
                "use the `constraints` API instead.",
                FutureWarning,
                stacklevel=2,
            )

        # Handle constraints= parameter (now a validated traitlet).
        constraints_value = kwargs.pop("constraints", _UNSET)
        constraints_passed = constraints_value is not _UNSET

        # Mixed API detection at constructor time:
        # only raise when constraints is non-None AND legacy traits present.
        if constraints_passed and constraints_value is not None and _legacy_in_kwargs:
            raise ValueError(
                "Legacy constraint parameters and `constraints` "
                "cannot be used together."
            )

        # Track which legacy traits were explicitly set (mutable set).
        object.__setattr__(self, "_explicit_legacy_traits", set(_legacy_in_kwargs))
        object.__setattr__(self, "_validating_legacy", False)

        # Re-inject constraints so the traitlet system handles validation
        # and observes during super().__init__().
        if constraints_passed:
            kwargs["constraints"] = constraints_value

        # call the super class for initialisation
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

        # Keep the deprecated percentage-valued attribute coherent for code
        # that still reads it.
        if not legacy_tol_passed and self.tol_residual_change is not None:
            super().__setattr__("tol", 100.0 * self.tol_residual_change)

        # deal with the callable that may have been serialized
        if self.getConc is not None and isinstance(self.getConc, str):
            self.getConc = dill.loads(base64.b64decode(self.getConc))  # noqa: S301
        if self.getSpec is not None and isinstance(self.getSpec, str):
            self.getSpec = dill.loads(base64.b64decode(self.getSpec))  # noqa: S301

        # storage for ALS diagnostics captured during _fit
        self._fit_meta = None
        self._active_constraints_ = ()

        # Augmented (multiset) data support
        self._augmented_structure = None
        self._X_inputs = None

        # Instance is now fully initialized.
        object.__setattr__(self, "_init_done", True)

    def __setattr__(self, name, value):
        # Emit deprecation warnings and mixed-API detection when legacy
        # parameters are assigned on an already-initialized instance.
        init_done = getattr(self, "_init_done", False)
        if name == "tol" and init_done:
            warnings.warn(
                "`tol` is deprecated; use `tol_residual_change` with a "
                "relative value instead.",
                FutureWarning,
                stacklevel=2,
            )
            super().__setattr__(name, value)
            super().__setattr__("tol_residual_change", value / 100.0)
            return
        if name == "tol_residual_change" and init_done:
            super().__setattr__(name, value)
            if value is not None:
                super().__setattr__("tol", 100.0 * value)
            return
        if name in ("solverConc", "solverSpec") and init_done:
            replacement = "solver_C" if name == "solverConc" else "solver_St"
            warnings.warn(
                f"`{name}` is deprecated; use `{replacement}` instead.",
                FutureWarning,
                stacklevel=2,
            )
        elif name in _LEGACY_CONSTRAINT_TRAITS and init_done:
            # Skip mixed-API checks and deprecation warnings during internal
            # re-validation (_n_components_change).
            if getattr(self, "_validating_legacy", False):
                super().__setattr__(name, value)
                return
            # Detect mixed API: if constraints is non-None, reject.
            constraints = self.constraints
            if constraints is not None:
                raise ValueError(
                    "Legacy constraint parameters and `constraints` "
                    "cannot be used together."
                )
            object.__getattribute__(self, "_explicit_legacy_traits").add(name)
            warnings.warn(
                f"Legacy MCR-ALS constraint parameter `{name}` is deprecated; "
                "use the `constraints` API instead.",
                FutureWarning,
                stacklevel=2,
            )
        super().__setattr__(name, value)

    # ----------------------------------------------------------------------------------
    # Public textual representation
    # ----------------------------------------------------------------------------------

    def __repr__(self):
        cls = self.__class__.__name__
        indent = "    "

        lines = [f"{cls}("]

        # Constraints list — keep readable, summarise if long
        constraints = self.constraints
        if constraints is None:
            lines.append(f"{indent}constraints=None,")
        else:
            max_show = 5
            total = len(constraints)
            if total <= max_show:
                lines.append(f"{indent}constraints=[")
                for c in constraints:
                    lines.append(f"{indent}{indent}{c!r},")
                lines.append(f"{indent}],")
            else:
                lines.append(f"{indent}constraints=[")
                for c in constraints[:max_show]:
                    lines.append(f"{indent}{indent}{c!r},")
                lines.append(f"{indent}{indent}... ({total - max_show} more),")
                lines.append(f"{indent}],")

        # Configuration parameters
        lines.append(f"{indent}solver_C={self.solver_C!r},")
        lines.append(f"{indent}solver_St={self.solver_St!r},")
        lines.append(f"{indent}max_iter={self.max_iter!r},")
        lines.append(f"{indent}tol_residual_change={self.tol_residual_change!r},")
        lines.append(
            f"{indent}tol_reconstruction_error=" f"{self.tol_reconstruction_error!r},"
        )
        lines.append(f"{indent}tol_profile_change={self.tol_profile_change!r},")
        lines.append(f"{indent}maxdiv={self.maxdiv!r},")

        # Fitted indicator
        if self._fitted:
            lines.append(f"{indent}fitted=True,")

        lines.append(")")
        return "\n".join(lines)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------

    def _nonneg_indices(self, side):
        """
        Return non-negative component indices for the pnnls solver.

        When the new ``constraints`` API is active, indices are derived
        from ``NonNegative`` constraints.  Otherwise falls back to the
        legacy traitlet (``nonnegConc`` / ``nonnegSpec``).
        """
        if self.constraints is None:
            return self.nonnegConc if side == "C" else self.nonnegSpec
        from spectrochempy.analysis.decomposition.mcrals_constraints import (
            NonNegative,  # noqa: PLC0415
        )

        indices = set()
        any_all = False
        for c in self.constraints:
            if isinstance(c, NonNegative) and c.profile == side:
                if c.components is None:
                    any_all = True
                else:
                    indices.update(c.components)
        if any_all:
            return list(range(self._n_components)) if self._n_components else []
        return sorted(indices) if indices else []

    def _solve_C(self, St):
        if self.solver_C == "lstsq":
            return _lstsq(St.T, self._X.data.T).T
        if self.solver_C == "nnls":
            return _nnls(St.T, self._X.data.T).T
        if self.solver_C == "pnnls":
            return _pnnls(St.T, self._X.data.T, nonneg=self._nonneg_indices("C")).T
        return None

    def _solve_St(self, C):
        if self.solver_St == "lstsq":
            return _lstsq(C, self._X.data)
        if self.solver_St == "nnls":
            return _nnls(C, self._X.data)
        if self.solver_St == "pnnls":
            return _pnnls(C, self._X.data, nonneg=self._nonneg_indices("St"))
        return None

    def _guess_profile(self, profile):
        # Set or guess an initial profile.

        if self._X_is_missing:
            return None

        # check the dimensions compatibility
        # As the dimension of profile should match the initial shape
        # of X we use self._X_shape not self._X.shape (because for this masked columns
        # or rows have already been removed.
        if (self._X_shape[1] != profile.shape[1]) and (
            self._X_shape[0] != profile.shape[0]
        ):
            raise ValueError(
                f"None of the dimensions of the given profile "
                f"[{profile.shape}] correspond to any of those "
                f"of X [{self._X_shape}].",
            )

        # mask info
        if np.any(self._X_mask):
            masked_rows, masked_columns = self._get_masked_rc(self._X_mask)

        # make the profile
        if profile.shape[0] == self._X_shape[0]:
            # this should be a concentration profile.
            C = profile.copy()
            self._C0 = C
            self._n_components = C.shape[1]
            info_(
                f"Concentration profile initialized with {self._n_components} components",
            )

            # compute initial spectra (using X eventually masked)
            St = self._solve_St(C)
            self._St0 = St
            info_("Initial spectra profile computed")
            # if everything went well here, C and St are set, we return
            # after having removed the eventual C mask!
            if np.any(self._X_mask):
                C = C[~masked_rows]
            return C, St

        # necessarily: profile.shape[1] == profile.shape[0]
        St = profile.copy()
        self._St0 = St
        self._n_components = St.shape[0]
        info_(f"Spectra profile initialized with {self._n_components} components")

        # compute initial spectra
        C = self._solve_C(St)
        self._C0 = C
        info_("Initial concentration profile computed")
        # if everything went well here, C and St are set, we return
        # after having removed the eventual St mask!
        if np.any(self._X_mask):
            St = St[:, ~masked_columns]
        # update the number of components
        return C, St

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def _C_2_NDDataset(self, C):
        # getconc takes the C NDDataset as first argument (to take advantage
        # of useful metadata). But the current C in fit method is a ndarray (without
        # the masked rows and columns, nor the coord information: this
        # function will create the corresponding dataset
        return C

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def _St_2_NDDataset(self, St):
        # getconc takes the C NDDataset as first argument (to take advantage
        # of useful metadata). The current St in fit method is a ndarray (without
        # the masked rows and columns, nor the coord information: this
        # function will create the corresponding dataset
        return St

    # ----------------------------------------------------------------------------------
    # Private validation methods and default getter
    # ----------------------------------------------------------------------------------
    @tr.validate("nonnegConc")
    def _validate_nonnegConc(self, proposal):
        if self._X_is_missing:
            return proposal.value
        nonnegConc = proposal.value
        if not self._n_components:  # not initialized or 0
            return nonnegConc
        if nonnegConc == "all":
            nonnegConc = np.arange(
                self._n_components,
            ).tolist()  # IMPORTANT! .tolist, not list()
            # to get integer type not int64 which are not compatible with the setting
        elif np.any(nonnegConc) and (
            len(nonnegConc) > self._n_components
            or max(nonnegConc) + 1 > self._n_components
        ):  # note that we use np.any(nnonnegConc) instead of nnonnegConc != []
            # due to a deprecation warning from traitlets.
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `nonnegConc` configuration (value: {nonnegConc})",
            )
        return nonnegConc

    @tr.validate("unimodConc")
    def _validate_unimodConc(self, proposal):
        if self._X_is_missing:
            return proposal.value
        unimodConc = proposal.value
        if not self._n_components:  # not initialized or 0
            return unimodConc
        if unimodConc == "all":
            unimodConc = np.arange(self._n_components).tolist()
        elif np.any(unimodConc) and (
            len(unimodConc) > self._n_components
            or max(unimodConc) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check the "
                f"`unimodConc` configuration (value: {unimodConc})",
            )
        return unimodConc

    @tr.validate("closureConc")
    def _validate_closureConc(self, proposal):
        if self._X_is_missing:
            return proposal.value
        closureConc = proposal.value
        if closureConc == "all":
            closureConc = list(range(self._n_components))
        elif len(closureConc) > self._n_components:
            raise ValueError(
                f"The model contains only {self._n_components} components, please check "
                f"the 'closureConc' configuration (value: {closureConc})",
            )
        return closureConc

    @tr.validate("closureTarget")
    def _validate_closureTarget(self, proposal):
        if self._X_is_missing:
            return proposal.value
        closureTarget = proposal.value
        ny = self.X.shape[0]
        if isinstance(closureTarget, str):
            if closureTarget == "default":
                closureTarget = np.ones(ny)
        elif len(closureTarget) != ny:
            raise ValueError(
                f"The data contain only {ny} observations, please check "
                f"the 'closureTarget' configuration (value: {closureTarget})",
            )
        return closureTarget

    @tr.validate("getC_to_C_idx")
    def _validate_getC_to_C_idx(self, proposal):
        if self._X_is_missing:
            return proposal.value
        getC_to_C_idx = proposal.value
        if not self._n_components:  # not initialized or 0
            return getC_to_C_idx
        if getC_to_C_idx == "default":
            getC_to_C_idx = np.arange(self._n_components).tolist()
        elif (
            len(getC_to_C_idx) > self._n_components
            #   or max(getC_to_C_idx) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `getC_to_C_idx`  configuration (value: {getC_to_C_idx})",
            )
        return getC_to_C_idx

    @tr.validate("nonnegSpec")
    def _validate_nonnegSpec(self, proposal):
        if self._X_is_missing:
            return proposal.value
        nonnegSpec = proposal.value
        if not self._n_components:  # not initialized or 0
            return nonnegSpec
        if nonnegSpec == "all":
            nonnegSpec = np.arange(self._n_components).tolist()
        elif np.any(nonnegSpec) and (
            len(nonnegSpec) > self._n_components
            or max(nonnegSpec) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `nonnegSpec`configuration (value: {nonnegSpec})",
            )
        return nonnegSpec

    @tr.validate("unimodSpec")
    def _validate_unimodSpec(self, proposal):
        if self._X_is_missing:
            return proposal.value
        unimodSpec = proposal.value
        if not self._n_components:  # not initialized or 0
            return unimodSpec
        if unimodSpec == "all":
            unimodSpec = np.arange(self._n_components).tolist()
        elif np.any(unimodSpec) and (
            len(unimodSpec) > self._n_components
            or max(unimodSpec) + 1 > self._n_components
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check the "
                f"`unimodSpec`configuration",
            )
        return unimodSpec

    @tr.validate("getSt_to_St_idx")
    def _validate_getSt_to_St_idx(self, proposal):
        if self._X_is_missing:
            return proposal.value
        getSt_to_St_idx = proposal.value
        if not self._n_components:  # not initialized or 0
            return getSt_to_St_idx
        if getSt_to_St_idx == "default":
            getSt_to_St_idx = np.arange(self._n_components).tolist()
        elif len(getSt_to_St_idx) > self._n_components or (
            max(i for i in getSt_to_St_idx if i is not None) + 1 > self._n_components
            if any(i is not None for i in getSt_to_St_idx)
            else False
        ):
            raise ValueError(
                f"The profile has only {self._n_components} species, please check "
                f"the `getSt_to_St_idx`  configuration (value: {getSt_to_St_idx})",
            )
        return getSt_to_St_idx

    @tr.observe("_n_components")
    def _n_components_change(self, change):
        # triggered in _guess_profile
        if self._n_components > 0:
            # perform a validation of default configuration parameters
            # Indeed, if not forced here these parameters are validated only when they
            # are set explicitly.
            # Here is an ugly trick to force this validation. # TODO: better way?
            # DEVNOTE: _validating_legacy flag prevents __setattr__ from
            # tracking these internal re-assignments as explicit user assignments.
            # The try/finally ensures the flag is reset even if validation raises.
            self._validating_legacy = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                    self.closureTarget = self.closureTarget
                    self.getC_to_C_idx = self.getC_to_C_idx
                    self.getSt_to_St_idx = self.getSt_to_St_idx
                    self.nonnegConc = self.nonnegConc
                    self.nonnegSpec = self.nonnegSpec
                    self.unimodConc = self.unimodConc
                    self.unimodSpec = self.unimodSpec
                    self.closureConc = self.closureConc
            finally:
                self._validating_legacy = False

    # ----------------------------------------------------------------------------------
    # Constraints traitlet — validated, observes fitted state
    # ----------------------------------------------------------------------------------
    @tr.validate("constraints")
    def _validate_constraints(self, proposal):
        value = proposal.value
        if value is None:
            return None
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError(
                "constraints must be None, a list, or a tuple, "
                f"got {type(value).__name__}"
            )
        for i, c in enumerate(value):
            if not isinstance(c, Constraint):
                raise TypeError(
                    f"constraints[{i}] must be a Constraint instance, "
                    f"got {type(c).__name__}"
                )
        if getattr(self, "_explicit_legacy_traits", None):
            raise ValueError(
                "Legacy constraint parameters and `constraints` "
                "cannot be used together."
            )
        return list(value)

    @tr.observe("constraints")
    def _on_constraints_changed(self, change):
        if self._init_done:
            self._fitted = False
            self._model_profile_constraints_ = []
            self._active_constraints_ = ()
            self._fit_meta = None

    @tr.default("_components")
    def _components_default(self):
        if self._fitted:
            # note: _outfit = (C, St, C_constrained, St_ls, extraOutputGetConc, extraOutputGetSpec, ...)
            return self._outfit[1]
        raise NotFittedError("The model was not yet fitted. Execute `fit` first!")

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    # To see all accessible members it is interesting to use the structure tab of
    # PyCharm
    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        # should be a tuple of profiles or only concentrations/spectra profiles
        profiles = change.new
        if isinstance(profiles, list | tuple):
            # we assume that the starting C and St are already computed
            # (for ex. from a previous run of fit)
            C, St = (item.data for item in profiles)
            self._n_components = C.shape[1]
            # eventually remove mask
            if np.any(self._X_mask):
                masked_rows, masked_columns = self._get_masked_rc(self._X_mask)
                St = St[:, ~masked_columns]
                C = C[~masked_rows]
        else:
            # not passed explicitly, try to guess.
            C, St = self._guess_profile(profiles.data)

        # we do a last validation
        shape = self._X.shape
        if shape[0] != C.shape[0]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of C do not match those of X.")
        if shape[1] != St.shape[1]:
            # An error will be raised before if X is None.
            raise ValueError("The dimensions of St do not match those of X.")
        # return the list of C and St data
        # (with mask removed to fit the size of the _X data)
        self._Y_preprocessed = (C, St)

    # ----------------------------------------------------------------------------------
    # ALS iteration helpers
    #
    # The historical ``_fit`` body performed every step of the ALS loop inline.
    # The helpers below factor the responsibility of each step into a single
    # place so that ``_fit`` reads as a high level sequence of operations.
    # They are all private, do not change the public API and do not change the
    # numerical behavior: the order of operations and the in-place vs.
    # copy semantics are preserved exactly.
    # ----------------------------------------------------------------------------------

    def _fit(self, X, Y):
        """
        Run the MCR-ALS optimization.

        This method is called by the abstract ``fit``. ``X`` is a raw
        ``np.ndarray``; ``Y`` is a tuple of guessed profiles (each a
        ``np.ndarray``). All computation below operates on plain numpy arrays,
        never on ``NDDataset`` instances.
        """
        state = self._init_als_state(X, Y)

        # Build the constraint pipelines from public ``Constraint`` objects.
        # When ``constraints=`` was provided, use those directly; otherwise
        # convert legacy traitlets through ``legacy_to_constraints``.
        # Both paths converge to the same internal ``_Constraint`` pipeline.
        if self.constraints is not None:
            public_constraints = self.constraints
        else:
            from spectrochempy.analysis.decomposition._legacy_constraint_converter import (
                legacy_to_constraints,
            )

            public_constraints = legacy_to_constraints(self)

        self._active_constraints_ = tuple(public_constraints)

        (
            conc_constraints,
            spec_constraints,
            normalization,
        ) = self._build_from_public_constraints(public_constraints)

        # Store internal model-profile constraints so callers can inspect
        # mutable state (``model_args``, ``extra_outputs``, etc.) after fit.
        self._model_profile_constraints_ = [
            c
            for c in (*conc_constraints, *spec_constraints)
            if isinstance(c, _ModelProfileConstraint)
        ]

        # Keep component-presence constraints for final enforcement after the
        # complete concentration and spectral constraint sequence.
        self._presence_constraints_ = [
            c for c in conc_constraints if isinstance(c, _ComponentPresenceConstraint)
        ]

        # Pre-compute block slices (they do not change per iteration)
        conc_block_slices = (
            state.augmentation.row_slices if state.augmentation is not None else None
        )
        if state.augmentation is not None and state.augmentation.mode == "horizontal":
            spec_block_slices = state.augmentation.column_slices
            spec_block_axis = 1
        else:
            spec_block_slices = None
            spec_block_axis = 0

        if not any(
            tolerance is not None
            for tolerance in (
                self.tol_residual_change,
                self.tol_reconstruction_error,
                self.tol_profile_change,
            )
        ):
            raise ValueError(
                "At least one MCRALS convergence tolerance must be enabled."
            )

        while (
            not state.converged
            and state.niter < self.max_iter
            and state.ndiv < self.maxdiv
        ):
            state.niter += 1

            # 1. Solve C from the previously constrained St, then apply ALL
            #    concentration constraints (structural + ModelProfile) to
            #    produce the accepted constrained C for this iteration.
            #    The previously constrained St is state.St (initialized from
            #    guess or from the previous iteration's St constraint output).
            state.C = self._solve_C(state.St)
            state.C_ls = state.C.copy()
            state.C = self._apply_constraint_pipeline(
                state.C,
                conc_constraints,
                state,
                block_slices=conc_block_slices,
            )
            state.C_constrained = state.C.copy()

            # 2. Solve St from constrained C, snapshot the least-squares St,
            #    then apply soft + hard spectral constraints.
            state.St = self._solve_St(state.C)
            state.St_ls = state.St.copy()
            state.St = self._apply_constraint_pipeline(
                state.St,
                spec_constraints,
                state,
                block_slices=spec_block_slices,
                block_axis=spec_block_axis,
            )

            # 3. Normalize spectra / concentrations. Normalization is a
            #    single joint constraint (it mutates both St and C at
            #    once) that preserves the product C @ St.
            if normalization is not None:
                normalization.apply(state.St, state)

            # 4. Re-enforce component presence on the final accepted C. The
            #    pipeline pass is required before Trilinear to populate
            #    block_presence; this final pass protects against any later
            #    concentration constraint that writes into an absent block.
            #    Joint normalization alone only rescales columns and therefore
            #    cannot turn an exact zero into a non-zero value.
            self._apply_post_conc_presence(state)

            # Update C_constrained to reflect the final C (post-normalization,
            # post-presence).  This ensures that the public ``C_constrained``
            # property and the iteration history refer to the same factor
            # that convergence uses.
            state.C_constrained = state.C.copy()

            # 5. History & convergence: record iteration profiles and update
            #    the convergence counters / log.
            self._store_iteration(state)
            self._update_convergence(state)

        self._finalize_fit(state)
        return self._build_fit_result(state)

    def _init_als_state(self, X, Y):
        """
        Build the transient ``_ALSState`` for a fit and emit the log header.

        Pre-positions the convergence counters so the first iteration is
        executed. The log columns use the same relative quantities as the
        three public stopping tolerances.
        """
        C, St = Y
        n_components = self._n_components

        info_(
            "***                         ALS optimisation log                         ***"
        )
        info_("#iter  reconstruction_error  residual_change  profile_change  trend")
        info_("-----------------------------------------------------------------------")

        return _ALSState(
            X=X,
            C=C,
            St=St,
            n_components=n_components,
            stdev=X.std(),
            residual_change=np.inf,
            niter=0,
            ndiv=0,
            previous_C=C.copy(),
            previous_St=St.copy(),
            augmentation=getattr(self, "_augmented_structure", None),
        )

    # -- constraint pipeline ---------------------------------------------------
    #
    # ``_fit`` builds the constraint pipelines from public ``Constraint``
    # objects once per fit and iterates over them.  The pipeline is split
    # into two phases, matching the standard Tauler ALS formulation:
    #
    #   1. concentration constraints, applied to ``C`` *after* the C solve;
    #   2. spectral constraints, applied to ``St`` *after* the St solve.
    #
    # Normalization is kept out of the per-profile pipelines because it
    # operates on both ``C`` and ``St`` at once.  It is applied after the
    # spectral constraint pipeline.
    #
    # Both the ``constraints=`` API and the legacy traitlet API converge
    # here: legacy traitlets are first converted to public ``Constraint``
    # objects by ``legacy_to_constraints``, then processed identically
    # through the same ``_public_to_internal`` conversion path.

    def _build_normalization(self):
        """
        Build the normalization constraint, or ``None`` if disabled.

        ``normSpec`` is ``None`` by default; in that case no
        ``_NormalizationConstraint`` is built, so the post-solve step is a
        complete no-op. This matches the historical ``if self.normSpec ==``
        guard in ``_apply_normalization``.
        """
        if self.normSpec is None:
            return None
        return _NormalizationConstraint(
            method=self.normSpec,
            n_components=self._n_components,
        )

    def _build_from_public_constraints(self, constraints):
        """
        Build constraint pipelines from public ``Constraint`` objects.

        Converts each public ``Constraint`` to the corresponding private
        ``_Constraint``, splits by profile side, and returns the three
        pipeline components expected by ``_fit``.

        Parameters
        ----------
        constraints : list[Constraint]
            Public constraint objects (from ``constraints=`` or from
            ``legacy_to_constraints``).

        Returns
        -------
        conc_constraints : list[_Constraint]
            Concentration-side constraint pipeline.
        spec_constraints : list[_Constraint]
            Spectral-side constraint pipeline.
        normalization : _NormalizationConstraint or None
            Normalization constraint, or ``None`` if disabled.
        """
        conc = []
        spec = []

        for c in constraints:
            internal = self._public_to_internal(c)
            if c.profile == "C":
                conc.append(internal)
            else:
                spec.append(internal)

        # Ensure that _ComponentPresenceConstraint runs before
        # _TrilinearConstraint so that block_presence is always
        # populated when Trilinear needs it, regardless of user order.
        presence_idx = None
        trilinear_idx = None
        for i, c in enumerate(conc):
            if isinstance(c, _ComponentPresenceConstraint):
                presence_idx = i
            elif isinstance(c, _TrilinearConstraint):
                trilinear_idx = i
        if (
            presence_idx is not None
            and trilinear_idx is not None
            and presence_idx > trilinear_idx
        ):
            # Swap so presence precedes trilinear
            conc[presence_idx], conc[trilinear_idx] = (
                conc[trilinear_idx],
                conc[presence_idx],
            )

        normalization = self._build_normalization()

        return conc, spec, normalization

    def _resolve_components(self, components):
        """
        Resolve a public constraint's component selection to a list of indices.

        ``None`` (meaning "all components") is expanded to
        ``list(range(self._n_components))``. A concrete list is passed through.

        This mirrors what the legacy traitlet validators do when they
        convert ``"all"`` to ``np.arange(self._n_components).tolist()``.
        """
        if components is None:
            return list(range(self._n_components))
        return components

    def _resolve_blocks(self, blocks, profile=None):
        """
        Resolve a public constraint's block selection.

        ``None`` (meaning "all blocks") is passed through as ``None``.
        A concrete list is validated against the current augmented structure.
        On non-augmented data, an explicit block selection raises an error.

        Parameters
        ----------
        profile : str or None
            ``"C"`` or ``"St"``.  In horizontal mode the number of blocks
            differs between concentration (1 block) and spectral
            (``n_column_blocks``) profiles.
        """
        if blocks is None:
            return None
        aug = getattr(self, "_augmented_structure", None)
        if aug is None:
            raise ValueError(
                "Block indices can only be specified when fitting "
                "augmented (multiset) data."
            )
        if aug is not None and aug.mode == "horizontal" and profile == "St":
            n_blocks = aug.n_column_blocks
        else:
            n_blocks = aug.n_row_blocks
        for b in blocks:
            if b < 0 or b >= n_blocks:
                raise ValueError(
                    f"Block index {b} is out of range. "
                    f"Data has {n_blocks} block{'s' if n_blocks != 1 else ''} "
                    f"(valid indices: 0-{n_blocks - 1})."
                )
        return blocks

    def _public_to_internal(self, constraint):
        """
        Convert a single public ``Constraint`` to a private ``_Constraint``.

        Parameters
        ----------
        constraint : Constraint
            Public constraint object.

        Returns
        -------
        _Constraint
            Internal constraint.

        Raises
        ------
        NotImplementedError
            If the public constraint type has no internal counterpart yet.
        """
        from spectrochempy.analysis.decomposition.mcrals_constraints import Closure
        from spectrochempy.analysis.decomposition.mcrals_constraints import (
            ComponentPresence,
        )
        from spectrochempy.analysis.decomposition.mcrals_constraints import ModelProfile
        from spectrochempy.analysis.decomposition.mcrals_constraints import Monotonic
        from spectrochempy.analysis.decomposition.mcrals_constraints import NonNegative
        from spectrochempy.analysis.decomposition.mcrals_constraints import Trilinear
        from spectrochempy.analysis.decomposition.mcrals_constraints import Unimodal

        blocks = self._resolve_blocks(constraint.blocks, profile=constraint.profile)

        if isinstance(constraint, NonNegative):
            indices = self._resolve_components(constraint.components)
            axis = 0 if constraint.profile == "C" else 1
            return _NonNegativeConstraint(indices, axis, blocks=blocks)

        if isinstance(constraint, Unimodal):
            indices = self._resolve_components(constraint.components)
            axis = 0 if constraint.profile == "C" else 1
            return _UnimodalConstraint(
                indices,
                axis=axis,
                tol=constraint.tolerance,
                mod=constraint.mod,
                blocks=blocks,
            )

        if isinstance(constraint, Monotonic):
            if constraint.profile != "C":
                raise NotImplementedError(
                    f"Monotonic({constraint.profile!r}) is not implemented. "
                    f"Only profile='C' is supported."
                )
            indices = self._resolve_components(constraint.components)
            tol = constraint.tolerance
            if constraint.direction == "increasing":
                return _MonotonicIncreaseConstraint(indices, tol=tol, blocks=blocks)
            return _MonotonicDecreaseConstraint(indices, tol=tol, blocks=blocks)

        if isinstance(constraint, Closure):
            if constraint.profile != "C":
                raise NotImplementedError(
                    f"Closure({constraint.profile!r}) is not implemented. "
                    f"Only profile='C' is supported."
                )
            indices = self._resolve_components(constraint.components)
            # Expand a scalar target to a 1-D array (matching the shape of
            # the constrained axis), mirroring the legacy trait validator.
            target = constraint.target
            if np.ndim(target) == 0:
                target = np.full(self._X.shape[0], target)
            return _ClosureConstraint(
                indices,
                method=constraint.method,
                target=target,
                blocks=blocks,
            )

        if isinstance(constraint, ComponentPresence):
            aug = getattr(self, "_augmented_structure", None)
            if aug is None:
                raise ValueError(
                    "ComponentPresence constraint requires augmented data."
                )
            presence = constraint.presence
            n_blocks = len(aug.row_slices)
            if len(presence) != n_blocks:
                raise ValueError(
                    f"Presence matrix has {len(presence)} rows but "
                    f"data has {n_blocks} blocks."
                )
            if self._n_components and any(
                len(row) != self._n_components for row in presence
            ):
                raise ValueError(
                    f"Each row of presence must have exactly "
                    f"{self._n_components} entries."
                )
            return _ComponentPresenceConstraint(presence, blocks=blocks)

        if isinstance(constraint, Trilinear):
            comps = self._resolve_components(constraint.components)
            return _TrilinearConstraint(
                components=comps,
                blocks=blocks,
                synchronization=constraint.synchronization,
            )

        if isinstance(constraint, ModelProfile):
            comps = self._resolve_components(constraint.components)
            if constraint.mapping is not None and len(constraint.mapping) != len(comps):
                raise ValueError(
                    f"mapping has length {len(constraint.mapping)} but "
                    f"components resolved to {len(comps)} components "
                    f"({comps!r}). The lengths must match."
                )
            side = "conc" if constraint.profile == "C" else "spec"
            return _ModelProfileConstraint(
                self,
                side=side,
                model=constraint.model,
                components=comps,
                model_args=constraint.model_args,
                model_kwargs=constraint.model_kwargs,
                profile_mapping=constraint.mapping,
                blocks=blocks,
            )

        raise NotImplementedError(
            f"{type(constraint).__name__} is not yet implemented in MCRALS."
        )

    def _apply_post_conc_presence(self, state):
        """
        Re-enforce ComponentPresence on the final accepted concentration factor.

        ComponentPresence first runs in the concentration pipeline, before
        Trilinear, so that it both zeros absent blocks and populates the
        ``block_presence`` metadata consumed by Trilinear. A second pass is
        necessary because a subsequent concentration constraint may write into
        those blocks. It runs after the spectral solve and joint normalization,
        immediately before the accepted factor is stored and evaluated.

        Normalization does not itself require this pass: multiplying a
        concentration column by a finite scale factor preserves exact zeros.
        """
        for pc in getattr(self, "_presence_constraints_", []):
            pc.apply(state.C, state)

    @staticmethod
    def _apply_constraint_pipeline(
        profile, constraints, state, block_slices=None, block_axis=0
    ):
        """
        Run a list of constraints in order against ``profile``.

        Each constraint may mutate ``profile`` in place or rebind it (see
        ``_UnimodalConstraint``); ``_apply_constraint_pipeline`` propagates
        the returned reference so every subsequent constraint sees the
        up-to-date array.

        When the data is augmented (``state.augmentation is not None``) and
        a constraint is block-local (``constraint.is_block_local``), the
        constraint is applied independently to each block slice, preventing
        artificial interactions between experiments. Constraints that are
        NOT block-local (e.g. ``_TrilinearConstraint``) see the full
        concatenated array and manage blocks internally.

        ``block_slices`` specifies which slices to use for per-block
        application.  For vertical augmentation of concentration profiles
        (C), pass ``state.augmentation.row_slices`` with ``block_axis=0``.
        For horizontal augmentation of spectral profiles (St), pass
        ``state.augmentation.column_slices`` with ``block_axis=1``.

        The optional ``_blocks`` attribute on the constraint limits which
        blocks receive the constraint. ``None`` means all compatible blocks.
        """
        aug = state.augmentation
        for constraint in constraints:
            if (
                aug is not None
                and constraint.is_block_local
                and block_slices is not None
            ):
                selected_slices = list(block_slices)
                if constraint._blocks is not None:
                    selected_slices = [block_slices[i] for i in constraint._blocks]
                for sl in selected_slices:
                    if block_axis == 0:
                        local = constraint.apply(profile[sl, :].copy(), state)
                        profile[sl, :] = local
                    else:
                        local = constraint.apply(profile[:, sl].copy(), state)
                        profile[:, sl] = local
            else:
                profile = constraint.apply(profile, state)
        return profile

    # -- external generator dispatch (used by ``_ModelProfileConstraint``) ----
    #
    # These helpers dispatch the external ``getConc`` / ``getSpec`` (now
    # ``ModelProfile``) generators. They remain on the estimator so that
    # ``_ModelProfileConstraint`` can call them through its back-reference
    # (``est._call_external_generator`` / ``est._unpack_generator_output``)
    # without duplicating the dispatch logic.

    def _call_external_generator(self, func, current, args, kwargs):
        """
        Dispatch a ``getConc`` / ``getSpec`` external generator.

        The historical dispatch chooses one of four call signatures
        depending on whether positional / keyword arguments are non-empty.
        Empty tuples / dicts trigger the bare call path; this preserves the
        exact behavior of the original inline ``if/elif/else`` chain.
        """
        has_args = args != ()
        has_kwargs = kwargs != {}
        if has_args and has_kwargs:
            return func(current, *args, **kwargs)
        if has_args and not has_kwargs:
            return func(current, *args)
        if not has_args and has_kwargs:
            return func(current, **kwargs)
        return func(current)

    def _unpack_generator_output(self, output):
        """
        Normalize the return value of an external generator.

        Returns ``(profiles, new_args, extra_output)`` where ``new_args`` is
        the second tuple element when present, or the ``_UNCHANGED`` sentinel
        when the generator returned a bare profile (so the caller leaves the
        previous args untouched). ``extra_output`` is a list containing the
        third tuple element if present, else an empty list.
        """
        extra_output = []
        if isinstance(output, tuple):
            profiles = output[0]
            new_args = output[1]
            if len(output) == 3:
                extra_output.append(output[2])
        else:
            profiles = output
            new_args = _UNCHANGED
        return profiles, new_args, extra_output

    # -- history / diagnostics -------------------------------------------------

    def _store_iteration(self, state):
        """
        Append the per-iteration profiles to the history lists.

        Only runs when ``storeIterations`` is True. The four lists are filled
        in the historical order so that index ``i`` of each list refers to the
        same iteration.
        """
        if self.storeIterations:
            state.C_constrained_list.append(state.C_constrained.copy())
            state.St_ls_list.append(state.St_ls.copy())
            C_ls = state.C_ls if state.C_ls is not None else state.C
            state.C_ls_list.append(C_ls.copy())
            state.St_constrained_list.append(state.St.copy())

    def _update_convergence(self, state):
        """
        Update all active convergence criteria and emit the log line.

        Residual change, reconstruction error, and profile change are
        dimensionless relative fractions, both in the log and when compared
        with their tolerances. Convergence is reached as soon as any enabled
        criterion is satisfied.
        """
        Xhat = state.C @ state.St
        residual = Xhat - state.X
        stdev2 = np.std(residual)
        signed_delta = stdev2 - state.stdev
        denominator = max(abs(state.stdev), np.finfo(float).eps)
        signed_change = signed_delta / denominator
        residual_change = abs(signed_change)
        state.stdev = stdev2
        state.residual_change = float(residual_change)

        data_norm = np.linalg.norm(state.X, "fro")
        residual_norm = np.linalg.norm(residual, "fro")
        if data_norm <= np.finfo(float).eps:
            state.reconstruction_error = (
                0.0 if residual_norm <= np.finfo(float).eps else np.inf
            )
        else:
            state.reconstruction_error = float(residual_norm / data_norm)

        state.profile_change = self._relative_profile_change(
            state.previous_C,
            state.previous_St,
            state.C,
            state.St,
        )
        state.previous_C = state.C.copy()
        state.previous_St = state.St.copy()

        if signed_change < 0.0:
            trend = "down"
        elif signed_change > 0.0:
            trend = "up"
        else:
            trend = "flat"
        info_(
            f"{state.niter:5d}  {state.reconstruction_error:20.6e}  "
            f"{state.residual_change:15.6e}  {state.profile_change:14.6e}  "
            f"{trend:>5s}",
        )

        # Track consecutive residual increases independently of the stopping
        # criteria, preserving the historical maxdiv safeguard.
        if signed_change > 0:
            state.ndiv += 1
        else:
            state.ndiv = 0

        criteria = (
            (
                "residual_change",
                state.residual_change,
                self.tol_residual_change,
            ),
            (
                "reconstruction_error",
                state.reconstruction_error,
                self.tol_reconstruction_error,
            ),
            ("profile_change", state.profile_change, self.tol_profile_change),
        )
        stopping_criterion = None
        for name, value, tolerance in criteria:
            if tolerance is not None and value <= tolerance:
                state.converged = True
                state.convergence_reason = name
                stopping_criterion = (name, value, tolerance)
                break

        if stopping_criterion is not None:
            name, value, tolerance = stopping_criterion
            info_(f"Converged on {name}:")
            info_(f"  {name}={value:.6e} <= " f"tol_{name}={tolerance:.6e}")

        if not state.converged and state.ndiv == self.maxdiv:
            info_(
                f"Optimization not improved after {self.maxdiv} iterations"
                "... unconverged or convergence tolerances set too small ?",
            )
            info_("Stop ALS optimization.")

        if not state.converged and state.niter == self.max_iter:
            info_(
                "Convergence criterion not reached after "
                f"{state.niter: d} iterations.",
            )
            info_("Stop ALS optimization.")

    @staticmethod
    def _canonical_factor_pair(C, St):
        """Remove reciprocal scale and sign ambiguity from a factor pair."""
        C = np.asarray(C, dtype=float).copy()
        St = np.asarray(St, dtype=float).copy()
        eps = np.finfo(float).eps

        for component in range(C.shape[1]):
            scale = np.linalg.norm(C[:, component])
            if scale > eps:
                C[:, component] /= scale
                St[component, :] *= scale

            column = C[:, component]
            if np.any(column):
                pivot = column[np.argmax(np.abs(column))]
            elif np.any(St[component, :]):
                row = St[component, :]
                pivot = row[np.argmax(np.abs(row))]
            else:
                pivot = 1.0
            if pivot < 0:
                C[:, component] *= -1.0
                St[component, :] *= -1.0

        return C, St

    @classmethod
    def _relative_profile_change(cls, previous_C, previous_St, C, St):
        """Return the scale/sign-invariant relative change of a factor pair."""
        if previous_C is None or previous_St is None:
            return np.inf

        previous_C, previous_St = cls._canonical_factor_pair(previous_C, previous_St)
        C, St = cls._canonical_factor_pair(C, St)
        eps = np.finfo(float).eps
        C_change = np.linalg.norm(C - previous_C, "fro") / max(
            np.linalg.norm(previous_C, "fro"), eps
        )
        St_change = np.linalg.norm(St - previous_St, "fro") / max(
            np.linalg.norm(previous_St, "fro"), eps
        )
        return float(max(C_change, St_change))

    # -- finalization ----------------------------------------------------------

    def _finalize_fit(self, state):
        """Persist ALS diagnostics and set the fitted spectral components."""
        # capture ALS diagnostics for the result property
        self._fit_meta = {
            "n_iter": state.niter,
            "residual_change": state.residual_change,
            "reconstruction_error": state.reconstruction_error,
            "profile_change": state.profile_change,
            "residual_std": state.stdev,
            "converged": state.converged,
            "convergence_reason": state.convergence_reason,
        }

        # Constraint diagnostics: distance from each raw, unconstrained LS
        # estimate to the final accepted factor. The accepted factors include
        # the configured constraint pipelines, joint normalization, and final
        # ComponentPresence enforcement where applicable.
        _eps = 1.0e-15

        C_ls = state.C_ls
        C_con = state.C
        St_ls = state.St_ls
        St_con = state.St

        constraint_diagnostics = {}

        if C_ls is not None:
            # Global projection distance for C
            d_C_num = np.linalg.norm(C_con - C_ls, "fro")
            d_C_den = max(np.linalg.norm(C_ls, "fro"), _eps)
            d_C = d_C_num / d_C_den
            constraint_diagnostics["C"] = {"relative_projection_distance": d_C}

            # Per-component projection distance for C
            d_C_comp = []
            for k in range(C_con.shape[1]):
                num = np.linalg.norm(C_con[:, k] - C_ls[:, k])
                den = max(np.linalg.norm(C_ls[:, k]), _eps)
                d_C_comp.append(num / den)
            constraint_diagnostics["C"][
                "relative_projection_distance_by_component"
            ] = np.array(d_C_comp)

        if St_ls is not None:
            # Global projection distance for St
            d_St_num = np.linalg.norm(St_con - St_ls, "fro")
            d_St_den = max(np.linalg.norm(St_ls, "fro"), _eps)
            d_St = d_St_num / d_St_den
            constraint_diagnostics["St"] = {"relative_projection_distance": d_St}

            # Per-component projection distance for St
            d_St_comp = []
            for k in range(St_con.shape[0]):
                num = np.linalg.norm(St_con[k, :] - St_ls[k, :])
                den = max(np.linalg.norm(St_ls[k, :]), _eps)
                d_St_comp.append(num / den)
            constraint_diagnostics["St"][
                "relative_projection_distance_by_component"
            ] = np.array(d_St_comp)

        # Conditional LS distance: how far is the constrained pair from
        # the unconstrained LS refit of one factor given the other.
        if C_ls is not None and St_ls is not None:
            X = state.X
            # C_refit = X @ pinv(St_con)  (LS solve for C given St_con)
            # St has shape (n_comp, n_feat), pinv(St) has shape (n_feat, n_comp)
            C_refit = X @ np.linalg.pinv(St_con)
            d_C_refit_num = np.linalg.norm(C_con - C_refit, "fro")
            d_C_refit_den = max(np.linalg.norm(C_refit, "fro"), _eps)
            d_C_refit = d_C_refit_num / d_C_refit_den

            # St_refit = pinv(C_con) @ X  (LS solve for St given C_con)
            # C has shape (n_obs, n_comp), pinv(C) has shape (n_comp, n_obs)
            St_refit = np.linalg.pinv(C_con) @ X
            d_St_refit_num = np.linalg.norm(St_con - St_refit, "fro")
            d_St_refit_den = max(np.linalg.norm(St_refit, "fro"), _eps)
            d_St_refit = d_St_refit_num / d_St_refit_den

            constraint_diagnostics["conditional_ls_distance"] = {
                "C": d_C_refit,
                "St": d_St_refit,
            }

        if constraint_diagnostics:
            self._fit_meta["constraint_diagnostics"] = constraint_diagnostics

        self._components = state.St

    def _build_fit_result(self, state):
        """
        Assemble the tuple returned to the abstract ``fit`` / ``_outfit``.

        The shape and order of this tuple are part of the public contract
        (indices are referenced by the ``C_constrained``, ``St_ls``,
        ``extraOutputGetConc`` ... properties) and must not change.
        """
        return (
            state.C,
            state.St,
            state.C_constrained,
            state.St_ls,
            state.extra_output_conc,
            state.extra_output_spec,
            state.C_constrained_list,
            state.C_ls_list,
            state.St_constrained_list,
            state.St_ls_list,
        )

    def _transform(self, X=None):
        # X is ignored for MCRALS
        return self._outfit[0]

    def _inverse_transform(self, X_transform=None):
        # X_transform is ignored for MCRALS
        return np.dot(self._transform(), self._components)

    def _get_components(self):
        return self._components

    # -- factor metadata -------------------------------------------------------

    @staticmethod
    def _snapshot_factor_metadata(value):
        """Copy title and value units before fit preprocessing drops them."""
        from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

        if not isinstance(value, NDDataset):
            return _FactorMetadata()
        title = value.title
        if title in (None, "", "<untitled>"):
            title = None
        return _FactorMetadata(title=title, units=value.units)

    @staticmethod
    def _profile_array(profile):
        """Return a profile as an ndarray without changing its metadata."""
        return np.asarray(profile.data if hasattr(profile, "data") else profile)

    @staticmethod
    def _common_block_metadata(metadata, label):
        """Validate one shared physical convention for block factor guesses."""
        if not metadata:
            return _FactorMetadata()

        have_units = [item.units is not None for item in metadata]
        if any(have_units) and not all(have_units):
            raise ValueError(f"{label} blocks must all define the same units.")
        if all(have_units):
            reference = metadata[0].units
            if any(item.units != reference for item in metadata[1:]):
                raise ValueError(f"{label} blocks have inconsistent units.")
        else:
            reference = None

        titles = {item.title for item in metadata if item.title is not None}
        if len(titles) > 1:
            raise ValueError(f"{label} blocks have inconsistent titles.")
        title = next(iter(titles), None)
        return _FactorMetadata(title=title, units=reference)

    def _capture_factor_metadata(self, X_blocks, Y, mode=None):
        """Capture initial metadata and classify the supplied factor guess."""
        X_metadata = tuple(self._snapshot_factor_metadata(item) for item in X_blocks)
        C = _FactorMetadata()
        St = _FactorMetadata()
        C_blocks = ()
        St_blocks = ()

        if mode == "vertical" and isinstance(Y, list):
            C_blocks = tuple(self._snapshot_factor_metadata(item) for item in Y)
            C = self._common_block_metadata(C_blocks, "C0")
        elif mode == "horizontal" and isinstance(Y, list):
            St_blocks = tuple(self._snapshot_factor_metadata(item) for item in Y)
        elif isinstance(Y, (list, tuple)):
            if len(Y) == 2:
                C = self._snapshot_factor_metadata(Y[0])
                St = self._snapshot_factor_metadata(Y[1])
        else:
            profile = self._profile_array(Y)
            if mode == "vertical":
                n_observations = sum(
                    self._profile_array(item).shape[0] for item in X_blocks
                )
            else:
                n_observations = self._profile_array(X_blocks[0]).shape[0]
            if profile.shape[0] == n_observations:
                C = self._snapshot_factor_metadata(Y)
            else:
                St = self._snapshot_factor_metadata(Y)

        context = _FactorMetadataContext(
            mode=mode,
            X_blocks=X_metadata,
            C=C,
            St=St,
            C_blocks=C_blocks,
            St_blocks=St_blocks,
        )
        self._factor_metadata_ = self._resolve_factor_metadata(context)

    @staticmethod
    def _validate_factor_unit_product(X_units, C_units, St_units):
        """Require the supplied factor dimensions to satisfy [X] = [C][St]."""
        if X_units is None or C_units is None or St_units is None:
            return
        product = C_units * St_units
        if product.dimensionality != X_units.dimensionality:
            raise ValueError(
                "incompatible units for C0 and St0 relative to X: "
                f"{C_units} * {St_units} does not match {X_units}."
            )

    @staticmethod
    def _common_derived_units(candidates):
        """Return one common derived unit, or None for mixed/missing blocks."""
        if not candidates or any(unit is None for unit in candidates):
            return None
        reference = candidates[0]
        if any(unit != reference for unit in candidates[1:]):
            return None
        return reference

    def _scale_is_physically_preserved(self):
        """Whether configured operations retain the guess-defined factor scale."""
        if self.normSpec is not None:
            return False

        if self.constraints is not None:
            from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: PLC0415
                Closure,
            )
            from spectrochempy.analysis.decomposition.mcrals_constraints import (  # noqa: PLC0415
                ModelProfile,
            )

            return not any(
                isinstance(item, (Closure, ModelProfile)) for item in self.constraints
            )

        return not (
            bool(self.closureConc)
            or bool(self.hardConc)
            or bool(self.hardSpec)
            or self.getConc is not None
            or self.getSpec is not None
        )

    def _resolve_factor_metadata(self, context):
        """Resolve titles and units without touching factor numerical arrays."""
        X_blocks = context.X_blocks
        scale_preserved = self._scale_is_physically_preserved()
        C_units = context.C.units
        St_units = context.St.units

        supplied_St_blocks = (
            context.St_blocks
            if context.St_blocks
            else tuple(context.St for _ in X_blocks)
        )

        # Validate every supplied calibrated factor pair before considering
        # whether a later operation will clear the output units.
        if C_units is not None and St_units is not None:
            for X_meta in X_blocks:
                self._validate_factor_unit_product(X_meta.units, C_units, St_units)
        if C_units is not None and context.St_blocks:
            for X_meta, St_meta in zip(X_blocks, context.St_blocks, strict=True):
                self._validate_factor_unit_product(X_meta.units, C_units, St_meta.units)

        if context.mode == "horizontal":
            block_St_units = [item.units for item in supplied_St_blocks]
            if C_units is None:
                C_units = self._common_derived_units(
                    [
                        X_meta.units / block_units
                        if X_meta.units is not None and block_units is not None
                        else None
                        for X_meta, block_units in zip(
                            X_blocks, block_St_units, strict=True
                        )
                    ]
                )
            if C_units is not None:
                for i, (X_meta, block_units) in enumerate(
                    zip(X_blocks, block_St_units, strict=True)
                ):
                    if block_units is None and X_meta.units is not None:
                        block_St_units[i] = X_meta.units / C_units

            if not scale_preserved:
                C_units = None
                block_St_units = [None] * len(X_blocks)

            C_title = (
                context.C.title
                if C_units is not None
                and context.C.units is not None
                and context.C.title
                else "concentration"
                if C_units is not None
                else "relative concentration"
            )
            C_meta = _FactorMetadata(C_title, C_units)
            St_block_meta = tuple(
                _FactorMetadata(
                    (
                        source.title
                        if units is not None
                        and source.units is not None
                        and source.title
                        else X_meta.title or "spectral profiles"
                    ),
                    units,
                )
                for X_meta, source, units in zip(
                    X_blocks, supplied_St_blocks, block_St_units, strict=True
                )
            )
            return _ResolvedFactorMetadata(
                C=C_meta,
                St=_FactorMetadata("concatenated spectral profiles", None),
                C_blocks=(C_meta,),
                St_blocks=St_block_meta,
                scale_is_physically_preserved=scale_preserved,
            )

        if C_units is None and St_units is not None:
            C_units = self._common_derived_units(
                [
                    X_meta.units / St_units if X_meta.units is not None else None
                    for X_meta in X_blocks
                ]
            )
        if St_units is None and C_units is not None:
            St_units = self._common_derived_units(
                [
                    X_meta.units / C_units if X_meta.units is not None else None
                    for X_meta in X_blocks
                ]
            )

        if not scale_preserved:
            C_units = None
            St_units = None

        C_title = (
            context.C.title
            if C_units is not None and context.C.units is not None and context.C.title
            else "concentration"
            if C_units is not None
            else "relative concentration"
        )
        St_title = (
            context.St.title
            if St_units is not None
            and context.St.units is not None
            and context.St.title
            else X_blocks[0].title or "spectral profiles"
        )
        C_meta = _FactorMetadata(C_title, C_units)
        St_meta = _FactorMetadata(St_title, St_units)
        n_C_blocks = len(X_blocks) if context.mode == "vertical" else 1
        return _ResolvedFactorMetadata(
            C=C_meta,
            St=St_meta,
            C_blocks=tuple(C_meta for _ in range(n_C_blocks)),
            St_blocks=(St_meta,),
            scale_is_physically_preserved=scale_preserved,
        )

    def _output_factor_metadata(self):
        """Resolve metadata, with conservative defaults for old fitted states."""
        return getattr(
            self,
            "_factor_metadata_",
            _ResolvedFactorMetadata(
                C=_FactorMetadata("relative concentration", None),
                St=_FactorMetadata("spectral profiles", None),
                C_blocks=(_FactorMetadata("relative concentration", None),),
                St_blocks=(_FactorMetadata("spectral profiles", None),),
                scale_is_physically_preserved=False,
            ),
        )

    @staticmethod
    def _apply_output_metadata(dataset, metadata):
        """Apply factor-value metadata without changing coordinates or values."""
        dataset.title = metadata.title
        dataset.units = metadata.units
        return dataset

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y, augmentation=None):
        """
        Fit the MCRALS model on an X dataset using initial concentration or spectra.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
            For augmented (multiset) analysis, a list or tuple of datasets
            can be passed. The augmentation mode is inferred from the
            input shapes (vertical if all share columns, horizontal if
            all share rows).  When all datasets have identical dimensions,
            an ambiguity error is raised and ``augmentation`` must be
            specified explicitly.

            .. note::

               Passing a single dataset in a list::

                   mcr.fit([X], guess)

               explicitly requests the augmented-dataset workflow and
               creates a single augmentation block.

               Passing the dataset directly::

                   mcr.fit(X, guess)

               uses the standard single-dataset workflow.

               Both approaches produce equivalent numerical MCR-ALS
               results, but the first exposes augmentation-specific
               features such as :attr:`is_augmented`, :attr:`C_blocks`,
               and :attr:`St_blocks`.

            Example::

                mcr.fit([X1, X2, X3], guess)

        Y : :term:`array-like` or list of :term:`array-like`
            Initial concentration or spectra. Unit-bearing `NDDataset` guesses
            define the physical factor scale when no configured operation
            subsequently resets it. A ``(C0, St0)`` pair with units is checked
            for dimensional compatibility with ``X``.

        augmentation : str, optional
            Force the augmentation mode when it cannot be inferred.
            One of ``"vertical"`` or ``"horizontal"``.

        Returns
        -------
        self
            The fitted instance itself.

        See Also
        --------
        transform : Apply dimensionality reduction.
        fit_transform : Fit the model and apply dimensionality reduction.

        """
        if isinstance(X, (list, tuple)):
            if not X:
                raise ValueError("X list is empty.")
            return self._fit_augmented(X, Y, augmentation=augmentation)
        self._augmented_structure = None
        self._X_inputs = None
        self._capture_factor_metadata((X,), Y, mode=None)
        return super().fit(X, Y)

    def _fit_augmented(self, X_list, Y, augmentation=None):
        """
        Fit MCRALS on a list of augmented datasets.

        Supports both vertical and horizontal augmentation:

        * **Vertical** — all datasets share the same number of columns
          (spectral variables). They are stacked vertically; the
          concentration profiles C are block-specific while the spectra
          St are common.

        * **Horizontal** — all datasets share the same number of rows
          (observations). They are stacked horizontally; the
          concentration profiles C are common while the spectra St are
          block-specific.

        The mode is inferred from the input shapes unless ``augmentation``
        is explicitly set to ``"vertical"`` or ``"horizontal"``.
        """
        n = len(X_list)

        # Convert all to numpy arrays for shape validation
        arrays = []
        datasets = []
        from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

        for i, xi in enumerate(X_list):
            if isinstance(xi, NDDataset):
                datasets.append(xi)
                arrays.append(np.asarray(xi.data))
            else:
                arrays.append(np.asarray(xi))

            if arrays[-1].ndim != 2:
                raise ValueError(
                    f"X[{i}] must be 2-dimensional, " f"got shape {arrays[-1].shape}."
                )

        # ---------------------------------------------------------------
        # Determine augmentation mode
        # ---------------------------------------------------------------
        if augmentation is not None:
            if augmentation not in ("vertical", "horizontal"):
                raise ValueError(
                    f"augmentation={augmentation!r} is not valid. "
                    f"Choose 'vertical' or 'horizontal'."
                )
            mode = augmentation
        else:
            n_cols_set = {arr.shape[1] for arr in arrays}
            n_rows_set = {arr.shape[0] for arr in arrays}
            all_same_cols = len(n_cols_set) == 1
            all_same_rows = len(n_rows_set) == 1

            if all_same_cols and not all_same_rows:
                mode = "vertical"
            elif all_same_rows and not all_same_cols:
                mode = "horizontal"
            elif all_same_cols and all_same_rows:
                if n == 1:
                    # Single dataset — arbitrary convention: default to vertical.
                    mode = "vertical"
                else:
                    raise ValueError(
                        "Cannot infer augmentation mode because all datasets "
                        "have identical dimensions. "
                        "Specify augmentation='vertical' or 'horizontal'."
                    )
            else:
                raise ValueError(
                    f"Cannot infer augmentation mode from shapes "
                    f"{[arr.shape for arr in arrays]}. "
                    "For vertical augmentation, all datasets must share the "
                    "same number of columns. For horizontal augmentation, "
                    "all datasets must share the same number of rows."
                )

        # ---------------------------------------------------------------
        # Mode-specific validation and slice construction
        # ---------------------------------------------------------------
        if mode == "vertical":
            # Validate column (spectral) compatibility
            n_cols = arrays[0].shape[1]
            for i, arr in enumerate(arrays):
                if arr.shape[1] != n_cols:
                    raise ValueError(
                        f"All vertically augmented datasets must have the same "
                        f"number of columns (spectral variables). "
                        f"X[0] has {n_cols} but X[{i}] has {arr.shape[1]}."
                    )

            # Check spectral coordinates
            if datasets and all(hasattr(d, "coordset") for d in datasets):
                coordset_0 = datasets[0].coordset
                spec_coord_0 = coordset_0[0] if coordset_0 is not None else None
                if spec_coord_0 is not None:
                    from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415

                    for i in range(1, n):
                        coordset_i = datasets[i].coordset
                        spec_coord_i = coordset_i[0] if coordset_i is not None else None
                        if spec_coord_i is None:
                            raise ValueError(
                                f"Dataset X[{i}] has no spectral coordinate."
                            )
                        if (
                            isinstance(spec_coord_0, Coord)
                            and isinstance(spec_coord_i, Coord)
                            and not np.array_equal(
                                np.asarray(spec_coord_0.data),
                                np.asarray(spec_coord_i.data),
                            )
                        ):
                            raise ValueError(
                                "All vertically augmented datasets must share "
                                "a compatible spectral axis."
                            )

            # Build row slices (one per block)
            row_slices = []
            input_shapes = []
            start = 0
            for arr in arrays:
                n_rows = arr.shape[0]
                row_slices.append(slice(start, start + n_rows))
                input_shapes.append(arr.shape)
                start += n_rows

            col_slices = (slice(0, arrays[0].shape[1]),)
            X_concat = np.vstack(arrays)

            # For metadata, use first dataset's spectral coord
            concat_coords = [
                None,
                datasets[0].coordset[0]
                if datasets and datasets[0].coordset is not None
                else None,
            ]

        else:  # horizontal
            # Validate row (observation) compatibility
            n_rows = arrays[0].shape[0]
            for i, arr in enumerate(arrays):
                if arr.shape[0] != n_rows:
                    raise ValueError(
                        f"All horizontally augmented datasets must have the same "
                        f"number of rows (observations). "
                        f"X[0] has {n_rows} but X[{i}] has {arr.shape[0]}."
                    )

            # Check observation coordinates compatibility
            if datasets and all(hasattr(d, "coordset") for d in datasets):
                coordset_0 = datasets[0].coordset
                obs_coord_0 = coordset_0[1] if coordset_0 is not None else None
                if obs_coord_0 is not None:
                    from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415

                    for i in range(1, n):
                        coordset_i = datasets[i].coordset
                        obs_coord_i = coordset_i[1] if coordset_i is not None else None
                        if obs_coord_i is None:
                            raise ValueError(
                                f"Dataset X[{i}] has no observation coordinate."
                            )
                        if (
                            isinstance(obs_coord_0, Coord)
                            and isinstance(obs_coord_i, Coord)
                            and not np.array_equal(
                                np.asarray(obs_coord_0.data),
                                np.asarray(obs_coord_i.data),
                            )
                        ):
                            raise ValueError(
                                "All horizontally augmented datasets must share "
                                "a compatible observation axis."
                            )

            # Build column slices (one per block)
            col_slices = []
            input_shapes = []
            start = 0
            for arr in arrays:
                n_cols = arr.shape[1]
                col_slices.append(slice(start, start + n_cols))
                input_shapes.append(arr.shape)
                start += n_cols

            row_slices = (slice(0, n_rows),)
            X_concat = np.hstack(arrays)

            # For metadata, use first dataset's observation coord
            concat_coords = [
                datasets[0].coordset[1]
                if datasets and datasets[0].coordset is not None
                else None,
                None,
            ]

        # ---------------------------------------------------------------
        # Build _AugmentedStructure
        # ---------------------------------------------------------------
        aug = _AugmentedStructure(
            mode=mode,
            row_slices=tuple(row_slices),
            column_slices=tuple(col_slices),
            input_shapes=tuple(input_shapes),
        )

        self._augmented_structure = aug
        self._X_inputs = tuple(item.copy() for item in datasets) if datasets else None
        if isinstance(Y, list) and len(Y) != n:
            raise ValueError(
                f"Number of initial profile blocks ({len(Y)}) must match "
                f"number of X blocks ({n})."
            )
        self._capture_factor_metadata(tuple(X_list), Y, mode=mode)

        # ---------------------------------------------------------------
        # Build an NDDataset for the concatenated data to preserve metadata
        # ---------------------------------------------------------------
        if datasets:
            from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

            concat_dataset = NDDataset(X_concat, coordset=concat_coords)
            if datasets[0].units is not None:
                concat_dataset.units = datasets[0].units
            if datasets[0].title is not None:
                concat_dataset.title = datasets[0].title
            X_fit = concat_dataset
        else:
            X_fit = X_concat

        # ---------------------------------------------------------------
        # Process list of per-block initial profiles
        # ---------------------------------------------------------------
        if isinstance(Y, list):
            if len(Y) != n:
                raise ValueError(
                    f"Number of initial profile blocks ({len(Y)}) must match "
                    f"number of X blocks ({n})."
                )
            if mode == "vertical":
                # Y is a list of C0 blocks — one per block, same n_components
                C0_blocks = []
                n_comp = None
                for i, yi in enumerate(Y):
                    yi_arr = np.asarray(yi.data if hasattr(yi, "data") else yi)
                    if yi_arr.ndim != 2:
                        raise ValueError(
                            f"C0[{i}] must be 2-dimensional, "
                            f"got shape {yi_arr.shape}."
                        )
                    if n_comp is None:
                        n_comp = yi_arr.shape[1]
                    elif yi_arr.shape[1] != n_comp:
                        raise ValueError(
                            f"All C0 blocks must have the same number of "
                            f"components. Block 0 has {n_comp} but "
                            f"block {i} has {yi_arr.shape[1]}."
                        )
                    if yi_arr.shape[0] != arrays[i].shape[0]:
                        raise ValueError(
                            f"C0 block {i} has {yi_arr.shape[0]} rows but "
                            f"X[{i}] has {arrays[i].shape[0]} rows."
                        )
                    C0_blocks.append(yi_arr)
                Y = np.vstack(C0_blocks)
            else:  # horizontal
                # Y is a list of St0 blocks — one per block, same n_components
                St0_blocks = []
                n_comp = None
                for i, yi in enumerate(Y):
                    yi_arr = np.asarray(yi.data if hasattr(yi, "data") else yi)
                    if yi_arr.ndim != 2:
                        raise ValueError(
                            f"St0[{i}] must be 2-dimensional, "
                            f"got shape {yi_arr.shape}."
                        )
                    if n_comp is None:
                        n_comp = yi_arr.shape[0]
                    elif yi_arr.shape[0] != n_comp:
                        raise ValueError(
                            f"All St0 blocks must have the same number of "
                            f"components. Block 0 has {n_comp} but "
                            f"block {i} has {yi_arr.shape[0]}."
                        )
                    if yi_arr.shape[1] != arrays[i].shape[1]:
                        raise ValueError(
                            f"St0 block {i} has {yi_arr.shape[1]} columns but "
                            f"X[{i}] has {arrays[i].shape[1]} columns."
                        )
                    St0_blocks.append(yi_arr)
                Y = np.hstack(St0_blocks)

        return super().fit(X_fit, Y)

    def fit_transform(self, X, Y, **kwargs):
        """
        Fit the model with ``X`` and apply the dimensionality reduction on ``X``.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : :term:`array-like` or list of :term:`array-like`
            Initial concentration or spectra.

        Returns
        -------
        `NDDataset`
            Transformed data with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction.

        """
        return super().fit_transform(X, Y, **kwargs)

    def transform(self, X=None, **kwargs):
        """
        Return accepted concentrations with MCRALS factor metadata.

        Parameters
        ----------
        X : `NDDataset` or array-like, optional
            Input dataset. MCRALS returns the fitted concentration factor;
            coordinates are taken from the fitted input dataset.
        **kwargs
            Additional transformation options handled by the base estimator.

        Returns
        -------
        `NDDataset`
            Final accepted concentration profiles with resolved factor title,
            units, observation coordinate, and component labels.
        """
        C = super().transform(X, **kwargs)
        return self._apply_output_metadata(C, self._output_factor_metadata().C)

    def inverse_transform(self, X_transform=None, **kwargs):
        r"""
        Transform data back to its original space.

        In other words, return an input `X_original` whose reduce/transform would be X.

        Parameters
        ----------
        X_transform : array-like of shape (:term:`n_observations`, :term:`n_components`), optional
            Reduced `X` data. If not provided, the transform of the fitted X is used.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_features`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction.

        """
        return super().inverse_transform(X_transform, **kwargs)

    @property
    def C(self):
        """
        The final accepted concentration profiles.

        ``C`` is the factor accepted in the last ALS iteration after applying
        the configured concentration constraint pipeline and joint spectral
        normalization. If no configured operation changes the least-squares
        estimate, it is accepted unchanged. Together with ``St`` it satisfies
        ``C @ St ≈ X`` (reconstruction of the input data), where the
        reconstruction fidelity depends on the active constraints.

        For non-augmented and vertically augmented fits, the observation
        axis (rows) carries the observation coordinate of the original
        input dataset.

        For horizontally augmented fits, all blocks share the same
        observation axis.  ``C`` preserves this shared coordinate
        (e.g. temperature) on its observation dimension.

        The component axis (columns) carries component labels obtained
        from ``_get_component_labels``.

        Value metadata follow the MCRALS scale policy: a calibrated ``C0``
        supplies the title and units, or the units are derived from calibrated
        ``St0`` as ``X.units / St0.units``. Without a preserved calibrated
        scale, the title is ``"relative concentration"`` and units are absent.
        """
        C = self.transform()
        self._apply_output_metadata(C, self._output_factor_metadata().C)
        C.name = "Pure concentration profile, mcs-als of " + self.X.name
        return C

    @property
    def St(self):
        """
        The final accepted spectral profiles.

        ``St`` is the factor accepted in the last ALS iteration after applying
        the configured spectral constraint pipeline and normalization. If no
        configured operation changes the least-squares estimate, it is
        accepted unchanged. Together with ``C`` it satisfies ``C @ St ≈ X``
        (reconstruction of the input data).

        For non-augmented and vertically augmented fits, ``St`` carries
        the original spectral coordinate on its feature axis.

        For horizontally augmented fits, the spectral axis is the
        concatenation of all blocks' feature axes.  Because different
        blocks may have incompatible spectral coordinates (e.g. UV
        wavelength vs CD wavelength), the global ``St`` uses a generic
        feature index.  Use ``St_blocks`` to obtain per-block spectral
        matrices with the original physical coordinates.

        A calibrated ``St0`` supplies value title and units, or units are
        derived from ``X.units / C0.units``. Without a preserved calibrated
        scale, units are absent. For horizontal augmentation the global
        concatenated factor is always unitless with the neutral title
        ``"concatenated spectral profiles"``; ``St_blocks`` is authoritative.
        """
        St = self.components
        self._apply_output_metadata(St, self._output_factor_metadata().St)
        St.name = "Pure spectra profile, mcs-als of " + self.X.name
        return St

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def C_constrained(self):
        """
        Final accepted concentration profiles.

        This compatibility output is synchronized with ``C`` after the
        configured concentration constraint pipeline, joint normalization,
        and final ComponentPresence enforcement.
        """
        return self._outfit[2]

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def St_ls(self):
        r"""
        The last spectral profiles obtained by least-square optimization, before constraints.

        Spectra obtained after solving :math:`C_{\textrm{constrained}} \cdot St = X`
        for :math:`St`.
        """
        return self._outfit[3]

    @property
    def extraOutputGetConc(self):
        """The extra outputs of the external function used to get concentrations."""
        return self._outfit[4]

    @property
    def extraOutputGetSpec(self):
        """The extra outputs of the external function used to get spectra."""
        return self._outfit[5]

    @property
    def C_constrained_list(self):
        """
        Final accepted concentration profiles by ALS iteration.

        Entries include the configured concentration constraint pipeline,
        joint normalization, and final ComponentPresence enforcement. They are
        independent snapshots and require `MCRALS.storeIterations` set to True.
        """
        return self._outfit[6]

    @property
    def C_ls_list(self):
        """
        Unconstrained concentration least-squares solutions by ALS iteration.

        Each entry is the unconstrained least-squares solution before
        concentration constraints and normalization. Entries are independent
        snapshots and require `MCRALS.storeIterations` set to True.

        """
        return self._outfit[7]

    @property
    def St_constrained_list(self):
        """
        Final accepted spectral profiles by ALS iteration.

        Entries include the configured spectral constraint pipeline and
        normalization. They are independent snapshots and require
        `MCRALS.storeIterations` set to True.
        """
        return self._outfit[8]

    @property
    def St_ls_list(self):
        """
        Unconstrained spectral least-squares solutions by ALS iteration.

        Each entry precedes spectral constraints and normalization. Entries are
        independent snapshots and require `MCRALS.storeIterations` set to True.
        """
        return self._outfit[9]

    # ----------------------------------------------------------------------------------
    # Augmented data properties
    # ----------------------------------------------------------------------------------

    @property
    def is_augmented(self):
        """Whether the fit was performed on augmented (multiset) data."""
        return self._augmented_structure is not None

    @property
    def augmented_structure(self):
        """
        The internal augmented data structure, or ``None`` for simple 2D data.

        Returns a read-only view of the ``_AugmentedStructure`` describing
        the block structure of vertically-augmented datasets. Only available
        after fitting.
        """
        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        return self._augmented_structure

    @property
    def C_blocks(self):
        """
        Concentration profiles split by experiment (block).

        For vertically augmented data, one block per input dataset is
        returned, each with its own observation coordinate and title.

        For horizontally augmented data, a single block is returned
        (all blocks share the same concentration matrix ``C``).

        When the input datasets were :class:`NDDataset`, each block is
        returned as an :class:`NDDataset` carrying the original observation
        coordinates and title where available.  For plain array inputs,
        blocks are returned as :class:`numpy.ndarray`.

        The blocks are **copies** of the relevant rows of the concatenated
        ``C`` matrix.  Modifying a returned block does **not** affect the
        fitted estimator (unlike ``_outfit[0]`` which returns the internal
        array directly).

        Each dataset block carries the resolved ``C`` value title and units.
        Experimental ``X`` value units are never copied directly: calibrated
        units come from ``C0`` or are derived from ``X.units / St0.units``.
        Coordinate metadata remain block-specific and come from the matching
        input dataset.

        Returns
        -------
        tuple of :class:`numpy.ndarray` or :class:`NDDataset`
            One element per block.  For non-augmented data, a single-element
            tuple containing the full ``C`` array is returned.
        """
        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        C = self._outfit[0]
        aug = self._augmented_structure
        if aug is None:
            return (C,)

        from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

        blocks = []
        for i, sl in enumerate(aug.row_slices):
            block_arr = C[sl, :].copy()
            inputs = getattr(self, "_X_inputs", None)
            if inputs and i < len(inputs) and hasattr(inputs[i], "coordset"):
                ds = inputs[i]
                # ds.coordset[0] = 'x' (spectral), ds.coordset[1] = 'y' (observation).
                # For C blocks, columns are components not spectral features,
                # so keep only the observation coordinate.
                obs_coord = (
                    ds.coordset[1].copy()
                    if ds.coordset is not None and ds.coordset[1] is not None
                    else None
                )
                block = NDDataset(block_arr, coordset=(obs_coord, None))
                metadata = self._output_factor_metadata().C_blocks[i]
                self._apply_output_metadata(block, metadata)
                if ds.name is not None:
                    block.name = f"C_profile_{ds.name}"
                # Units are intentionally NOT copied from X (see note in docstring).
                blocks.append(block)
            else:
                blocks.append(block_arr)
        return tuple(blocks)

    @property
    def St_blocks(self):
        """
        Spectral profiles split by experiment (block).

        Only meaningful for horizontally augmented data where each block
        has its own spectral matrix.  For vertically augmented data (and
        for non-augmented data), ``St`` is common across blocks and this
        property returns a single-element tuple containing the full
        ``St`` matrix.

        When the input datasets were :class:`NDDataset`, each block is
        returned as an :class:`NDDataset` carrying:

        * the **spectral coordinate** (e.g. wavelength) from the
          corresponding input block on its feature axis;
        * **component labels** on its component axis.
        * block-specific value title and units from calibrated ``St0`` blocks,
          or units derived from the corresponding ``X`` block and calibrated
          ``C0``.

        For plain array inputs, blocks are returned as
        :class:`numpy.ndarray`.

        The blocks are **copies** of the relevant columns of the
        concatenated ``St`` matrix.  Modifying a returned block does
        **not** affect the fitted estimator.

        Returns
        -------
        tuple of :class:`numpy.ndarray` or :class:`NDDataset`
            One element per block.
        """
        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        St = self._outfit[1]
        aug = self._augmented_structure
        if aug is None:
            return (St,)

        from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415
        from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

        comp_labels = self._get_component_labels(self._n_components)
        comp_coord = Coord(
            None,
            labels=comp_labels,
            title="components",
        )

        blocks = []
        for i, sl in enumerate(aug.column_slices):
            block_arr = St[:, sl].copy()
            inputs = getattr(self, "_X_inputs", None)
            if inputs and i < len(inputs) and hasattr(inputs[i], "coordset"):
                ds = inputs[i]
                # ds.coordset[0] = 'x' (spectral), ds.coordset[1] = 'y' (observation).
                # For St blocks, keep only the spectral coordinate.
                spec_coord = (
                    ds.coordset[0].copy()
                    if ds.coordset is not None and ds.coordset[0] is not None
                    else None
                )
                block = NDDataset(block_arr, coordset=(comp_coord, spec_coord))
                metadata = self._output_factor_metadata().St_blocks[i]
                self._apply_output_metadata(block, metadata)
                if ds.name is not None:
                    block.name = f"St_profile_{ds.name}"
                blocks.append(block)
            else:
                blocks.append(block_arr)
        return tuple(blocks)

    # ----------------------------------------------------------------------------------
    # Result property
    # ----------------------------------------------------------------------------------

    @property
    def result(self):
        """
        ``AnalysisResult`` object wrapping the fitted MCRALS estimator.

        Returns
        -------
        AnalysisResult
            Container with ``parameters``, ``outputs``, and ``diagnostics``
            derived from the fitted estimator.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        parameters = {
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "tol_residual_change": self.tol_residual_change,
            "tol_reconstruction_error": self.tol_reconstruction_error,
            "tol_profile_change": self.tol_profile_change,
            "maxdiv": self.maxdiv,
            "solver_C": self.solver_C,
            "solver_St": self.solver_St,
            "constraints": tuple(repr(item) for item in self._active_constraints_),
            "warm_start": self.warm_start,
            "augmentation": (
                self._augmented_structure.mode
                if self._augmented_structure is not None
                else None
            ),
        }

        outputs = {
            "C": self.C,
            "components": self.components,
        }

        diagnostics = {}
        if self._fit_meta is not None:
            diagnostics = {k: v for k, v in self._fit_meta.items() if v is not None}

        return AnalysisResult(
            estimator="MCRALS",
            parameters=parameters,
            outputs=outputs,
            diagnostics=diagnostics,
        )


# --------------------------------------------------------------------------------------
# Public introspection (signature, docstring, repr)
# --------------------------------------------------------------------------------------

# Override the autogenerated __signature__ from @signature_has_configurable_traits
# so that only the modern public API is visible in help(), while legacy traitlets
# remain accessible through **kwargs for backward compatibility.
from inspect import Parameter
from inspect import Signature

MCRALS.__signature__ = Signature(
    [
        Parameter("args", kind=Parameter.VAR_POSITIONAL),
        Parameter("constraints", kind=Parameter.KEYWORD_ONLY, default=None),
        Parameter("solver_C", kind=Parameter.KEYWORD_ONLY, default="lstsq"),
        Parameter("solver_St", kind=Parameter.KEYWORD_ONLY, default="lstsq"),
        Parameter("max_iter", kind=Parameter.KEYWORD_ONLY, default=50),
        Parameter("tol_residual_change", kind=Parameter.KEYWORD_ONLY, default=1.0e-3),
        Parameter(
            "tol_reconstruction_error", kind=Parameter.KEYWORD_ONLY, default=None
        ),
        Parameter("tol_profile_change", kind=Parameter.KEYWORD_ONLY, default=None),
        Parameter("maxdiv", kind=Parameter.KEYWORD_ONLY, default=5),
        Parameter("warm_start", kind=Parameter.KEYWORD_ONLY, default=False),
        Parameter("log_level", kind=Parameter.KEYWORD_ONLY, default=logging.WARNING),
    ]
)

# The decorator @signature_has_configurable_traits strips manual entries for
# ``config=True`` traitlets (max_iter, tol, maxdiv, …) and merges auto‑generated
# entries for ALL such traitlets (including legacy names) into a single
# ``Parameters`` section.  We keep only entries whose name matches a param in the
# custom ``__signature__`` (excluding VAR_POSITIONAL ``*args``) and drop everything
# legacy.  This keeps the keep-list in sync with the public API automatically:
# adding a param to ``__signature__`` adds it to the docstring, and removing one
# removes it.
_doc = MCRALS.__doc__

if _doc:
    import re as _re

    _PH = "Parameters\n----------\n"
    _params_start = _doc.find(_PH)
    if _params_start >= 0:
        _before_params = _doc[:_params_start]
        _params_body = _doc[_params_start + len(_PH) :]

        # Derive keep-set and order directly from __signature__.
        _sig_params = [
            name
            for name in MCRALS.__signature__.parameters
            if MCRALS.__signature__.parameters[name].kind.name != "VAR_POSITIONAL"
        ]
        _keep_names = frozenset(_sig_params)

        # Find the next section after Parameters
        _lines = _params_body.split("\n")
        _next_section_start = len(_params_body)
        for _i, _line in enumerate(_lines):
            if _line.strip() and _i + 1 < len(_lines):
                _next = _lines[_i + 1]
                if _next.strip() and set(_next.strip()) == {"-"}:
                    _next_section_start = sum(len(_ln) + 1 for _ln in _lines[:_i])
                    break

        _params_only = _params_body[:_next_section_start]
        _after_params = _params_body[_next_section_start:]

        # Parse parameter entries in the Parameters section
        _entries = []
        _current_name = None
        _current_lines = []

        def _save_entry():
            if _current_name is not None:
                _entries.append((_current_name, "\n".join(_current_lines)))

        for _line in _params_only.split("\n"):
            _m = _re.match(r"^(\w+)\s*:", _line)
            if _m:
                _save_entry()
                _current_name = _m.group(1)
                _current_lines = [_line]
            elif _current_name is not None:
                _current_lines.append(_line)
        _save_entry()

        _entry_map = dict(_entries)

        # Rebuild in signature order, keeping only signature params.
        _filtered = [_entry_map[name] for name in _sig_params if name in _entry_map]

        # Rebuild docstring (blank line before subsequent sections is essential for numpydoc)
        MCRALS.__doc__ = (
            _before_params + _PH + "\n".join(_filtered) + "\n\n" + _after_params
        )


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

# Sentinel returned by ``_unpack_generator_output`` when the external
# generator (``getConc`` / ``getSpec``) returned a bare profile and did not
# update the args for the next iteration. Using a private sentinel (rather
# than ``None``) keeps ``None`` usable as a legitimate "no args" value.
_UNCHANGED = object()


def _profile_asarray(profile):
    # Return an ndarray view of a profile returned by an
    # external generator (``getConc``/``getSpec`` or ModelProfile model).
    #
    # NDDataset (and subclasses) expose the raw ndarray through `.data`;
    # for plain ndarrays we fall back to `np.asarray`. This avoids producing
    # object arrays and preserves the historical `.data` extraction semantics
    # without relying on `__array__`.
    data = getattr(profile, "data", None)
    if data is not None:
        return np.asarray(data)
    return np.asarray(profile)


# LS solvers for W in the linear matrix equation X @ W = Y
def _lstsq(X, Y, rcond=None):
    # Least-squares solution to a linear matrix equation X @ W = Y
    # Return W
    return np.linalg.lstsq(X, Y, rcond)[0]


def _nnls(X, Y, withres=False):
    # Non negative least-squares solution to a linear matrix equation X @ W = Y
    # Return W >= 0
    # TODO: look at may be faster algorithm: see: https://gist.github.com/vene/7224672
    nsamp, nfeat = X.shape
    nsamp, ntarg = Y.shape
    W = np.empty((nfeat, ntarg))
    residuals = 0
    for i in range(ntarg):
        Y_ = Y[:, i]
        W[:, i], res = scipy.optimize.nnls(X, Y_)
        residuals += res**2
    return (W, np.sqrt(residuals)) if withres else W


def _pnnls(X, Y, nonneg=None, withres=False):
    # Least-squares solution to X @ W = Y with coefficient-wise partial
    # non-negativity. ``nonneg`` indexes rows of W (component coefficients),
    # not the independent target columns of Y / W.
    if nonneg is None:
        nonneg = []
    _, nfeat = X.shape
    _, ntarg = Y.shape
    nonneg = np.asarray(nonneg)
    if nonneg.size and not np.issubdtype(nonneg.dtype, np.integer):
        raise ValueError(
            "nonneg must contain integer coefficient indices, "
            f"got {nonneg.tolist()!r}."
        )
    nonneg = nonneg.astype(int, copy=False)
    if nonneg.ndim != 1 or np.any(nonneg < 0) or np.any(nonneg >= nfeat):
        raise ValueError(
            "nonneg must contain valid coefficient indices in "
            f"[0, {nfeat}), got {nonneg.tolist()!r}."
        )

    lower = np.full(nfeat, -np.inf)
    lower[nonneg] = 0.0
    upper = np.full(nfeat, np.inf)
    all_nonnegative = len(np.unique(nonneg)) == nfeat

    W = np.empty((nfeat, ntarg))
    residuals_sq = 0.0
    for i in range(ntarg):
        Y_ = Y[:, i]
        if all_nonnegative:
            W[:, i], res = scipy.optimize.nnls(X, Y_)
        elif len(nonneg) == 0:
            W[:, i] = np.linalg.lstsq(X, Y_, rcond=None)[0]
            res = np.linalg.norm(X @ W[:, i] - Y_)
        else:
            result = scipy.optimize.lsq_linear(X, Y_, bounds=(lower, upper))
            W[:, i] = result.x
            res = np.linalg.norm(X @ result.x - Y_)
        residuals_sq += res**2
    return (W, np.sqrt(residuals_sq)) if withres else W


def _unimodal_2D(a, axis, idxes, tol, mod):
    # Force unimodality on given lines or columns of a 2D ndarray
    #
    # a: ndarray
    #
    # axis: int
    #     axis along which the correction is applied
    #
    # idxes: list of int
    #     indexes at which the correction is applied
    #
    # mod : str
    #     When set to `"strict"`, values deviating from unimodality are reset to the
    #     value of the previous point. When set to `"smooth"`, both values (deviating
    #     point and previous point) are modified to avoid "steps" in the profile.
    #
    # tol: float
    #     Tolerance parameter for unimodality. Correction is applied only if:
    #     `a[i] > a[i-1] * tol`  on a decreasing branch of profile,
    #     `a[i] < a[i-1] * tol`  on an increasing branch of profile.

    if axis == 0:
        a_ = a
    elif axis == 1:
        a_ = a.T

    for col, idx in zip(a_[:, idxes].T, idxes, strict=False):
        a_[:, idx] = _unimodal_1D(col, tol, mod)

    return a


def _unimodal_1D(a: np.ndarray, tol: float, mod: str) -> np.ndarray:
    # force unimodal concentration
    #
    # makes a vector unimodal
    #
    # Parameters
    # ----------
    # a : 1D ndarray
    #
    # mod : str
    #     When set to `"strict"`, values deviating from unimodality are reset to the value
    #     of the previous point. When set to `"smooth"`, both values (deviating point and
    #     previous point) are modified to avoid "steps"
    #     in the profile.
    #
    # tol: float
    #     Tolerance parameter for unimodality. Correction is applied only if:
    #     `a[i] > a[i-1] * unimodTol`  on a decreasing branch of profile,
    #     `a[i] < a[i-1] * unimodTol`  on an increasing branch of profile.

    # Safety bound on the number of iterations per sweep. The documented
    # regime is `tol >= 1` (default 1.1), where the algorithm terminates in
    # O(len(a)) steps; this cap is never reached there. It only guards against
    # the out-of-bounds reads and infinite loops that occur for pathological
    # `tol < 1` settings (B6), without changing the result for `tol >= 1`.
    max_iter = 100 * len(a) + 10

    maxid = np.argmax(a)
    curmax = max(a)
    curid = maxid

    n = 0
    while curid > 0:
        # run backward
        n += 1
        curid -= 1
        if a[curid] > curmax * tol:
            if mod == "strict":
                a[curid] = a[curid + 1]
            if mod == "smooth":
                a[curid] = (a[curid] + a[curid + 1]) / 2
                a[curid + 1] = a[curid]
                # advance past the merged point (historical semantics).
                # For `tol >= 1` this stays within bounds and is identical to
                # the original. For `tol < 1`, stop at the boundary to avoid
                # out-of-bounds reads (B6).
                curid = curid + 2
                if curid > len(a) - 1:
                    curid = len(a) - 1
                    curmax = a[curid]
                    break
        curmax = a[curid]
        if n >= max_iter:
            break

    curid = maxid
    curmax = a[maxid]
    n = 0
    while curid < len(a) - 1:
        n += 1
        curid += 1
        if a[curid] > curmax * tol:
            if mod == "strict":
                a[curid] = a[curid - 1]
            if mod == "smooth":
                a[curid] = (a[curid] + a[curid - 1]) / 2
                a[curid - 1] = a[curid]
                # go back past the merged point (historical semantics).
                # Symmetric to the backward sweep (B6).
                curid = curid - 2
                if curid < 0:
                    curid = 0
                    curmax = a[curid]
                    break
        curmax = a[curid]
        if n >= max_iter:
            break
    return a


def _project_rank_one_profiles(profiles):
    """
    Project a matrix of profiles onto the best rank-1 approximation via SVD.

    Parameters
    ----------
    profiles : np.ndarray of shape (n_points, n_blocks)
        Matrix where each column is a profile from one block.

    Returns
    -------
    reconstruction : np.ndarray of shape (n_points, n_blocks)
        Rank-1 reconstruction.
    amplitudes : np.ndarray of shape (n_blocks,)
        Per-block amplitudes from the SVD. The sign is resolved so that
        the shape vector has a positive sum.

    Notes
    -----
    The rank-1 approximation is ``u[:, 0] @ (s[0] * vh[0, :])``.
    The sign ambiguity is resolved by checking the sign of the sum of
    the left singular vector: if it is negative, both ``u`` and ``vh``
    are negated so that the shape (``u``) is mostly positive.
    """
    u, s, vh = np.linalg.svd(profiles, full_matrices=False)
    shape = u[:, 0].copy()
    ampl = vh[0, :].copy() * s[0]

    # Resolve SVD sign ambiguity: enforce sign by the element with
    # largest absolute value.  This is more deterministic than the
    # sum-based rule (which fails near zero-sum symmetric profiles).
    pivot = np.argmax(np.abs(shape))
    if shape[pivot] < 0:
        shape *= -1
        ampl *= -1

    reconstruction = np.outer(shape, ampl)
    return reconstruction, ampl
