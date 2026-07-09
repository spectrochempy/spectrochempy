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
from sklearn import decomposition

from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._analysisbase import _wrap_ndarray_output_to_nddataset
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis.decomposition.mcrals_constraints import Constraint
from spectrochempy.application.application import info_
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.decorators import signature_has_configurable_traits

# --------------------------------------------------------------------------------------
# Names of legacy traitlet-based constraint parameters.
# Used to detect mixed-API usage (legacy traitlets + new constraints= parameter).
# --------------------------------------------------------------------------------------
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
    Xpca: np.ndarray
    stdev: float
    change: float
    niter: int
    ndiv: int

    # per-iteration snapshots, refreshed at every ALS step
    C_constrained: np.ndarray | None = None
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
# The public traitlets (``nonnegConc``, ``unimodConc``, ...) remain the single
# user-facing configuration surface. They are translated into private
# constraint objects once, at the start of ``_fit``, by the builder methods
# ``_build_concentration_constraints`` and ``_build_spectral_constraints``.
# Inactive constraints are not created, so the runtime cost of an absent
# constraint is a missing list entry rather than a guarded no-op call.
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
    unimodality, monotonicity, closure, normalization, hard profile) and
    exposes a uniform ``apply(values, state)`` entry point. ``values`` is
    the current profile array (``C`` for concentrations, ``St`` for
    spectra); ``state`` is the transient ``_ALSState`` for the current
    fit. Implementations return the constrained profile so that callers
    can follow array reassignments (see ``_UnimodalConstraint``).
    """

    #: Short human-readable name used in debug logs / future introspection.
    name: str = "constraint"

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

    def __init__(self, indices, axis):
        self._indices = indices
        self._axis = axis

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

    def __init__(self, indices, axis, tol, mod):
        self._indices = indices
        self._axis = axis
        self._tol = tol
        self._mod = mod

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
    observations ``ny`` carried on ``state.X.shape[0]``.

    The activation guard uses an explicit truthiness test on the
    selection list rather than ``np.any(...)``, so that selecting only
    component 0 (``[0]``) is honoured. Same fix family as PR1 #911.
    """

    name = "monotonic-increase"

    def __init__(self, indices, tol):
        self._indices = indices
        self._tol = tol

    def apply(self, values, state):
        if not self._indices:
            return values
        ny = state.X.shape[0]
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

    def __init__(self, indices, tol):
        self._indices = indices
        self._tol = tol

    def apply(self, values, state):
        if not self._indices:
            return values
        ny = state.X.shape[0]
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

    def __init__(self, indices, method, target):
        self._indices = indices
        self._method = method
        self._target = target

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
    normalization constraint, which is applied after the second C solve
    — never as part of a generic per-profile pipeline.
    """

    name = "normalization"

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


class _HardProfileConstraint(_Constraint):
    """
    Inject externally generated profiles (``getConc`` / ``getSpec``).

    Wraps the historical ``_apply_conc_hard_constraints`` /
    ``_apply_spec_hard_constraints`` helpers and their shared dispatch /
    unpack helpers (``_call_external_generator``, ``_unpack_generator_output``).

    The class is parameterised by the binding side ("conc" or "spec")
    and stores a back-reference to the owning estimator so it can read
    the public traitlets (``getConc``/``getSpec``, ``argsGetConc``/
    ``argsGetSpec``, ...) and, where required, update ``argsGetConc`` /
    ``argsGetSpec`` between iterations. The external generator API is
    preserved exactly — this is purely a structural wrapper and does not
    redesign generated profiles (deferred to a later PR).
    """

    name = "hard-profile"

    def __init__(self, estimator, side):
        self._estimator = estimator
        self._side = side

    def apply(self, values, state):
        est = self._estimator
        if self._side == "conc":
            # Activation guard: use an explicit truthiness test on the
            # selection list rather than ``np.any(...)``, so that selecting
            # only component 0 (``hardConc=[0]``) is honoured —
            # ``np.any([0])`` evaluates to ``False`` and would silently
            # disable the hard-profile injection. Same fix family as PR1
            # #911 applied to ``_ClosureConstraint``.
            if not est.hardConc:
                state.extra_output_conc = []
                return values
            current = est._C_2_NDDataset(values)
            output = est._call_external_generator(
                est.getConc,
                current,
                est.argsGetConc,
                est.kwargsGetConc,
            )
            profiles, new_args, extra_output = est._unpack_generator_output(output)
            if new_args is not _UNCHANGED:
                est.argsGetConc = new_args
            values[:, est.hardConc] = _profile_asarray(profiles)[:, est.getC_to_C_idx]
            state.extra_output_conc = extra_output
            return values
        # _side == "spec"
        if not est.hardSpec:
            state.extra_output_spec = []
            return values
        current = est._St_2_NDDataset(values)
        output = est._call_external_generator(
            est.getSpec,
            current,
            est.argsGetSpec,
            est.kwargsGetSpec,
        )
        profiles, new_args, extra_output = est._unpack_generator_output(output)
        if new_args is not _UNCHANGED:
            est.argsGetSpec = new_args
        values[est.hardSpec, :] = _profile_asarray(profiles)[est.getSt_to_St_idx, :]
        state.extra_output_spec = extra_output
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
        ``Monotonic``, ``ModelProfile``).  This parameter cannot be combined
        with the legacy traitlet-based constraint parameters (``nonnegConc``,
        ``unimodConc``, ``closureConc``, etc.) — choose one API.

        Example::

            from spectrochempy.analysis import constraints as mc

            mcr = MCRALS(
                constraints=[
                    mc.NonNegative("C"),
                    mc.Closure("C"),
                ]
            )

        .. versionadded:: 0.7.0
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

    tol = tr.Float(
        0.1,
        help=(
            "Convergence criterion on the change of residuals (percent change of "
            "standard deviation of residuals)."
        ),
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
- ``'pnnls'``\ : non-negative least squares (`~scipy.optimize.nnls`) are applied on
  profiles indicated in `nonnegConc` and ordinary least squares on other profiles.
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
  each observation. Hence, ``np.ones(X.shape[0])`` would be equivalent to
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
  to ``range(len(hardConc))``
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
- ``'pnnls'``\ : non-negative least squares (`~scipy.optimize.nnls`) are applied on
  profiles indicated in `nonnegConc` and ordinary least squares on other profiles."""
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
  to ``range(len(hardSpec))``.
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
        if len(args) > 0:
            raise ValueError(
                "Passing arguments such as MCRALS(X, profile) "
                "is now deprecated. "
                "Instead, use MCRAL() followed by MCRALS.fit(X, profile). "
                "See the documentation and examples",
            )

        # Extract and validate the new constraints= parameter (not a traitlet).
        constraints = kwargs.pop("constraints", None)

        if constraints is not None:
            # Detect mixed API: legacy constraint traitlets + constraints=.
            mixed = _LEGACY_CONSTRAINT_TRAITS & set(kwargs)
            if mixed:
                raise ValueError(
                    "The legacy constraint parameters and the constraints "
                    "parameter cannot be used together. Please choose one API."
                )
            # Validate that every element is a Constraint instance.
            if not isinstance(constraints, (list, tuple)):
                raise TypeError(
                    f"constraints must be a list or tuple, got {type(constraints).__name__}"
                )
            for idx, c in enumerate(constraints):
                if not isinstance(c, Constraint):
                    raise TypeError(
                        f"constraints[{idx}] must be a Constraint instance, "
                        f"got {type(c).__name__}"
                    )
            self._constraints = list(constraints)
        else:
            self._constraints = None

        # call the super class for initialisation
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

        # deal with the callable that may have been serialized
        if self.getConc is not None and isinstance(self.getConc, str):
            self.getConc = dill.loads(base64.b64decode(self.getConc))  # noqa: S301
        if self.getSpec is not None and isinstance(self.getSpec, str):
            self.getSpec = dill.loads(base64.b64decode(self.getSpec))  # noqa: S301

        # storage for ALS diagnostics captured during _fit
        self._fit_meta = None

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------

    def _solve_C(self, St):
        if self.solverConc == "lstsq":
            return _lstsq(St.T, self._X.data.T).T
        if self.solverConc == "nnls":
            return _nnls(St.T, self._X.data.T).T
        if self.solverConc == "pnnls":
            return _pnnls(St.T, self._X.data.T, nonneg=self.nonnegConc).T
        return None

    def _solve_St(self, C):
        if self.solverSpec == "lstsq":
            return _lstsq(C, self._X.data)
        if self.solverSpec == "nnls":
            return _nnls(C, self._X.data)
        if self.solverSpec == "pnnls":
            return _pnnls(C, self._X.data, nonneg=self.nonnegSpec)
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

        # Build the constraint pipelines. When ``constraints=`` was provided
        # at construction time, translate the public Constraint objects into
        # the internal ``_Constraint`` pipeline directly. Otherwise, fall
        # back to the legacy traitlet-based builders. Both paths produce the
        # same internal constraint objects before entering the ALS loop.
        if self._constraints is not None:
            (
                conc_constraints,
                spec_constraints,
                normalization,
            ) = self._build_from_public_constraints(X)
        else:
            conc_constraints = self._build_concentration_constraints()
            spec_constraints = self._build_spectral_constraints()
            normalization = self._build_normalization()

        while (
            state.change >= self.tol
            and state.niter < self.max_iter
            and state.ndiv < self.maxdiv
        ):
            state.niter += 1

            # 1. Concentrations: apply soft + hard constraints, then snapshot
            #    the constrained C for storage / St resolution.
            state.C = self._apply_constraint_pipeline(
                state.C,
                conc_constraints,
                state,
            )
            state.C_constrained = state.C.copy()

            # 2. Spectra: solve St from constrained C, snapshot the least-squares
            #    St, then apply soft + hard spectral constraints.
            state.St = self._solve_St(state.C)
            state.St_ls = state.St.copy()
            state.St = self._apply_constraint_pipeline(
                state.St,
                spec_constraints,
                state,
            )

            # 3. Concentrations again: solve C from constrained St, then
            #    optionally normalize spectra / concentrations. Normalization
            #    is a single joint constraint (it mutates both St and C at
            #    once), so it is run outside the per-profile pipelines.
            state.C = self._solve_C(state.St)
            if normalization is not None:
                normalization.apply(state.St, state)

            # 4. History & convergence: record iteration profiles and update
            #    the convergence counters / log.
            self._store_iteration(state)
            self._update_convergence(state)

        self._finalize_fit(state)
        return self._build_fit_result(state)

    def _init_als_state(self, X, Y):
        """
        Build the transient ``_ALSState`` for a fit and emit the log header.

        Computes the PCA reference reconstruction used for the ``RSE / PCA``
        log column and pre-positions the convergence counters so the first
        iteration is executed.
        """
        C, St = Y
        n_components = self._n_components

        info_("***           ALS optimisation log            ***")
        info_("#iter     RSE / PCA        RSE / Exp      %change")
        info_("-------------------------------------------------")

        # get sklearn PCA with same number of components for further comparison
        pca = decomposition.PCA(n_components=n_components)
        Xtransf = pca.fit_transform(X)
        Xpca = pca.inverse_transform(Xtransf)

        return _ALSState(
            X=X,
            C=C,
            St=St,
            n_components=n_components,
            Xpca=Xpca,
            stdev=X.std(),
            change=self.tol + 1,
            niter=0,
            ndiv=0,
        )

    # -- constraint pipeline ---------------------------------------------------
    #
    # ``_fit`` no longer calls individual ``_apply_*`` helpers by name. It
    # builds an ordered list of ``_Constraint`` objects once per fit (in
    # ``_build_concentration_constraints`` / ``_build_spectral_constraints``)
    # and iterates over them. This keeps the historical constraint order
    # exactly where it was, while making the constraint surface uniform so
    # that future PRs can introduce new constraint objects without touching
    # the iteration scaffolding.
    #
    # The pipeline is split into three phases, matching the historical ALS
    # step structure:
    #
    #   1. concentration soft + hard constraints, applied to ``C`` *before*
    #      the St solve (``_conc_constraints``);
    #   2. spectral soft + hard constraints, applied to ``St`` *after* the
    #      St solve (``_spec_constraints``);
    #   3. normalization, applied to (``St``, ``C``) jointly *after* the
    #      second C solve (``_normalization``).
    #
    # Normalization is intentionally kept out of the per-profile pipelines
    # because it operates on both ``C`` and ``St`` at once.

    def _build_concentration_constraints(self):
        """
        Build the ordered list of concentration constraints.

        Translates the public ``nonnegConc`` / ``unimodConc`` /
        ``monoIncConc`` / ``monoDecConc`` / ``closureConc`` / ``hardConc``
        traitlets into private ``_Constraint`` objects. Inactive
        constraints are not appended, so the pipeline only pays for what
        is configured. The order matches the historical PR2 sequence
        (non-negativity → unimodality → monotonic increase → monotonic
        decrease → closure → hard profile) so the numerical result is
        byte-identical.
        """
        constraints = [
            _NonNegativeConstraint(self.nonnegConc, axis=0),
            _UnimodalConstraint(
                self.unimodConc,
                axis=0,
                tol=self.unimodConcTol,
                mod=self.unimodConcMod,
            ),
            _MonotonicIncreaseConstraint(self.monoIncConc, tol=self.monoIncTol),
            _MonotonicDecreaseConstraint(self.monoDecConc, tol=self.monoDecTol),
        ]
        # Closure is emitted only when ``closureConc`` is truthy. This
        # preserves the PR1 fix for issue #911 (a single selected
        # component such as ``[0]`` must activate closure, which
        # ``np.any`` alone did not honour).
        if self.closureConc:
            constraints.append(
                _ClosureConstraint(
                    self.closureConc,
                    method=self.closureMethod,
                    target=self.closureTarget,
                ),
            )
        # Hard profile injection is always present in the pipeline: when
        # ``hardConc`` is empty the constraint is a no-op that resets the
        # per-iteration extra-output buffer to ``[]``. Keeping it in the
        # pipeline unconditionally preserves the historical behaviour
        # where ``extraOutputGetConc`` was always an empty list for an
        # inactive ``hardConc`` (rather than the stale value from a
        # previous iteration).
        constraints.append(_HardProfileConstraint(self, side="conc"))
        return constraints

    def _build_spectral_constraints(self):
        """
        Build the ordered list of spectral constraints.

        Translates the public ``nonnegSpec`` / ``unimodSpec`` /
        ``hardSpec`` traitlets into private ``_Constraint`` objects.
        Inactive soft constraints are not appended. The hard-profile
        wrapper is always present (same rationale as in
        ``_build_concentration_constraints``: it resets the per-iteration
        ``extraOutputGetSpec`` buffer to ``[]`` when ``hardSpec`` is
        empty).
        """
        return [
            _NonNegativeConstraint(self.nonnegSpec, axis=1),
            _UnimodalConstraint(
                self.unimodSpec,
                axis=1,
                tol=self.unimodSpecTol,
                mod=self.unimodSpecMod,
            ),
            _HardProfileConstraint(self, side="spec"),
        ]

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

    def _build_from_public_constraints(self, X):
        """
        Build constraint pipelines from public ``Constraint`` objects.

        This is the single entry point used when the user passes
        ``constraints=`` to ``MCRALS``.  It converts each public
        ``Constraint`` to the corresponding private ``_Constraint``
        (or configures the estimator traitlets for ``ModelProfile``,
        which is picked up by ``_HardProfileConstraint``), splits by
        profile side, and returns the three pipeline components expected
        by ``_fit``.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed data array (the same ``X`` that was passed to
            ``_fit``). Used to expand scalar closure targets to the
            correct size.

        Returns
        -------
        conc_constraints : list[_Constraint]
            Concentration-side constraint pipeline.
        spec_constraints : list[_Constraint]
            Spectral-side constraint pipeline.
        normalization : _NormalizationConstraint or None
            Normalization constraint, or ``None`` if disabled.
        """
        # Reset model-profile traitlets to their defaults so that
        # _HardProfileConstraint is a no-op unless a ModelProfile is present.
        self.hardConc = []
        self.getConc = None
        self.hardSpec = []
        self.getSpec = None

        conc = []
        spec = []

        for c in self._constraints:
            internal = self._public_to_internal(c)
            if internal is not None:
                if c.profile == "C":
                    conc.append(internal)
                else:
                    spec.append(internal)

        # Always append the hard-profile constraint.  If no ModelProfile
        # was provided, the traitlets above remain at defaults and the
        # constraint is a no-op (it simply resets the extra-output buffer).
        conc.append(_HardProfileConstraint(self, side="conc"))
        spec.append(_HardProfileConstraint(self, side="spec"))

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

    def _public_to_internal(self, constraint):
        """
        Convert a single public ``Constraint`` to a private ``_Constraint``.

        Parameters
        ----------
        constraint : Constraint
            Public constraint object.

        Returns
        -------
        _Constraint or None
            Internal constraint, or ``None`` if the public type has no
            internal counterpart yet (e.g. ``ReferenceProfile``).
        """
        from spectrochempy.analysis.decomposition.mcrals_constraints import Closure
        from spectrochempy.analysis.decomposition.mcrals_constraints import ModelProfile
        from spectrochempy.analysis.decomposition.mcrals_constraints import Monotonic
        from spectrochempy.analysis.decomposition.mcrals_constraints import NonNegative
        from spectrochempy.analysis.decomposition.mcrals_constraints import Unimodal

        if isinstance(constraint, NonNegative):
            indices = self._resolve_components(constraint.components)
            axis = 0 if constraint.profile == "C" else 1
            return _NonNegativeConstraint(indices, axis)

        if isinstance(constraint, Unimodal):
            indices = self._resolve_components(constraint.components)
            axis = 0 if constraint.profile == "C" else 1
            return _UnimodalConstraint(
                indices,
                axis=axis,
                tol=1.1,
                mod=constraint.mod,
            )

        if isinstance(constraint, Monotonic):
            indices = self._resolve_components(constraint.components)
            tol = constraint.tolerance
            if constraint.direction == "increasing":
                return _MonotonicIncreaseConstraint(indices, tol=tol)
            return _MonotonicDecreaseConstraint(indices, tol=tol)

        if isinstance(constraint, Closure):
            indices = self._resolve_components(constraint.components)
            # Expand a scalar target to a 1-D array (matching the shape of
            # the constrained axis), mirroring the legacy trait validator.
            target = constraint.target
            if np.ndim(target) == 0:
                target = np.full(self._X.shape[0], target)
            return _ClosureConstraint(
                indices,
                method="scaling",
                target=target,
            )

        if isinstance(constraint, ModelProfile):
            # ModelProfile is handled by configuring the estimator traitlets
            # that _HardProfileConstraint reads at each iteration.
            comps = self._resolve_components(constraint.components)
            if constraint.profile == "C":
                self.hardConc = comps
                self.getConc = constraint.model
                self.argsGetConc = constraint.model_args
                self.kwargsGetConc = constraint.model_kwargs
            else:
                self.hardSpec = comps
                self.getSpec = constraint.model
                self.argsGetSpec = constraint.model_args
                self.kwargsGetSpec = constraint.model_kwargs
            # No _Constraint produced here; _HardProfileConstraint (always
            # appended in _build_from_public_constraints) handles it.
            return None

        # Remaining public types (ReferenceProfile, FixedValues, ZeroRegion,
        # Selectivity) have no internal counterpart yet.
        return None

    @staticmethod
    def _apply_constraint_pipeline(profile, constraints, state):
        """
        Run a list of constraints in order against ``profile``.

        Each constraint may mutate ``profile`` in place or rebind it (see
        ``_UnimodalConstraint``); ``_apply_constraint_pipeline`` propagates
        the returned reference so every subsequent constraint sees the
        up-to-date array.
        """
        for constraint in constraints:
            profile = constraint.apply(profile, state)
        return profile

    # -- external generator dispatch (used by ``_HardProfileConstraint``) ----
    #
    # These two small helpers were the central pieces introduced by PR2 for
    # the external ``getConc`` / ``getSpec`` dispatch. PR3 keeps them as
    # estimator methods because the constraints need to call them through
    # their back-reference to the estimator (``est._call_external_generator``
    # / ``est._unpack_generator_output``). They are intentionally left
    # unchanged so the external-generator contract is preserved verbatim.

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
            state.C_constrained_list.append(state.C_constrained)
            state.St_ls_list.append(state.St_ls.copy())
            state.C_ls_list.append(state.C)
            state.St_constrained_list.append(state.St)

    def _update_convergence(self, state):
        """
        Update the residual-based convergence counters and emit the log line.

        Computes the percent change of the residuals standard deviation, logs
        the iteration row (RSE vs PCA and vs the data), updates the
        non-improvement counter ``ndiv`` and folds the signed change into its
        absolute value for the loop test. The convergence diagnostics text
        ("converged !", "Stop ALS optimization.", ...) is emitted here so it
        keeps the historical ordering with respect to the log row.
        """
        Xhat = state.C @ state.St
        stdev2 = np.std(Xhat - state.X)
        change = 100 * (stdev2 - state.stdev) / state.stdev
        state.stdev = stdev2

        stdev_PCA = np.std(Xhat - state.Xpca)
        info_(
            f"{state.niter: 3d}{' ' * 6}{stdev_PCA: 10f}{' ' * 6}"
            f"{stdev2: 10f}{' ' * 6}{change: 10f}",
        )

        # check convergence
        if change > 0:
            state.ndiv += 1
        else:
            state.ndiv = 0
            change = -change
        state.change = change

        if change < self.tol:
            info_("converged !")

        if state.ndiv == self.maxdiv:
            info_(
                f"Optimization not improved after {self.maxdiv} iterations"
                f"... unconverged or 'tol' set too small ?",
            )
            info_("Stop ALS optimization.")

        if state.niter == self.max_iter:
            info_(
                f"Convergence criterion ('tol') not reached after "
                f"{self.max_iter: d} iterations.",
            )
            info_("Stop ALS optimization.")

    # -- finalization ----------------------------------------------------------

    def _finalize_fit(self, state):
        """Persist ALS diagnostics and set the fitted spectral components."""
        # capture ALS diagnostics for the result property
        self._fit_meta = {
            "n_iter": state.niter,
            "change": state.change,
            "residual_std": state.stdev,
            "converged": state.change < self.tol,
        }
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

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y):
        """
        Fit the MCRALS model on an X dataset using initial concentration or spectra.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : :term:`array-like` or list of :term:`array-like`
            Initial concentration or spectra.

        Returns
        -------
        self
            The fitted instance itself.

        See Also
        --------
        transform : Apply dimensionality reduction.
        fit_transform : Fit the model and apply dimensionality reduction.

        """
        return super().fit(X, Y)

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
            The number of components to use for the reconstruction.

        """
        return super().inverse_transform(X_transform, **kwargs)

    @property
    def C(self):
        """The final concentration profiles."""
        C = self.transform()
        C.name = "Pure concentration profile, mcs-als of " + self.X.name
        return C

    @property
    def St(self):
        """The final spectra profiles."""
        St = self.components
        St.name = "Pure spectra profile, mcs-als of " + self.X.name
        return St

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def C_constrained(self):
        """The last constrained concentration profiles, i.e. after applying the hard and soft constraints."""
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
        The list of constrained concentration profiles at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[6]

    @property
    def C_ls_list(self):
        """
        The list of concentration profiles obtained by least square optimization and scaling at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[7]

    @property
    def St_constrained_list(self):
        """
        The list of constrained spectral profiles at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[8]

    @property
    def St_ls_list(self):
        """
        The list of optimized spectral profiles at each ALS iteration.

        Requires `MCRALS.storeIterations` set to True.
        """
        return self._outfit[9]

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
            "tol": self.tol,
            "maxdiv": self.maxdiv,
            "solverConc": self.solverConc,
            "solverSpec": self.solverSpec,
            "nonnegConc": self.nonnegConc,
            "nonnegSpec": self.nonnegSpec,
            "unimodConc": self.unimodConc,
            "unimodSpec": self.unimodSpec,
            "unimodConcMod": self.unimodConcMod,
            "unimodSpecMod": self.unimodSpecMod,
            "unimodConcTol": self.unimodConcTol,
            "unimodSpecTol": self.unimodSpecTol,
            "closureConc": self.closureConc,
            "closureTarget": (
                f"<array shape={self.closureTarget.shape}>"
                if isinstance(self.closureTarget, np.ndarray)
                else self.closureTarget
            ),
            "closureMethod": self.closureMethod,
            "monoDecConc": self.monoDecConc,
            "monoDecTol": self.monoDecTol,
            "monoIncConc": self.monoIncConc,
            "monoIncTol": self.monoIncTol,
            "hardConc": self.hardConc,
            "hardSpec": self.hardSpec,
            "normSpec": self.normSpec,
            "storeIterations": self.storeIterations,
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
# Utilities
# --------------------------------------------------------------------------------------

# Sentinel returned by ``_unpack_generator_output`` when the external
# generator (``getConc`` / ``getSpec``) returned a bare profile and did not
# update the args for the next iteration. Using a private sentinel (rather
# than ``None``) keeps ``None`` usable as a legitimate "no args" value.
_UNCHANGED = object()


def _profile_asarray(profile):
    # Return an ndarray view of a hard-constraint profile returned by an
    # external generator (`getConc`/`getSpec`).
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
    # Least-squares  solution to a linear matrix equation X @ W = Y
    # with partial nonnegativity (indicated by the nonneg list of targets)
    # Return W with eventually some column non negative.
    if nonneg is None:
        nonneg = []
    nsamp, nfeat = X.shape
    nsamp, ntarg = Y.shape
    W = np.empty((nfeat, ntarg))
    residuals = 0
    for i in range(ntarg):
        Y_ = Y[:, i]
        if i in nonneg:
            W[:, i], res = scipy.optimize.nnls(X, Y_)
        else:
            W[:, i], res = np.linalg.lstsq(X, Y_)[:2]
        residuals += res**2
    return (W, np.sqrt(residuals)) if withres else W


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
