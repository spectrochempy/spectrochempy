# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["Optimize"]
__configurables__ = __all__

import inspect
import re
import sys
from copy import deepcopy

import numpy as np
import traitlets as tr
from IPython import display

from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import FitResult
from spectrochempy.analysis.curvefitting import _fit_backend
from spectrochempy.analysis.curvefitting import _models as models_
from spectrochempy.analysis.curvefitting._callable_spec import _callable_model_spec
from spectrochempy.analysis.curvefitting._callable_spec import _callable_models_spec
from spectrochempy.analysis.curvefitting._fit_backend import _extract_solver_artifacts
from spectrochempy.analysis.curvefitting._fit_backend import _freeze_jacobian_artifact
from spectrochempy.analysis.curvefitting._fit_backend import _normalize_solver_meta
from spectrochempy.analysis.curvefitting._fit_backend import _run_optimizer
from spectrochempy.analysis.curvefitting._model_evaluation import _evaluate_callable
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._modelspec import prepare_model
from spectrochempy.analysis.curvefitting._optimize_curves import CURVE_MODELS
from spectrochempy.analysis.curvefitting._optimize_curves import fit_curve
from spectrochempy.analysis.curvefitting._optimize_model import fit_callable_model
from spectrochempy.analysis.curvefitting._optimize_nd import fit_separable_peaks
from spectrochempy.analysis.curvefitting._optimize_nd import snapshot_peaks
from spectrochempy.analysis.curvefitting._parameter_space import _internal_vector
from spectrochempy.analysis.curvefitting._parameter_space import (
    _modelspec_parameter_count,
)
from spectrochempy.analysis.curvefitting._parameter_space import _prepared_parameters
from spectrochempy.analysis.curvefitting._parameter_space import _restore_external
from spectrochempy.analysis.curvefitting._parameter_space import _restore_physical
from spectrochempy.analysis.curvefitting._parameter_space import _transform_artifacts
from spectrochempy.analysis.curvefitting._parameter_transform import _LOB_THRESHOLD
from spectrochempy.analysis.curvefitting._parameter_transform import _UPB_THRESHOLD
from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.analysis.curvefitting._stationarity import _physical_stationarity
from spectrochempy.analysis.curvefitting.model import Model
from spectrochempy.analysis.curvefitting.model import _prepare_public_model
from spectrochempy.application.application import info_
from spectrochempy.application.application import warning_
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import signature_has_configurable_traits

optimize = _fit_backend.optimize


# ======================================================================================
# Public data types
# ======================================================================================
class ScriptError:
    """
    Structured error returned by :meth:`Optimize.validate_script`.

    Attributes
    ----------
    line : int
        1-indexed line number where the error was found.
    text : str
        The offending line content.
    message : str
        Human-readable explanation of the error.
    """

    __slots__ = ("line", "text", "message")

    def __init__(self, line, text, message):
        self.line = line
        self.text = text
        self.message = message

    def __repr__(self):
        return f"ScriptError(line={self.line}, text={self.text!r}, message={self.message!r})"

    def __str__(self):
        return f"Line {self.line}: {self.message}\n  {self.text}"


class ConstraintError:
    """
    Structured error returned by :meth:`Optimize.validate_constraints`.

    Attributes
    ----------
    index : int
        0-indexed index of the offending constraint specification.
    constraint : any
        The offending constraint object as provided by the user.
    message : str
        Human-readable explanation of the error.
    """

    __slots__ = ("index", "constraint", "message")

    def __init__(self, index, constraint, message):
        self.index = index
        self.constraint = constraint
        self.message = message

    def __repr__(self):
        return (
            "ConstraintError("
            f"index={self.index}, constraint={self.constraint!r}, "
            f"message={self.message!r})"
        )

    def __str__(self):
        return f"Constraint {self.index}: {self.message}\n  {self.constraint!r}"


# ======================================================================================
# Module-level helpers
# ======================================================================================
def _scalar_from_masked(value, *, default=np.nan):
    """Return a Python float from a masked scalar-like value."""
    if np.ma.is_masked(value):
        return float(default)
    return float(value)


def _freeze_covariance_matrix(covariance):
    """Return an immutable ndarray snapshot of a computed covariance matrix."""
    if covariance is None:
        return None
    array = np.array(covariance, copy=True, dtype=np.float64)
    array.flags.writeable = False
    return array


def _count_varying_parameters(fp):
    """Count free parameters using the same rule as the optimizer internals."""
    if fp is None:
        return 0

    if isinstance(fp, _FitModelSpec):
        return int(fp.count_varying())

    n_varying = 0
    for key in sorted(fp.keys()):
        if fp.fixed[key]:
            continue
        key_prefix = key.split("_")[0]
        if key_prefix in fp.expvars:
            n_varying += fp.expnumber
        else:
            n_varying += 1
    return int(n_varying)


def _extract_varying_parameter_values(fp):
    """Return fitted varying-parameter values in optimizer order."""
    if fp is None:
        return None

    if isinstance(fp, _FitModelSpec):
        return fp.extract_varying_values()

    values = []
    for key in sorted(fp.keys()):
        if fp.fixed[key]:
            continue
        key_prefix = key.split("_")[0]
        value = fp[key]
        if key_prefix in fp.expvars:
            values.extend(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
        else:
            values.append(float(value))

    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


def _compute_fit_diagnostics(observed, fitted, solver_meta=None, fit_parameters=None):
    """Compute residual output and basic fit-quality diagnostics."""
    if getattr(observed, "size", None) == 0 or getattr(fitted, "size", None) == 0:
        residuals = observed.copy()
    else:
        residuals = observed - fitted

    residual_data = np.ma.masked_invalid(np.ma.asarray(residuals.real.masked_data))
    observed_data = np.ma.masked_invalid(np.ma.asarray(observed.real.masked_data))

    count = int(residual_data.count())
    n_varying_parameters = _count_varying_parameters(fit_parameters)
    degrees_of_freedom = count - n_varying_parameters

    sse = _scalar_from_masked(np.ma.sum(np.ma.abs(residual_data) ** 2), default=0.0)
    rss = sse

    if count == 0:
        rmse = float("nan")
        r_squared = float("nan")
        adjusted_r_squared = float("nan")
    else:
        rmse = float(np.sqrt(sse / count))
        observed_mean = np.ma.mean(observed_data)
        tss = _scalar_from_masked(
            np.ma.sum((observed_data - observed_mean) ** 2),
            default=np.nan,
        )
        if not np.isfinite(tss) or np.isclose(tss, 0.0):
            r_squared = float("nan")
        else:
            r_squared = float(1.0 - (sse / tss))

        if degrees_of_freedom > 0 and count > 1 and np.isfinite(r_squared):
            adjusted_r_squared = float(
                1.0 - ((1.0 - r_squared) * (count - 1) / degrees_of_freedom),
            )
        else:
            adjusted_r_squared = float("nan")

    if degrees_of_freedom > 0:
        reduced_chi_square = float(rss / degrees_of_freedom)
    else:
        reduced_chi_square = float("nan")

    if count > 0:
        mean_squared_residual = rss / count
        if mean_squared_residual < 0.0 or not np.isfinite(mean_squared_residual):
            aic = float("nan")
            bic = float("nan")
        elif mean_squared_residual == 0.0:
            aic = float("-inf")
            bic = float("-inf")
        else:
            log_likelihood_term = float(count * np.log(mean_squared_residual))
            aic = float(log_likelihood_term + (2.0 * n_varying_parameters))
            bic = float(log_likelihood_term + (n_varying_parameters * np.log(count)))
    else:
        aic = float("nan")
        bic = float("nan")

    diagnostics = {
        "n_observations": count,
        "n_varying_parameters": n_varying_parameters,
        "degrees_of_freedom": int(degrees_of_freedom),
        "sse": sse,
        "rss": rss,
        "rmse": rmse,
        "r_squared": r_squared,
        "reduced_chi_square": reduced_chi_square,
        "adjusted_r_squared": adjusted_r_squared,
        "aic": aic,
        "bic": bic,
    }
    diagnostics.update(dict(solver_meta or {}))
    return residuals, diagnostics


def _sync_fitparameters_from_modelspec(fp, spec):
    """
    Reset a ``FitParameters`` view in place to faithfully reflect a spec.

    Preserves the object identity of *fp* while rebuilding its full view from
    the canonical model spec: values, ``fixed``/``vary`` flags, bounds,
    reference flags, model structure and experiment metadata are all restored.
    Direct legacy-view mutations (``fp.fixed``, ``fp.lob``/``fp.upb``,
    ``fp.reference``, ...) therefore cannot survive into the public
    ``Optimize.fp`` after a fit when they are not authoritative.
    """
    view = spec.to_fitparameters()
    fp.data = view.data
    fp.lob = view.lob
    fp.upb = view.upb
    fp.fixed = view.fixed
    fp.reference = view.reference
    fp.common = view.common
    fp.model = view.model
    fp.models = view.models
    fp.expvars = view.expvars
    fp.expnumber = view.expnumber
    return fp


def _modelspec_parameter_names(spec):
    """Return flat historical parameter names for a canonical model spec."""
    names = set(spec.common_params)
    for comp in spec.components:
        for name in comp.params:
            names.add(f"{name}_{comp.label}")
    return names


def _compute_covariance_matrix(
    observed,
    fitted,
    jacobian,
    diagnostics,
    *,
    transform_derivatives=None,
    near_bounds=False,
):
    """
    Compute an approximate parameter covariance matrix from the retained Jacobian.

    The approximation follows the local linear least-squares convention:

        covariance = D * (sigma^2 * inv(J.T @ J)) * D.T

    where ``sigma^2`` is the residual variance estimate based on the reduced
    chi-square (equivalently ``rss / degrees_of_freedom``). J is the raw
    optimizer-coordinate Jacobian and D is the diagonal physical-coordinate
    derivative. A column-normalized SVD avoids squaring the condition number.
    Rank deficiency, active bounds and unsuccessful fits exclude this local
    unconstrained approximation. The diagnostics record the exclusion reason.

    Returns
    -------
    ndarray or None
        Immutable covariance matrix when available, otherwise ``None``.
    """
    diagnostics["covariance_coordinates"] = "physical"
    diagnostics["covariance_reason"] = "missing_jacobian"
    if jacobian is None:
        return None

    if diagnostics.get("success") is False:
        diagnostics["covariance_reason"] = "not_converged"
        return None

    if near_bounds:
        diagnostics["covariance_reason"] = "active_bounds"
        return None

    if observed is None or fitted is None:
        return None

    degrees_of_freedom = int(diagnostics.get("degrees_of_freedom", 0))
    sigma2 = diagnostics.get("reduced_chi_square", np.nan)
    n_varying_parameters = int(diagnostics.get("n_varying_parameters", 0))

    if degrees_of_freedom <= 0 or n_varying_parameters <= 0:
        diagnostics["covariance_reason"] = "insufficient_degrees_of_freedom"
        return None

    sigma2 = float(sigma2)
    if not np.isfinite(sigma2):
        diagnostics["covariance_reason"] = "numeric_range"
        return None

    jacobian_array = np.asarray(jacobian, dtype=np.float64)
    diagnostics["covariance_reason"] = "invalid_jacobian"
    if jacobian_array.ndim != 2:
        return None

    residual_data = np.ma.masked_invalid(
        np.ma.asarray((observed - fitted).real.masked_data)
    )
    valid_mask = ~np.ma.getmaskarray(residual_data)
    valid_mask &= np.isfinite(np.ma.getdata(residual_data))
    valid_rows = valid_mask.reshape(-1)

    if jacobian_array.shape[1] != n_varying_parameters:
        return None

    if jacobian_array.shape[0] == valid_rows.size:
        jacobian_used = jacobian_array[valid_rows]
    elif jacobian_array.shape[0] == int(diagnostics.get("n_observations", 0)):
        jacobian_used = jacobian_array
    else:
        return None

    if jacobian_used.shape[0] <= 0:
        return None

    derivatives = (
        np.ones(n_varying_parameters)
        if transform_derivatives is None
        else np.asarray(transform_derivatives, dtype=float)
    )
    if derivatives.shape != (n_varying_parameters,):
        return None
    if not np.isfinite(jacobian_used).all() or not np.isfinite(derivatives).all():
        diagnostics["covariance_reason"] = "numeric_range"
        return None
    if np.any(derivatives == 0):
        diagnostics["covariance_reason"] = "singular_transform"
        return None

    column_scale = np.max(np.abs(jacobian_used), axis=0)
    diagnostics["covariance_reason"] = "rank_deficient"
    if np.any(column_scale == 0):
        return None
    try:
        _, singular, right = np.linalg.svd(
            jacobian_used / column_scale, full_matrices=False
        )
    except np.linalg.LinAlgError:
        diagnostics["covariance_reason"] = "invalid_jacobian"
        return None
    if singular.size < n_varying_parameters or singular[-1] <= (
        singular[0] * np.sqrt(np.finfo(float).eps)
    ):
        return None
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        physical_scale = np.sqrt(np.longdouble(sigma2)) * derivatives / column_scale
        scaled = np.asarray((right / singular[:, None]) * physical_scale, dtype=float)
        covariance = scaled.T @ scaled
    if not np.isfinite(covariance).all() or (
        sigma2 > 0 and np.any(np.diag(covariance) == 0)
    ):
        diagnostics["covariance_reason"] = "numeric_range"
        return None
    diagnostics["covariance_reason"] = None
    return _freeze_covariance_matrix(covariance)


def _parse_script_to_modelspec(script, usermodels=None):
    """
    Parse and validate a curve-fitting script into the canonical model spec.

    Returns the canonical fitting model plus the historical list of parser
    errors. This is the architectural parser entry point; compatibility helpers
    may derive ``FitParameters`` from the returned model spec when needed.
    """
    spec = _FitModelSpec()
    errors = []

    current_component = None
    in_common_block = False
    models_missing_shape = set()

    lines = script.split("\n")
    lc = 0

    for item in lines:
        lc += 1
        line = item.strip()
        if line == "" or line.startswith("#"):
            continue

        s = line.split(":")
        if len(s) != 2:
            errors.append(
                ScriptError(
                    lc,
                    line,
                    "Cannot interpret line: A semi-column is missing?",
                ),
            )
            continue

        key, values = s
        key = key.strip().lower()

        if key.startswith("model"):
            modlabel = values.lower().strip()
            if any(comp.label == modlabel for comp in spec.components):
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        f"Duplicate model label: '{modlabel}'",
                    ),
                )
            else:
                current_component = _ComponentSpec(label=modlabel)
                spec.components.append(current_component)
                if modlabel:
                    models_missing_shape.add(modlabel)
            in_common_block = False
            continue

        if key.startswith("common") or key.startswith("vars"):
            current_component = None
            in_common_block = True
            continue

        if key.startswith("shape"):
            if current_component is None or in_common_block:
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        "shape: must appear after a MODEL: declaration",
                    ),
                )
                continue
            models_missing_shape.discard(current_component.label)
            shape = values.lower().strip()
            if not shape:
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        "Shape of this model was not specified or is not implemented",
                    ),
                )
                continue
            model_ok = hasattr(models_, shape) or (
                usermodels is not None and shape in usermodels
            )
            if not model_ok:
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        f"Model {shape} not found in spectrochempy nor in usermodels.",
                    ),
                )
                continue
            current_component.model_name = shape
            in_common_block = False
            continue

        if current_component is None and not in_common_block:
            errors.append(
                ScriptError(
                    lc,
                    line,
                    "The first definition should be a label for a model"
                    " or a block of variables or constants.",
                ),
            )
            continue

        if key.startswith("*"):
            vary = False
            reference = False
            key = key[1:].strip()
        elif key.startswith("$"):
            vary = True
            reference = False
            key = key[1:].strip()
        elif key.startswith(">"):
            vary = False
            reference = True
            key = key[1:].strip()
        else:
            errors.append(
                ScriptError(
                    lc,
                    line,
                    "Cannot interpret line: A parameter definition must start"
                    " with *,$ or >",
                ),
            )
            continue

        s = values.split(",")
        s = [ss.strip() for ss in s]
        if len(s) > 1 and ("[" in s[0]) and ("]" in s[1]):
            s[0] = f"{s[0]}, {s[1]}"
            if len(s) > 2:
                s[1:] = s[2:]
        if len(s) > 3:
            errors.append(
                ScriptError(
                    lc,
                    line,
                    "value, min, max should be defined in this order",
                ),
            )
            continue
        if len(s) == 2:
            errors.append(
                ScriptError(
                    lc,
                    line,
                    "only two items; value, min, max (or value only) should be defined",
                ),
            )
            continue
        if len(s) == 1:
            s.extend(["none", "none"])
        value, mini, maxi = s

        if reference:
            param_spec = _ParamSpec(
                name=key, value=0.0, vary=False, reference=value.strip()
            )
        else:
            val = value.strip()
            try:
                val = eval(str(val))  # noqa: S307
            except Exception as exc:
                flat_key = (
                    key if in_common_block else f"{key}_{current_component.label}"
                )
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        f"Cannot evaluate value for '{flat_key}': {exc}",
                    ),
                )
                continue

            min_bound = (
                None if mini.strip().lower() in ["none", ""] else float(mini.strip())
            )
            max_bound = (
                None if maxi.strip().lower() in ["none", ""] else float(maxi.strip())
            )
            param_spec = _ParamSpec(
                name=key,
                value=val,
                vary=vary,
                bounds=(min_bound, max_bound),
            )

        if in_common_block:
            spec.common_params[key] = param_spec
        else:
            current_component.params[key] = param_spec

    for label in list(models_missing_shape):
        errors.append(
            ScriptError(
                0,
                "",
                f"Model '{label}' has no shape definition",
            ),
        )

    return spec, errors


def _validate_script_content(script, usermodels=None):
    """
    Parse and validate a curve-fitting script without running an optimisation.

    Parameters
    ----------
    script : str
        The script to validate.
    usermodels : dict or None
        Optional user-defined model registry (same format as
        :attr:`Optimize.usermodels`).

    Returns
    -------
    fp : FitParameters
        Parsed parameters (always returned, even when *errors* is non-empty).
    errors : list of ScriptError
        Empty list when the script is valid.
    """
    spec, errors = _parse_script_to_modelspec(script, usermodels=usermodels)
    return spec.to_fitparameters(), errors


def _normalize_constraint_spec(constraint):
    """Normalize accepted constraint spellings to a common dict form."""
    if isinstance(constraint, dict) and "max_connections" in constraint:
        extra_keys = set(constraint) - {"max_connections", "parameters"}
        if extra_keys:
            return (
                None,
                f"Unsupported keys for max_connections constraint: {sorted(extra_keys)!r}",
            )
        return {
            "type": "max_connections",
            "limit": constraint["max_connections"],
            "parameters": constraint.get("parameters"),
        }, None

    if isinstance(constraint, dict):
        return dict(constraint), None

    return None, "Each constraint must be a dict specification."


def _validate_constraints_content(constraints, fit_parameters=None):
    r"""
    Validate lightweight constraint specifications without running a fit.

    Supported minimal schemas currently include:

    - ``{\"max_connections\": <positive int>}``
    - ``{\"type\": \"max_connections\", \"limit\": <positive int>, \"parameters\": [...]}``

    Validation is limited to structure and parameter-name references. It does
    not guarantee that the current optimization backend will enforce every
    accepted constraint semantically.
    """
    if constraints in (None, {}, []):
        return []

    if isinstance(constraints, dict):
        constraints_list = [constraints]
    elif isinstance(constraints, (list, tuple)):
        constraints_list = list(constraints)
    else:
        return [
            ConstraintError(
                0,
                constraints,
                "Constraints must be a dict or a list/tuple of dicts.",
            ),
        ]

    if isinstance(fit_parameters, _FitModelSpec):
        known_parameters = _modelspec_parameter_names(fit_parameters)
    else:
        known_parameters = (
            set(fit_parameters.keys()) if fit_parameters is not None else set()
        )
    errors = []

    for index, raw_constraint in enumerate(constraints_list):
        constraint, normalization_error = _normalize_constraint_spec(raw_constraint)
        if normalization_error is not None:
            errors.append(
                ConstraintError(index, raw_constraint, normalization_error),
            )
            continue

        constraint_type = str(constraint.get("type", "")).strip().lower()
        if constraint_type != "max_connections":
            errors.append(
                ConstraintError(
                    index,
                    raw_constraint,
                    "Unsupported constraint type. Currently only 'max_connections' "
                    "is recognized by validate_constraints().",
                ),
            )
            continue

        limit = constraint.get("limit")
        try:
            limit_int = int(limit)
        except Exception:
            errors.append(
                ConstraintError(
                    index,
                    raw_constraint,
                    "Constraint 'limit' must be an integer.",
                ),
            )
            continue

        if limit_int <= 0:
            errors.append(
                ConstraintError(
                    index,
                    raw_constraint,
                    "Constraint 'limit' must be a positive integer.",
                ),
            )

        parameters = constraint.get("parameters")
        if parameters is None:
            continue

        if not isinstance(parameters, (list, tuple)) or not parameters:
            errors.append(
                ConstraintError(
                    index,
                    raw_constraint,
                    "Constraint 'parameters' must be a non-empty list or tuple of parameter names.",
                ),
            )
            continue

        invalid_names = [
            name for name in parameters if not isinstance(name, str) or not name
        ]
        if invalid_names:
            errors.append(
                ConstraintError(
                    index,
                    raw_constraint,
                    "Constraint 'parameters' entries must be non-empty strings.",
                ),
            )
            continue

        if known_parameters:
            missing = [name for name in parameters if name not in known_parameters]
            if missing:
                errors.append(
                    ConstraintError(
                        index,
                        raw_constraint,
                        f"Unknown parameter name(s) referenced by constraint: {missing!r}",
                    ),
                )

    return errors


def _canonicalize_constraints(constraints):
    """Return a normalized constraints payload for estimator storage."""
    if constraints in (None, {}, []):
        return None

    if isinstance(constraints, dict):
        constraint, _ = _normalize_constraint_spec(constraints)
        return constraint

    canonical_constraints = []
    for constraint in constraints:
        normalized, _ = _normalize_constraint_spec(constraint)
        canonical_constraints.append(normalized)
    return canonical_constraints


# ======================================================================================
@signature_has_configurable_traits
class Optimize(DecompositionAnalysis):
    """
    Non-linear curve fitting driven by the SpectroChemPy fitting DSL.

    `Optimize` combines:

    - script-based model definition through :attr:`script`;
    - pre-fit validation through :meth:`validate_script`;
    - multiple public optimization-method families through :attr:`method`;
    - estimator-level access to raw solver artifacts such as :attr:`jacobian`;
    - grouped scientific outputs and diagnostics through :attr:`result`.

    The current public `method` values are:

    - ``"least_squares"``
    - ``"leastsq"``
    - ``"simplex"``
    - ``"basinhopping"``

    In current SpectroChemPy releases, ``"least_squares"`` is the clearest
    recommended entrypoint for ordinary local least-squares fitting.
    ``"leastsq"`` is still accepted as a public compatibility alias, but it
    currently uses the same least-squares backend family rather than defining a
    separate maintained fitting strategy of its own.

    Least-squares-backed methods are the only ones that currently expose the
    retained Jacobian and the resulting uncertainty path on
    :class:`~spectrochempy.analysis._base._result.FitResult`
    (covariance, standard errors, correlation, and confidence intervals).

    Method selection
    ----------------
    In current SpectroChemPy releases, the practical guidance is:

    - use ``"least_squares"`` as the default choice for ordinary local
      least-squares fitting;
    - treat ``"leastsq"`` mainly as a compatibility alias rather than as a
      separate maintained strategy;
    - use ``"simplex"`` when you explicitly want a derivative-free local
      search and can accept losing the least-squares uncertainty path;
    - reserve ``"basinhopping"`` for harder landscapes where a slower
      exploratory global-style search is justified.

    The low-level SciPy least-squares backend variant (currently ``lm`` or
    ``trf``) is still selected internally; users choose the high-level
    SpectroChemPy `method`, not the backend variant directly.

    Notes
    -----
    The script interface fits 1D spectra. Pass ``peaks`` to fit separable
    Gaussian or Lorentzian profiles jointly on a multidimensional grid.
    Pass ``model`` and ``parameters`` for a nonseparable full-grid ND model,
    optionally profiling positive linear amplitudes.
    The ND fit uses unweighted residuals at finite, unmasked observations.
    Separable-peak results include fitted peaks for ``simulate_peaks``.
    Results identify parameter names and units in Jacobian column order. The local
    covariance is available for converged, full-rank fits with positive
    residual degrees of freedom and no active bounds.

    For a recovery or decay curve, use ``model="saturation_recovery"`` or
    ``model="exponential_decay"`` with ``parameters={"amplitude": ..., "tau": ...}``.
    These named models fit a native 1D dataset without a user function. They
    retain the recorded coordinates, including nonuniform or decreasing delays.
    The fitted ``tau`` is available with its coordinate unit through
    ``result.fitted_parameters["tau"]``. Recovery starts at zero and approaches
    ``amplitude``; decay starts at ``amplitude`` and approaches zero.

    Full-grid models receive a zero-valued template with detached, full-precision
    coordinates and the fixed finite/unmasked observation support. Parameters
    are passed by keyword. Array outputs use the observations' intensity units;
    dataset outputs must preserve the grid and have compatible units. The
    callback is evaluated on the full grid and must remain finite on the fit
    support. Component names and ordering must remain stable during a fit.

    With ``linear_amplitudes=True``, the callback returns a mapping of raw
    component responses. Nonnegative linear coefficients are fitted internally
    at every nonlinear step. ``result.fitted_parameters`` reports the physical
    model parameters, ``result.amplitudes`` the dimensionless linear coefficients,
    and ``result.component_names`` the ordering of ``result.components``.
    The residual degrees of freedom count both sets of estimated parameters.
    The Jacobian and ``result.parameter_names`` describe only the profiled
    nonlinear problem; a joint covariance is deliberately unavailable.

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    warm_start : `bool`, optional, default: `False`
        Preserve the current estimator configuration instead of forcing a full
        reset to default configuration values during estimator reinitialization.
        This is a general estimator-state option shared with other
        SpectroChemPy analysis classes; it should not be interpreted as a
        dedicated `Optimize` solver-backend feature.

    Examples
    --------
    Fit a measured 1D recovery curve named ``signal``:

    .. code-block:: python

        fit = scp.Optimize(
            model="saturation_recovery",
            parameters={"amplitude": 4000, "tau": 2 * scp.ur.s},
        ).fit(signal)
        print(fit.result.fitted_parameters["tau"])
        prediction = fit.predict()
        residual = fit.result.residuals

    """

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------
    max_iter = tr.Integer(
        default_value=500,
        help="Maximum number of fitting iteration.",
    ).tag(config=True)

    max_fun_calls = tr.Integer(
        allow_none=True,
        help="Maximum number of function calls at each iteration.",
    ).tag(config=True)

    callback_every = tr.Integer(
        default_value=10,
        help="Number of iteration between each callback report. "
        "Used for printing or display intermediate results.",
    ).tag(config=True)

    method = tr.CaselessStrEnum(
        ["least_squares", "leastsq", "simplex", "basinhopping"],
        default_value="least_squares",
        help=(
            "High-level optimization-method selector. "
            "'least_squares' is the preferred local least-squares entrypoint; "
            "'leastsq' is a compatibility alias using the same backend family; "
            "'simplex' uses a derivative-free local search; "
            "'basinhopping' uses a global-style exploratory search."
        ),
    ).tag(config=True)

    script = tr.Unicode(help="Script defining models and parameters for fitting.").tag(
        config=True,
    )

    constraints = tr.Any(allow_none=True, help="Constraints.").tag(
        config=True,
    )  # TODO: adjust this

    dry = tr.Bool(
        default_value=False,
        help="If True, assemble the starting model without running the optimizer.",
    ).tag(config=True)

    autobase = tr.Bool(
        default_value=False,
        help="Whether to estimate and apply a linear baseline correction automatically.",
    ).tag(config=True)

    autoampl = tr.Bool(
        default_value=False,
        help="Whether to estimate initial amplitudes automatically during setup.",
    ).tag(config=True)

    amplitude_mode = tr.CaselessStrEnum(
        ["area", "height"],
        default_value="height",
        help="How line-shape amplitudes are interpreted during initialisation.",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    usermodels = tr.Dict(default_value={}, help="User defined models.")
    fp = tr.Instance(FitParameters, allow_none=True)
    modeldata = tr.List(Array())
    _nd_model = None

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        warm_start=False,
        peaks=None,
        shared_parameters=None,
        model=None,
        params=None,
        parameters=None,
        linear_amplitudes=False,
        solver_options=None,
        **kwargs,
    ):
        """
        Initialize the Optimize class with configuration parameters.

        Parameters
        ----------
        log_level : str, optional
            Logging level, by default "WARNING".
        warm_start : bool, optional
            Preserve the current estimator configuration rather than resetting
            it to defaults during estimator reinitialization. This does not
            add any dedicated `Optimize` backend-specific warm-start strategy.
        peaks : sequence of mappings, optional
            Enable ND fitting. Use the structure accepted by
            :func:`simulate_peaks`, replacing each numerical value with an
            optional mapping with keys ``value``, ``bounds`` and ``fixed``.
            A varying width requires a strictly positive lower bound.
            General axes accept ``model`` and a ``parameters`` mapping.
            A model can be a built-in name or a callable receiving numerical
            axis coordinates and keyword parameters, returning a dimensionless
            real vector. Custom parameters can declare a ``unit``.
        shared_parameters : mapping, optional
            Named parameter definitions used by ``{"shared": "name"}``
            references in heights or axis parameters. Each shared value is
            optimized once. Definitions accept ``value``, ``bounds``, ``fixed``
            and ``unit``. This option requires ``peaks``.
        model : Model, callable or str, optional
            A composable Model enables the canonical 1D fitting path, configured
            with params. Functions are evaluated on numerical coordinates without
            rescaling. This path requires amplitude_mode='area', autobase=False,
            autoampl=False and warm_start=False; script, peaks, parameters and
            grid-model options cannot be combined with it. All declarations are
            copied on entry. See the Python model guide for profile conventions.

            Enable full-grid ND fitting with ``model(template, **parameters)``.
            The template is a detached NDDataset with zero data, full-precision
            coordinates and the effective observation mask. Return a real array
            or compatible NDDataset on that grid, or an ordered mapping of
            named components. Components are summed without implicit scaling.
            Cannot be combined with peaks or script.
            The named laws instead fit a real 1D dataset with nonnegative,
            strictly monotone coordinates. Both require ``amplitude`` and ``tau``
            initial values. Amplitude uses intensity units and tau uses coordinate
            units; compatible Pint quantities are converted automatically.
            A scalar tau has a positive lower bound by default. Explicit bounds
            must also be positive. Named laws do not use ``linear_amplitudes``
            or ``solver_options``.
        params : Parameters, optional
            Initial configuration for a composable Model, copied on entry.
            Defaults to model.make_params(). Adjusted values are exposed through
            result.fitted_parameters; params remains an initial configuration.
        parameters : mapping, optional
            Named values for model. Each scalar is initially free. A definition
            can contain value, bounds, fixed, unit and a positive scale for free
            parameters. Numeric
            callback arguments use the declared unit's magnitudes. Results are
            available in ``result.fitted_parameters``.
        linear_amplitudes : bool, optional
            Require named components and estimate their nonnegative amplitudes
            at every step. Result components include the fitted amplitudes;
            ``result.amplitudes`` contains their dimensionless coefficients.
            Degrees of freedom count both nonlinear and linear parameters.
            The retained Jacobian describes the profiled nonlinear residuals;
            covariance is unavailable with reason ``profiled_amplitudes``.
        solver_options : mapping, optional
            Full-grid model solver options: ftol, xtol, gtol, diff_step and
            x_scale. The solver is bounded TRF with linear least squares loss.
            Use max_fun_calls to limit the number of solver evaluations.
        **kwargs : dict
            Additional keyword arguments.
        """
        # An empty __doc__ must be placed here, else Configurable.__doc__ will appear

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        public_model = None
        self._python_model_configuration = None
        if isinstance(model, Model):
            if (
                warm_start
                or peaks is not None
                or shared_parameters is not None
                or parameters is not None
                or linear_amplitudes
                or solver_options is not None
                or "script" in kwargs
                or "usermodels" in kwargs
                or kwargs.get("constraints") is not None
            ):
                raise ValueError(
                    "A composable Model cannot be combined with script, ND/grid configuration, usermodels, legacy constraints or warm_start=True"
                )
            if (
                kwargs.get("autobase", False)
                or kwargs.get("autoampl", False)
                or str(kwargs.get("amplitude_mode", "area")).lower() != "area"
            ):
                raise ValueError(
                    "Composable models require autobase=False, autoampl=False and amplitude_mode='area'"
                )
            public_model = _prepare_public_model(model, params)
            kwargs.update(
                script="",
                usermodels=public_model[4],
                amplitude_mode="area",
                autobase=False,
                autoampl=False,
            )
            model = None
        elif params is not None:
            raise ValueError(
                "params requires a composable Model; grid callbacks use parameters"
            )
        named_curve = isinstance(model, str) and model in CURVE_MODELS
        if model is not None and (
            not (callable(model) or named_curve)
            or peaks is not None
            or "script" in kwargs
        ):
            raise ValueError(
                "model requires a callable or a named recovery/decay law "
                "and cannot be combined with peaks or script"
            )
        if named_curve and (linear_amplitudes or solver_options is not None):
            raise ValueError(
                "Named curves do not accept linear_amplitudes or solver_options"
            )
        if model is None and (
            parameters is not None or linear_amplitudes or solver_options is not None
        ):
            raise ValueError(
                "parameters, linear_amplitudes and solver_options require model"
            )
        if not isinstance(linear_amplitudes, (bool, np.bool_)):
            raise ValueError("linear_amplitudes must be boolean")
        self._nd_model = model
        self._nd_parameters = snapshot_peaks(
            parameters if parameters is not None else {}
        )
        self._nd_linear_amplitudes = bool(linear_amplitudes)
        self._nd_solver_options = deepcopy(dict(solver_options or {}))
        if shared_parameters is not None and peaks is None:
            raise ValueError("shared_parameters requires peaks")
        self._nd_shared_parameters = snapshot_peaks(shared_parameters)
        self._nd_peaks = snapshot_peaks(peaks)
        self._nd_fit = None
        if peaks is not None and "script" in kwargs:
            raise ValueError("ND fitting cannot be combined with a script")

        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )
        if public_model is not None:
            definition, initial, names, spec, _ = public_model
            self._install_model_spec(spec)
            self._python_model_configuration = (definition, initial, names)

    @classmethod
    def _from_callables(
        cls, functions, parameters=None, *, common_params=None, **kwargs
    ):
        """Compose private callable components through the canonical 1D pipeline."""
        if "usermodels" in kwargs:
            raise ValueError("Callable components cannot be combined with usermodels")
        spec, usermodels = _callable_models_spec(
            functions, parameters, common_params=common_params
        )
        kwargs.setdefault("amplitude_mode", "area")
        kwargs.setdefault("autobase", False)
        kwargs.setdefault("autoampl", False)
        return cls._from_model_spec(spec, usermodels=usermodels, **kwargs)

    @classmethod
    def _from_callable(
        cls, function, parameters=None, *, label="model", common_params=None, **kwargs
    ):
        """Prepare a private scalar callable model without a user-written script."""
        if "usermodels" in kwargs:
            raise ValueError("A callable model cannot be combined with usermodels")
        spec, usermodels = _callable_model_spec(
            function, parameters, label=label, common_params=common_params
        )
        kwargs.setdefault("amplitude_mode", "area")
        kwargs.setdefault("autobase", False)
        kwargs.setdefault("autoampl", False)
        return cls._from_model_spec(spec, usermodels=usermodels, **kwargs)

    @classmethod
    def _from_model_spec(cls, model_spec, **kwargs):
        """Construct a fresh 1D estimator from a private canonical definition."""
        if not isinstance(model_spec, _FitModelSpec):
            raise TypeError("model_spec must be a _FitModelSpec")
        if kwargs.get("warm_start", False):
            raise ValueError("A model spec cannot be combined with warm_start=True")
        if "script" in kwargs:
            raise ValueError("A model spec cannot be combined with a script")
        if (
            kwargs.get("peaks") is not None
            or kwargs.get("shared_parameters") is not None
        ):
            raise ValueError("A model spec cannot be combined with ND peak definitions")
        constraints = kwargs.pop("constraints", None)
        estimator = cls(script="", **kwargs)
        estimator._install_model_spec(model_spec)
        if constraints is not None:
            estimator.constraints = constraints
        return estimator

    def _install_model_spec(self, model_spec):
        self._model_spec = deepcopy(model_spec)
        self.fp = self._model_spec.to_fitparameters()
        self._set_rendered_script(str(self.fp))

    @property
    def initial_params(self):
        """
        Independent copy of the initial Python-model configuration, or None.

        Fitting does not update this configuration. Read result.fitted_parameters
        for fitted magnitudes and result.stderr for free-parameter uncertainty.
        """
        configuration = self._python_model_configuration
        return configuration[1].copy() if configuration is not None else None

    # ----------------------------------------------------------------------------------
    # Private methods ( overriding abstract methods)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):
        if self._python_model_configuration is not None and (
            self.autobase or self.autoampl or self.amplitude_mode.lower() != "area"
        ):
            raise ValueError(
                "Composable models require autobase=False, autoampl=False and amplitude_mode='area'"
            )
        if self._python_model_configuration is not None:
            for _, parameter in self._model_spec._iter_varying():
                if parameter.value in parameter.bounds:
                    raise ValueError(
                        "Free initial values must lie strictly inside their bounds; "
                        "create another Optimize with an interior starting value"
                    )
        # NMR
        # sequence = kargs.get('sequence', 'ideal_pulse')
        # self.sequence = PulseSequence(type=sequence)

        # for now, we only work with 1D data
        if X.ndim > 1 and all(np.array(X.shape) > 1):
            raise NotImplementedError("Only 1D data are supported for now")

        self._model_spec = getattr(self, "_model_spec", None)
        if self._model_spec is None:
            self._model_spec = _FitModelSpec.from_fitparameters(self.fp)

        # create model data
        modeldata, modelnames, model_A, model_a, model_b = self._get_modeldata(X)

        global niter, everyiter, ncalls, chi2
        ncalls = 0
        everyiter = self.callback_every
        niter = 0

        # # internally defined function chi2
        # def funchi2(params, datasets, *constraints):
        #     """
        #     Return sum((y - x)**2)
        #     """
        #     global chi2, ncalls
        #     # model spectrum
        #
        #     chi2 = 0
        #     som = 0
        #     ncalls += 1
        #
        #     for exp_idx, dataset in enumerate(datasets):
        #         modeldata = self._get_modeldata(dataset, exp_idx)[0]
        #         # baseline is already summed with modeldata[-1]
        #
        #         # important to work with the real component of dataset
        #         # not the complex number
        #         data = dataset.real.data.squeeze()
        #
        #         # if not dataset.is_2d:
        #         mdata = modeldata[-1]  # modelsum
        #
        #         # else:
        #         #    mdata = modeldata.values
        #
        #         merror = 1.0
        #         # if dataset.is_2d:
        #         #     if constraints:
        #         #
        #         #         # Case of SQ-DQ experiments
        #         #         if self.kind == 'SQ-DQ' and \
        #         #                         'max_connections' in constraints[0]:
        #         #             # check connectivity numbers
        #         #             nbconnections = {}
        #         #             for key in params.keys():
        #         #                 if 'pos1' in key:
        #         #                     connect = key[-2:]
        #         #                     key = 'ampl_line_' + connect  # get amplitude
        #         #                     ki = connect[0].upper()
        #         #                     if ki not in nbconnections.keys():
        #         #                         nbconnections[ki] = 0
        #         #                     if int(params[key]) > 0:
        #         #                         nbconnections[ki] += 1
        #         #             for k, v in nbconnections.iteritems():
        #         #                 if v > constraints[0]['max_connections']:
        #         #                     merror *= v * 10.
        #
        #         diff = data - mdata
        #         chi2 += np.sum(diff**2) * merror
        #         som += np.sum(data[0] ** 2)
        #
        #     chi2 = np.sqrt(chi2 / som)
        #     # reset log_level
        #     return chi2

        # Residuals and chi2 functions -----------------------------------------------

        full_axis, keep, fit_axis = self._model_grid()

        def fun_residuals(params, data):
            global ncalls
            ncalls += 1
            return data - self._get_modelsum(params, data, full_axis, keep, fit_axis)

        def fun_chi2(params, data):
            """Return sum((y - x)**2)."""
            global chi2

            # model
            res = fun_residuals(params, data)
            chi2 = np.sum(res**2)  # * merror
            return chi2

        # callback function--------------------------------------------------------
        def callback(*args, **kwargs):
            """Log info function for callback."""
            global niter, chi2, everyiter, ncalls
            niter += 1

            if niter % everyiter != 0:
                return

            display.clear_output(wait=True)
            info_(f"Iterations: {niter}, Calls: {ncalls} (chi2: {chi2:.5f})")
            sys.stdout.flush()

        # ------------------------------------------------------------------------------

        model_spec = self._model_spec  # starting canonical model representation

        func = (
            fun_chi2
            if self.method not in ["leastsq", "least_squares"]
            else fun_residuals
        )

        fopt = None
        solver_meta = {
            "success": False,
            "status": None,
            "message": "",
            "physical_stationarity_status": "not_checked",
            "physical_stationarity_reason": "no_optimization",
            "physical_projected_gradient": None,
            "stationarity_evaluations": 0,
        }
        solver_artifacts = {
            "jacobian": None,
            "jacobian_backend": None,
        }
        fit_config = {
            "method": self.method,
            "max_iter": self.max_iter,
            "max_fun_calls": self.max_fun_calls,
            "dry": self.dry,
            "autobase": self.autobase,
            "autoampl": self.autoampl,
            "amplitude_mode": self.amplitude_mode,
            "constraints": self.constraints,
        }
        evaluate_only = (
            self._python_model_configuration is not None
            and model_spec.count_varying() == 0
        )
        if not self.dry and evaluate_only:
            residuals = np.ma.masked_invalid(
                fun_residuals(model_spec, X.real.squeeze())
            )
            fopt = _scalar_from_masked(np.ma.sum(residuals**2), default=0.0)
            if self.method in ["leastsq", "least_squares", "lm", "trf"]:
                fopt *= 0.5
            solver_meta = {
                **solver_meta,
                "success": True,
                "status": None,
                "message": "No free parameters; model evaluated without optimization.",
            }
        elif not self.dry:
            model_spec, fopt, solver_meta, solver_artifacts = _optimize(
                func,
                model_spec,
                args=(X.real.squeeze(),),
                maxfun=self.max_fun_calls,
                maxiter=self.max_iter,
                method=self.method,
                constraints=self.constraints,
                callback=callback,
            )
            self._model_spec = model_spec

        fp = _sync_fitparameters_from_modelspec(self.fp, model_spec)

        # Store solver metadata for the result object.
        # Created on every _fit call so it always reflects the last fit.
        self._fit_meta = {
            "cost": float(fopt) if fopt is not None else None,
            "niter": niter,
            "ncalls": ncalls,
            **solver_meta,
        }
        if self._python_model_configuration is not None:
            self._fit_meta["optimization_performed"] = (
                not self.dry and not evaluate_only
            )
        self._fit_config = fit_config
        self._solver_artifacts = solver_artifacts

        # Replace the previous script with a rendered view of the new parameter
        # values without re-parsing it into canonical state (see maintainer
        # decision recorded in the FitParameters ADR amendment, 2026-08-13).
        # The canonical spec keeps the full-precision optimized values and the
        # public fp view keeps its identity and synced values. The render is
        # written without running the script validator (which would rebuild the
        # spec from the rounded text) but observers of ``script`` are still
        # notified as for a normal assignment.
        self._set_rendered_script(str(fp))

        # log.info the results
        display.clear_output(wait=True)
        info_("*" * 50)
        if not self.dry:
            info_("Result:")
        else:
            info_("Starting parameters:")
        info_("*" * 50 + "\n")
        info_(self.script)

        # reset dry and continue to show starting model
        self.dry = False

        # return fit results
        modeldata, names, A, a, b = self._get_modeldata(X)
        if X.squeeze().ndim == 1:
            # C in this case is just the A for all species
            # (not very useful here but it will be necessary for 2D
            C = np.ones((1, self._n_components)) * A  # TODO: check this
            # we eventually add baseline to the components
            start = 0 if self.autobase else 1
            components = modeldata[start:-1]
            total = modeldata[-1]
        else:
            # todo
            raise NotImplementedError("Fit not implemented for nD data yet!")
        return C, components, total, A, a, b

    # ----------------------------------------------------------------------------------
    # Private methods for validation
    # ----------------------------------------------------------------------------------
    @tr.validate("method")
    def _method_validate(self, proposal):
        method = proposal.value

        return method.lower()

    @tr.validate("usermodels")
    def _usermodels_validate(self, proposal):
        usermodels = proposal.value
        if usermodels is None:
            usermodels = {}
        newdict = {}
        for key, value in usermodels.items():

            class usermodel(models_.usermodel):
                pass

            if isinstance(value, models_.usermodel):
                usermodel.f = staticmethod(value.f)
                usermodel.args = list(value.args)
            else:
                usermodel.f = staticmethod(value)
                usermodel.args = inspect.getfullargspec(value).args[1:]
            newdict[key.lower()] = usermodel
        return newdict

    @tr.validate("script")
    def _script_validate(self, proposal):
        if self._python_model_configuration is not None:
            raise ValueError(
                "Create another Optimize to replace a Python model with a script"
            )
        spec, errors = _parse_script_to_modelspec(proposal.value, self.usermodels)
        if errors:
            err = errors[0]
            raise ValueError(str(err))
        self._model_spec = spec
        self.fp = spec.to_fitparameters()
        return proposal.value

    # ----------------------------------------------------------------------------------
    def _set_rendered_script(self, value):
        """
        Assign the ``script`` trait without re-parsing it into canonical state.

        The post-fit script is a rendered representation of the fitted values;
        the canonical spec and the ``fp`` view already reflect the executed
        optimization. Unlike a regular assignment, the script validator (which
        would rebuild the canonical state from the rendered text) is bypassed,
        but observers of ``script`` are still notified as for a normal
        assignment (see traitlets ``TraitType.set`` semantics).
        """
        old = self.script  # materialize the default value when needed
        self._trait_values["script"] = value
        if old != value:
            self._notify_trait("script", old, value)

    @tr.validate("constraints")
    def _constraints_validate(self, proposal):
        constraints = proposal.value
        if constraints in (None, {}, []):
            return None

        # Validate parameter-name references against the canonical model spec
        # rather than the legacy FitParameters view. The spec is always kept in
        # sync with ``self.fp`` by ``_script_validate`` and exposes the same
        # parameter names, so this removes the only FitParameters dependency of
        # the structured-validation flow.
        model_spec = getattr(self, "_model_spec", None)
        errors = _validate_constraints_content(constraints, model_spec)
        if errors:
            raise ValueError(str(errors[0]))

        return _canonicalize_constraints(constraints)

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------
    def validate_script(self, script=None):
        """
        Validate a fitting script without running an optimisation.

        Returns a list of :class:`ScriptError` objects describing every problem
        found in the script.  An empty list means the script is valid.

        Parameters
        ----------
        script : str, optional
            The script to validate.  If ``None`` (default), validates the
            currently assigned script.

        Returns
        -------
        list of :class:`ScriptError`
            Empty list when the script is valid.

        Examples
        --------
        >>> opt = scp.Optimize()
        >>> errors = opt.validate_script(script)
        >>> if errors:
        ...     for err in errors:
        ...         print(err)
        """
        if script is None:
            script = self.script
        _, errors = _parse_script_to_modelspec(script, self.usermodels)
        return errors

    def validate_constraints(self, constraints=None, script=None):
        """
        Validate lightweight constraint specifications without running a fit.

        This method currently validates:

        - the overall constraints container shape;
        - recognized constraint keys / types;
        - references to parameter names defined by the fitting script.

        It does not guarantee that every accepted constraint is fully enforced
        by the current optimization backend. The present goal is to provide a
        stable validation surface before broader constraint execution semantics
        are expanded.

        Parameters
        ----------
        constraints : dict or sequence of dict, optional
            Constraint specification(s) to validate. If ``None`` (default),
            validates :attr:`constraints`.
        script : str, optional
            Script used to resolve parameter names. If ``None`` (default), uses
            the currently assigned :attr:`script`.

        Returns
        -------
        list of :class:`ConstraintError`
            Empty list when the constraint specification is valid.
        """
        if constraints is None:
            constraints = self.constraints
        if script is None:
            script = self.script

        model_spec, script_errors = _parse_script_to_modelspec(script, self.usermodels)
        if script_errors:
            return [
                ConstraintError(
                    0,
                    constraints,
                    "Cannot validate constraints because the fitting script is invalid. "
                    f"First script error: {script_errors[0].message}",
                ),
            ]

        return _validate_constraints_content(constraints, model_spec)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.default("_script")
    def _script_default(self):
        """Return a default script."""
        if self._is_grid_fit:
            return ""
        return """
        # -----------------------------------------------------------------------
        # syntax for parameters definition:
        # name: value, low_bound,  high_bound
        # prefix:
        #  # for comments
        #  * for fixed parameters
        #  $ for variable parameters
        #  > for reference to a parameter in the COMMON block
        #    (> is forbidden in the COMMON block)
        # common block parameters should not have a _ (underscore) in their names
        # -----------------------------------------------------------------------

        COMMON:
        # common parameters block
        # $ gwidth: 1.0, 0.0, none
          $ gratio: 0.5, 0.0, 1.0

        MODEL: LINE_1
        shape: voigtmodel
            $ ampl:  1.0, 0.0, none
            $ pos:   0.0, -100.0, 100.0
            > ratio: gratio
            $ width: 1.0, 0, 100
        """

    # def _repr_html_(self):
    #     if not self.datasets:
    #         return htmldoc(self.script)
    #     else:
    #         return self.message

    def _evaluate_component(self, x, model, prepared_spec):
        return getmodel(
            x,
            modelname=model,
            par=prepared_spec.component_view(model),
            amplitude_mode=self.amplitude_mode,
            usermodels=self.usermodels,
        )

    def _model_grid(self):
        _, dim = self._X.get_axis(-1)
        x = self._X_coordset[dim].data
        keep = ~np.all(self._X_mask, axis=-2) if np.any(self._X_mask) else slice(None)
        return x, keep, x[keep]

    def _get_modelsum(self, model_spec, data, full_axis, keep, fit_axis):
        """Stream ordinary solver totals, preserving component scaling and callbacks."""
        prepared_spec = prepare_model(model_spec)
        models = [component.label for component in prepared_spec.components]
        baseline_count = sum(model.startswith("baseline") for model in models)
        baseline_first = all(
            model.startswith("baseline") for model in models[:baseline_count]
        )
        if (
            self.autobase
            or self.usermodels
            or not models
            or not baseline_first
            or full_axis.size <= 1
        ):
            return self._assemble_modeldata(
                prepared_spec, models, data, full_axis, keep, fit_axis
            )[0][-1]

        total = np.zeros(full_axis.size, dtype=np.float64)
        for model in models[:baseline_count]:
            total += self._evaluate_component(full_axis, model, prepared_spec)

        peaks = iter(models[baseline_count:])
        component = np.empty_like(total)
        trial = np.empty_like(total)
        for index, model in enumerate(peaks, start=baseline_count):
            component[:] = self._evaluate_component(full_axis, model, prepared_spec)
            try:
                with np.errstate(over="raise", invalid="raise"):
                    np.add(total, component, out=trial)
            except FloatingPointError:
                remaining = np.empty((len(models) - index + 1, total.size))
                remaining[0] = total
                remaining[1] = component
                for row, pending in zip(remaining[2:], peaks, strict=True):
                    row[:] = self._evaluate_component(full_axis, pending, prepared_spec)
                total = remaining.sum(axis=0)
                break
            total, trial = trial, total

        total = total[keep]
        total += 0.0 * fit_axis + 0.0
        return total

    def _get_modeldata(self, X, exp_idx=1, model_spec=None):
        # exp_idx is not used for the moment, but will be necessary for multidataset
        # fitting

        if model_spec is None:
            model_spec = getattr(self, "_model_spec", None)
        if model_spec is None:
            model_spec = _FitModelSpec.from_fitparameters(self.fp)

        # Prepare canonical model representation.
        prepared_spec = prepare_model(model_spec)

        # Get the list of models
        models = [comp.label for comp in prepared_spec.components]
        expedata = X.real.squeeze()
        if expedata.ndim > 1:
            raise NotImplementedError("Fit not implemented for nD data yet!")

        return self._assemble_modeldata(
            prepared_spec, models, expedata, *self._model_grid()
        )

    def _assemble_modeldata(self, prepared_spec, models, expedata, x, keep, fit_axis):
        self._n_components = nbmodels = len(models)
        modeldata = np.zeros((nbmodels + 2, x.size), dtype=np.float64)

        if nbmodels < 1:
            names = ["baseline", "modelsum"]
            return modeldata, names

        # Calculates model data
        # The first row (i=0) of the modeldata array is the baseline,
        # so we fill the array starting at row 1
        row = 0
        names = [
            "baseline",
        ]

        for model in models:
            calc = self._evaluate_component(x, model, prepared_spec)
            if not model.startswith("baseline"):
                row += 1
                modeldata[row] = calc
                names.append(model)
            else:
                modeldata[0] += calc

        # make the sum
        modeldata[row + 1] = modeldata.sum(axis=0)
        names.append("modelsum")

        # remove unused column
        modeldata = modeldata[: row + 2]

        # remove masked column
        modeldata = modeldata[:, keep]
        x = fit_axis

        if self.autobase:
            A, a, b = self._ampbas(x, expedata, modeldata[-1])
        else:
            A, a, b = 1.0, 0.0, 0.0

        # (fitzone-fitzone[0], data.take(fitzone),
        # modeldata[-1].take(fitzone))

        modeldata = A * modeldata
        baseline = a * x + b  # a*(xi-fitzone[0]) + b

        # update the modeldata
        modeldata[0] += baseline
        modeldata[-1] += baseline

        # return modeldata
        return modeldata, names, A, a, b

    @staticmethod
    def _parsing(expr, param):
        def _replace(m):
            kw = m.group(1)
            if kw in param:
                return str(param[kw])
            if kw in np.__dict__:
                return f"np.{kw}"
            return m.group(0)

        return re.sub(r"\b(?<!np\.)([a-z]+[0-9]*)\b", _replace, str(expr))

    def _prepare(self, param, exp_idx=1):
        # This function is needed for the script related to modelfunction
        #
        # exp_idx: int, contains the index of the experiment

        new_param = param.copy()

        for key in param:
            if param.reference[key]:
                new_param.reference[key] = False
                # important to put it here
                # before other instruction
                # try to interpret the given refpar expression
                refpar = param[key]
                while True:
                    par = self._parsing(refpar, new_param)
                    if par == refpar:
                        break
                    refpar = par
                try:
                    new_param[key] = eval(str(refpar))  # noqa: S307
                except Exception as err:
                    raise ValueError(
                        f"Cannot evaluate the expression {key}: {param[refpar]}",
                    ) from err

                new_param.fixed[key] = True
                new_param.reference[key] = True  # restore it for the next call

            # if isinstance(new_param[key], list):
            #     new_param.data[key] = new_param.data[key][exp_idx]

        return new_param

    # ==================================================================================
    # automatic calculation of amplitude and baseline
    # ==================================================================================
    @staticmethod
    def _ampbas(xi, expe, calc):
        # Automatically calculate correct amplitude A and baseline
        # (baseline linear model a*i+b) by determining the zero of the first derivative
        # with respect to A, a, and b
        expe = expe.squeeze()
        n = xi.size
        reduce = sum
        if n and all(
            values.shape == (n,) and values.dtype == np.float64
            for values in (xi, expe, calc)
        ):
            cumulative = np.empty(n + 1, dtype=np.float64)

            def ordered_sum(values):
                cumulative[0] = 0.0
                cumulative[1:] = values
                with np.errstate(over="ignore", invalid="ignore"):
                    np.add.accumulate(cumulative, out=cumulative)
                result = cumulative[-1]
                return result if np.isfinite(result) else sum(values)

            reduce = ordered_sum

        sE = reduce(expe)
        sF = reduce(calc)
        sFI = reduce(xi * calc)
        sFd = reduce(calc * calc)
        sI = reduce(xi)
        sEF = reduce(expe * calc)
        sEI = reduce(xi * expe)
        sId = reduce(xi**2)

        den = n * sFI**2 - n * sFd * sId + sF**2 * sId - 2 * sF * sFI * sI + sFd * sI**2

        a = (
            -sE * (sF * sFI - sFd * sI)
            + sEF * (n * sFI - sF * sI)
            - sEI * (n * sFd - sF**2)
        ) / den

        A = (
            sE * (sF * sId - sFI * sI)
            - sEF * (n * sId - sI**2)
            + sEI * (n * sFI - sF * sI)
        ) / den

        b = (
            sE * (sFI**2 - sFd * sId)
            + sEF * (sF * sId - sFI * sI)
            - sEI * (sF * sFI - sFd * sI)
        ) / den

        # in case the modeldata is zero, to avoid further errors
        if np.isnan(A):  # pragma: no cover
            A = 0.0
        if np.isnan(a):  # pragma: no cover
            a = 0.0
        if np.isnan(b):  # pragma: no cover
            b = 0.0
        return A, a, b

    @staticmethod
    def _ampbas2D(xi, yj, expe, calc):  # pragma: no cover
        n = float(xi.size)
        m = float(yj.size)
        sE = expe.sum()
        sF = calc.sum()
        sFI = (xi * calc).sum()
        sFJ = (yj * calc.T).sum()
        sFd = (calc * calc).sum()
        sI = sum(xi)
        sJ = sum(yj)
        sIJ = ((np.ones_like(calc) * xi).T * yj).sum()
        sEF = (expe * calc).sum()
        sEI = (xi * expe).sum()
        sEJ = (yj * expe.T).sum()
        sId = sum(xi**2)
        sJd = sum(yj**2)

        den = (
            -(m**2) * n**2 * sFd * sId * sJd
            + m**2 * n * sFJ**2 * sId
            + m**2 * n * sFd * sI**2 * sJd
            - m**2 * sFJ**2 * sI**2
            + m * n**2 * sFI**2 * sJd
            + m * n**2 * sFd * sId * sJ**2
            + m * n * sF**2 * sId * sJd
            - 2 * m * n * sF * sFI * sI * sJd
            - 2 * m * n * sF * sFJ * sId * sJ
            + 2 * m * n * sFI * sFJ * sI * sJ
            - 2 * m * n * sFI * sFJ * sIJ
            - 2 * m * n * sFd * sI * sIJ * sJ
            + m * n * sFd * sIJ**2
            + 2 * m * sF * sFJ * sI * sIJ
            - n**2 * sFI**2 * sJ**2
            + 2 * n * sF * sFI * sIJ * sJ
            - sF**2 * sIJ**2
        )

        c = (
            sE
            * (
                -m * n * sFd * sId * sJd
                + m * sFJ**2 * sId
                + n * sFI**2 * sJd
                - 2 * sFI * sFJ * sIJ
                + sFd * sIJ**2
            )
            + sEF
            * (
                m * n * sF * sId * sJd
                - m * n * sFI * sI * sJd
                - m * n * sFJ * sId * sJ
                + m * sFJ * sI * sIJ
                + n * sFI * sIJ * sJ
                - sF * sIJ**2
            )
            + sEI
            * (
                m * n * sFd * sI * sJd
                - m * sFJ**2 * sI
                - n * sF * sFI * sJd
                + n * sFI * sFJ * sJ
                - n * sFd * sIJ * sJ
                + sF * sFJ * sIJ
            )
            + sEJ
            * (
                m * n * sFd * sId * sJ
                - m * sF * sFJ * sId
                + m * sFI * sFJ * sI
                - m * sFd * sI * sIJ
                - n * sFI**2 * sJ
                + sF * sFI * sIJ
            )
        ) / den

        a = (
            n
            * sEF
            * (
                m * n * sFI * sJd
                - m * sF * sI * sJd
                + m * sFJ * sI * sJ
                - m * sFJ * sIJ
                - n * sFI * sJ**2
                + sF * sIJ * sJ
            )
            + n
            * sEI
            * (
                -m * n * sFd * sJd
                + m * sFJ**2
                + n * sFd * sJ**2
                + sF**2 * sJd
                - 2 * sF * sFJ * sJ
            )
            + sE
            * (
                m * n * sFd * sI * sJd
                - m * sFJ**2 * sI
                - n * sF * sFI * sJd
                + n * sFI * sFJ * sJ
                - n * sFd * sIJ * sJ
                + sF * sFJ * sIJ
            )
            - sEJ
            * (
                m * n * sFI * sFJ
                + m * n * sFd * sI * sJ
                - m * n * sFd * sIJ
                - m * sF * sFJ * sI
                - n * sF * sFI * sJ
                + sF**2 * sIJ
            )
        ) / den

        A = (
            m
            * n
            * sEF
            * (
                -m * n * sId * sJd
                + m * sI**2 * sJd
                + n * sId * sJ**2
                - 2 * sI * sIJ * sJ
                + sIJ**2
            )
            + m
            * sEJ
            * (
                m * n * sFJ * sId
                - m * sFJ * sI**2
                - n * sF * sId * sJ
                + n * sFI * sI * sJ
                - n * sFI * sIJ
                + sF * sI * sIJ
            )
            + n
            * sEI
            * (
                m * n * sFI * sJd
                - m * sF * sI * sJd
                + m * sFJ * sI * sJ
                - m * sFJ * sIJ
                - n * sFI * sJ**2
                + sF * sIJ * sJ
            )
            + sE
            * (
                m * n * sF * sId * sJd
                - m * n * sFI * sI * sJd
                - m * n * sFJ * sId * sJ
                + m * sFJ * sI * sIJ
                + n * sFI * sIJ * sJ
                - sF * sIJ**2
            )
        ) / den

        b = (
            m
            * sEF
            * (
                m * n * sFJ * sId
                - m * sFJ * sI**2
                - n * sF * sId * sJ
                + n * sFI * sI * sJ
                - n * sFI * sIJ
                + sF * sI * sIJ
            )
            + m
            * sEJ
            * (
                -m * n * sFd * sId
                + m * sFd * sI**2
                + n * sFI**2
                + sF**2 * sId
                - 2 * sF * sFI * sI
            )
            + sE
            * (
                m * n * sFd * sId * sJ
                - m * sF * sFJ * sId
                + m * sFI * sFJ * sI
                - m * sFd * sI * sIJ
                - n * sFI**2 * sJ
                + sF * sFI * sIJ
            )
            - sEI
            * (
                m * n * sFI * sFJ
                + m * n * sFd * sI * sJ
                - m * n * sFd * sIJ
                - m * sF * sFJ * sI
                - n * sF * sFI * sJ
                + sF**2 * sIJ
            )
        ) / den

        # in case the modeldata is zero, to avoid further errors
        if np.isnan(A):
            A = 0.0
        if np.isnan(a):
            a = 0.0
        if np.isnan(b):
            b = 0.0
        if np.isnan(c):
            c = 0.0
        return A, a, b, c

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    def _transform(self, X=None):
        # X is ignored for Optimize
        # this method is present for coherence with other decomposition methods
        return self._outfit[0].copy()

    def _inverse_transform(self, X_transform=None):
        # X_transform is ignored for Optimize
        # this method is present for coherence with other decomposition methods
        X_transform = self._outfit[2].copy()
        if X_transform.ndim == 1:
            # we need a seconddimension of size 1 for the restoration of masks
            X_transform = X_transform[np.newaxis]
        return X_transform

    def predict(self):
        """
        Return the fitted model.

        Returns
        -------
        `NDDataset`
            The fitted model.

        """
        if self._is_grid_fit:
            return self._require_nd_fit()["fitted"].copy()
        return self.inverse_transform()

    def _get_components(self):
        return self._outfit[1]  # the first is the baseline, the last is the sum

    # ----------------------------------------------------------------------------------
    # Public methods/properties
    # ----------------------------------------------------------------------------------
    def fit(self, X):
        """
        Perform a non-linear optimization of the ``X`` dataset.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.

        Returns
        -------
        self
            The fitted instance itself.

        See Also
        --------
        fit_transform :  Fit the model with an input dataset ``X`` and apply the dimensionality reduction on ``X``.
        fit_reduce : Alias of `fit_transform` (Deprecated).

        """
        if not self._is_grid_fit:
            return super().fit(X, Y=None)

        self._fitted = False
        self._nd_fit = None
        if (
            self.method.lower() != "least_squares"
            or self.script
            or self.constraints
            or self.usermodels
            or self.autobase
            or self.autoampl
            or self.dry
            or self.amplitude_mode.lower() != "height"
        ):
            raise ValueError(
                "ND fitting requires least_squares and explicit peak parameters"
            )
        max_calls = self.max_fun_calls or 4 * self.max_iter

        if isinstance(self._nd_model, str):
            fit, solution = fit_curve(X, self._nd_model, self._nd_parameters, max_calls)
        elif self._nd_model is not None:
            fit, solution = fit_callable_model(
                X,
                self._nd_model,
                self._nd_parameters,
                max_calls,
                self._nd_linear_amplitudes,
                self._nd_solver_options,
            )
        else:
            fit, solution = fit_separable_peaks(
                X, self._nd_peaks, max_calls, self._nd_shared_parameters
            )
        diagnostics = fit["diagnostics"]
        diagnostics.update(_normalize_solver_meta("trf", solution))
        if solution is None:
            diagnostics.update(
                success=True,
                message="No nonlinear parameters to optimize"
                if self._nd_model is not None
                else "All peak parameters are fixed",
            )
        else:
            diagnostics.update(cost=diagnostics["rss"] / 2, ncalls=solution.nfev)
        fit["jacobian"] = _freeze_jacobian_artifact(fit["jacobian"])
        fit["config"] = {
            "method": "trf",
            "mode": "named_curve"
            if isinstance(self._nd_model, str)
            else "callable_model"
            if self._nd_model is not None
            else "separable_peaks",
            "max_fun_calls": max_calls,
        }
        if self._nd_model is not None:
            fit["config"]["solver_options"] = deepcopy(self._nd_solver_options)
            fit["config"]["linear_amplitudes"] = self._nd_linear_amplitudes
        if isinstance(self._nd_model, str):
            fit["config"]["model"] = self._nd_model
        self._nd_fit = fit
        self._n_components = len(fit["components"])
        self._fitted = True
        return self

    @property
    def _is_grid_fit(self):
        return self._nd_peaks is not None or self._nd_model is not None

    def _require_nd_fit(self):
        if not self._fitted or self._nd_fit is None:
            raise NotFittedError("Call fit before accessing the ND fit")
        return self._nd_fit

    @property
    def X(self):
        """Return the fitted observations."""
        if self._is_grid_fit:
            return self._require_nd_fit()["observed"].copy()
        return super().X

    @property
    def components(self):
        """Return fitted components, one dataset per peak in ND mode."""
        if self._is_grid_fit:
            return self.get_components()
        return super().components

    def get_components(self, n_components=None):
        """Return all or the requested number of fitted components."""
        if self._is_grid_fit:
            return tuple(
                part.copy()
                for part in self._require_nd_fit()["components"][:n_components]
            )
        return super().get_components(n_components)

    def transform(self, X=None, **kwargs):
        """Apply the legacy dimensionality reduction."""
        if self._is_grid_fit:
            raise NotImplementedError("ND fitting does not provide projections")
        return super().transform(X, **kwargs)

    def inverse_transform(self, X_transform=None, **kwargs):
        """Apply the legacy inverse transform."""
        if self._is_grid_fit:
            raise NotImplementedError("Use predict() to reconstruct the ND fit")
        return super().inverse_transform(X_transform, **kwargs)

    def fit_transform(self, X, Y=None, **kwargs):
        """Fit and apply the legacy dimensionality reduction."""
        if self._is_grid_fit:
            raise NotImplementedError("ND fitting does not provide projections")
        return super().fit_transform(X, Y, **kwargs)

    @property
    def jacobian(self):
        """
        Return the retained solver Jacobian when the backend provides one.

        In ND mode, rows follow finite, unmasked observations in native
        C order. Columns follow the free parameters in physical magnitudes,
        as identified by ``result.parameter_names`` and ``result.parameter_units``.

        General separable ND profiles use the fitted physical scale for numerical
        post-fit derivatives of built-in widths and time constants. Their local
        covariance uses this post-fit Jacobian. Finite-difference accuracy and
        local least-squares assumptions still apply; this does not independently
        establish convergence or stationarity.

        For 1D script and composable Python models, this remains the raw
        optimizer-coordinate Jacobian, including the historical bound
        transformation. Result covariance and stderr are transformed back to
        physical free parameters; this property is preserved for compatibility.

        Returns
        -------
        ndarray or None
            Immutable Jacobian matrix for least-squares-backed methods.
            Returns ``None`` for backends that do not naturally expose a
            Jacobian, and for dry fits.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before accessing the jacobian",
            )

        if self._is_grid_fit:
            return self._require_nd_fit()["jacobian"]
        return getattr(self, "_solver_artifacts", {}).get("jacobian")

    # ----------------------------------------------------------------------------------
    # Result object
    # ----------------------------------------------------------------------------------
    @property
    def result(self):
        """
        Return the Optimize fit result object.

        Returns
        -------
        FitResult
            Result object containing fitted model outputs,
            solver diagnostics, and estimator parameters.

        Notes
        -----
        For a script fit, the entries of ``covariance``, ``variance``,
        ``stderr``, ``correlation`` and ``confidence_intervals`` follow the free
        parameters sorted by name, not their order in the script. The names are
        those listed in `fp`: ``<parameter>_<model>`` for model parameters
        (e.g., ``pos_line1``; script names are lowercased) and their own name
        for COMMON parameters. Fixed and referenced parameters are excluded, so
        the order is ``[name for name in sorted(fp) if not fp.fixed[name]]``.
        For a fit with ``peaks``, the order is given by
        ``result.parameter_names``, which script fits do not provide.

        For 1D script and Python-model fits, covariance and its derived
        uncertainty arrays describe physical free parameters, accounting for
        the bound transform. They are unavailable near bounds, for deficient
        rank, unsuccessful optimization or missing prerequisites; diagnostics
        include ``covariance_reason``. The raw ``jacobian`` property remains
        in optimizer coordinates for these fits.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before accessing the result",
            )

        if self._is_grid_fit:
            fit = self._require_nd_fit()
            return FitResult(
                estimator="Optimize",
                parameters=deepcopy(fit["config"]),
                outputs={
                    "fitted": fit["fitted"].copy(),
                    "components": self.components,
                    "residuals": fit["residuals"].copy(),
                    "peaks": list(snapshot_peaks(fit["peaks"])),
                    "parameter_names": fit["names"],
                    "parameter_units": fit["units"],
                    **(
                        {
                            key: snapshot_peaks(fit[key])
                            for key in (
                                "fitted_parameters",
                                "amplitudes",
                                "component_names",
                            )
                        }
                        if self._nd_model is not None
                        else {}
                    ),
                },
                diagnostics=fit["diagnostics"].copy(),
                covariance=fit["covariance"],
                parameter_values=fit["values"],
            )

        # NOTE: a new FitResult is created on every access.
        # Caching is deliberately deferred to keep the implementation
        # simple and aligned with the PCA / SVD result behaviour.

        fitted = self.predict()
        observed = self.X
        residuals, fit_diagnostics = _compute_fit_diagnostics(
            observed,
            fitted,
            getattr(self, "_fit_meta", {}),
            getattr(self, "_model_spec", None),
        )
        residuals = self._apply_analysis_output_metadata(
            residuals,
            role_id="residuals",
            meta_from="_X",
            direct_x=None,
            direct_y=None,
            direct_x_kind="none",
            direct_y_kind="none",
        )
        covariance = _compute_covariance_matrix(
            observed,
            fitted,
            self.jacobian,
            fit_diagnostics,
            transform_derivatives=self._solver_artifacts.get(
                "parameter_transform_derivatives"
            ),
            near_bounds=self._solver_artifacts.get("near_parameter_bounds", False),
        )
        fit_diagnostics["jacobian_coordinates"] = "optimizer"
        parameter_values = _extract_varying_parameter_values(
            getattr(self, "_model_spec", None),
        )
        fit_config = getattr(
            self,
            "_fit_config",
            {
                "method": self.method,
                "max_iter": self.max_iter,
                "max_fun_calls": self.max_fun_calls,
                "dry": self.dry,
                "autobase": self.autobase,
                "autoampl": self.autoampl,
                "amplitude_mode": self.amplitude_mode,
                "constraints": self.constraints,
            },
        )

        python_outputs = {}
        if self._python_model_configuration is not None:
            _, _, names = self._python_model_configuration
            resolved = prepare_model(self._model_spec)
            inverse_names = {internal: public for public, internal in names.items()}
            python_outputs = {
                "fitted_parameters": {
                    public: resolved.common_params[internal].value
                    for public, internal in names.items()
                },
                "parameter_names": tuple(
                    inverse_names[key] for key, _ in self._model_spec._iter_varying()
                ),
            }
            fit_diagnostics["derived_uncertainty_status"] = "not_propagated"

        return FitResult(
            estimator="Optimize",
            parameters=fit_config.copy(),
            outputs={
                "fitted": fitted,
                "components": self.components,
                "residuals": residuals,
                **python_outputs,
            },
            diagnostics=fit_diagnostics,
            covariance=covariance,
            parameter_values=parameter_values,
        )

    def report(self):
        """
        Return a plain-text report from the canonical FitResult.

        Covariance and standard errors describe the free physical parameters;
        uncertainty propagation to expression-defined parameters is not provided.
        """
        result = self.result
        lines = [
            "Optimize fit report",
            f"Method: {result.parameters.get('method', self.method)}",
        ]
        errors = result.stderr
        error_map = (
            dict(zip(result.outputs.get("parameter_names", ()), errors, strict=True))
            if errors is not None and "parameter_names" in result.outputs
            else {}
        )
        for name, value in result.outputs.get("fitted_parameters", {}).items():
            uncertainty = f" +/- {error_map[name]:.4g}" if name in error_map else ""
            lines.append(f"{name}: {value:.8g}{uncertainty}")
        for name, value in result.diagnostics.items():
            lines.append(f"{name}: {value}")
        return "\n".join(lines)

    def plot(self, **kwargs):
        """
        Plot the existing observation, fit and residual merit view.

        Keyword arguments are forwarded to :meth:`plot_merit`.
        """
        return self.plot_merit(**kwargs)


# ======================================================================================
def _optimize(
    func,
    fp0,
    args=(),
    constraints=None,
    method="least_squares",
    maxfun=None,
    maxiter=1000,
    ftol=1e-8,
    xtol=1e-8,
    callback=None,
):
    if constraints is None:
        constraints = {}
    global keys
    use_model_spec = isinstance(fp0, _FitModelSpec)

    def internal_func(p, dat, fp, keys, *args):
        fp = _restore_external(fp, p, keys)
        return func(fp, dat)

    def internal_callback(*args):
        if callback is None:
            return None
        return callback(*args)

    if not isinstance(fp0, FitParameters | _FitModelSpec):
        raise TypeError("fp0 is not of FitParameter or _FitModelSpec type")

    par, keys = _internal_vector(fp0)

    args = list(args)
    args.append(fp0)
    args.append(keys)
    if constraints:
        args.append(constraints)

    if not maxfun:
        maxfun = 4 * maxiter

    if method in ["leastsq", "least_squares"]:
        n_parameters = _modelspec_parameter_count(fp0) if use_model_spec else len(fp0)
        method = "lm" if n_parameters < 10 else "trf"

    result = _run_optimizer(
        internal_func,
        par,
        args=tuple(args),
        method=method,
        maxfun=maxfun,
        maxiter=maxiter,
        ftol=ftol,
        xtol=xtol,
        callback=internal_callback,
    )
    if method.lower() in ["lm", "trf"]:
        res, fopt, warnmess = result.x, result.cost, result.message
        solver_result = result
    elif method.lower() == "simplex":
        res, fopt, _, _, warnmess = result
        solver_result = None
    else:
        res, fopt, warnmess = result.x, result.fun, result.message
        solver_result = result
    solver_meta = _normalize_solver_meta(method, solver_result, warnmess)
    solver_artifacts = _extract_solver_artifacts(method, solver_result)
    solver_artifacts.update(_transform_artifacts(fp0, res, keys))

    # restore the external parameter
    fpe = _restore_external(fp0, res, keys)
    parameters = {p.name: p for p in _prepared_parameters(fpe) if not p.fixed}
    ordered = [parameters[key] for key in keys]
    physical = np.array([p.value for p in ordered], dtype=float)
    bounds = [
        (
            p.bounds[0]
            if p.bounds[0] is not None and p.bounds[0] > _LOB_THRESHOLD
            else -np.inf,
            p.bounds[1]
            if p.bounds[1] is not None and p.bounds[1] < _UPB_THRESHOLD
            else np.inf,
        )
        for p in ordered
    ]

    def physical_residual(values):
        trial = _restore_physical(deepcopy(fpe), values, keys)
        return func(trial, args[0])

    solver_meta.update(
        _physical_stationarity(
            physical_residual,
            physical,
            bounds,
            getattr(solver_result, "fun", None),
            solver_artifacts["jacobian"],
            solver_artifacts["parameter_transform_derivatives"],
            reference=getattr(args[0], "masked_data", args[0]),
        )
    )
    # for i, key in enumerate(keys):
    #    fp0.to_external(key, res[i])

    if warnmess == 1:
        warning_("Maximum number of function evaluations made.")
    if warnmess == 2:
        warning_("Maximum number of iterations reached.")

    return fpe, fopt, solver_meta, solver_artifacts


# ======================================================================================
def getmodel(
    x,
    y=None,
    modelname=None,
    par=None,
    usermodels=None,
    amplitude_mode="height",
    **kwargs,
):
    """
    Get the model for a given x vector.

    Parameters
    ----------
    x : ndarray
        Array of frequency where to evaluate the model values returned by the
        f function.
    y : ndarray or None
        None for 1D, or index for the second dimension.
    modelname : str
        Name of the model class to use.
    par : :class:`Parameters` instance
        Parameter to pass to the f function.
    usermodels: dict, optional, default is None
        Dictionary of user defined models used to extend the models available
        in spectrochempy.
    amplitude_mode : str, optional, default is 'height'
        Select the amplitude mode calculation. Can be 'height' or 'area'.
    kwargs : any
        Keywords arguments to pass to the f function.

    Returns
    -------
    `~numpy.ndarray`
        Array containing the calculated model.

    """
    model = par.model[modelname]
    try:
        modelcls = getattr(models_, model)
    except AttributeError:
        if usermodels is not None:
            try:
                modelcls = usermodels[model]
            except KeyError as e:
                raise ValueError(
                    f"Model {model} not found in spectrochempy nor in usermodels.",
                ) from e
        else:
            raise ValueError(f"Model {model} not found in spectrochempy.") from None

    # take an instance of the model
    a = modelcls()

    # get the parameters for the given model
    args = []
    for p in a.args:
        try:
            args.append(par[f"{p.lower()}_{modelname}"])
        except KeyError as e:
            if p.startswith("c_"):
                # probably the end of the list
                # due to a limited polynomial degree
                pass
            else:
                raise ValueError(e) from e

    return _evaluate_callable(
        lambda *values, **options: a.f(*values, **options),
        x,
        args,
        y=y,
        amplitude_mode=amplitude_mode,
        **kwargs,
    )
