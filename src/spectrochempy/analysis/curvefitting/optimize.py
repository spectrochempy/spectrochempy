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

import numpy as np
import traitlets as tr
from IPython import display
from scipy import optimize

from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import FitResult
from spectrochempy.analysis.curvefitting import _models as models_
from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.application.application import info_
from spectrochempy.application.application import warning_
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import signature_has_configurable_traits


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


def _freeze_jacobian_artifact(jacobian):
    """Return an immutable ndarray snapshot of a solver-provided Jacobian."""
    if jacobian is None:
        return None
    array = np.array(jacobian, copy=True)
    array.flags.writeable = False
    return array


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


def _compute_covariance_matrix(observed, fitted, jacobian, diagnostics):
    """
    Compute an approximate parameter covariance matrix from the retained Jacobian.

    The approximation follows the local linear least-squares convention:

        covariance = sigma^2 * pinv(J.T @ J)

    where ``sigma^2`` is the residual variance estimate based on the reduced
    chi-square (equivalently ``rss / degrees_of_freedom``).

    Returns
    -------
    ndarray or None
        Immutable covariance matrix when available, otherwise ``None``.
    """
    if jacobian is None:
        return None

    if observed is None or fitted is None:
        return None

    degrees_of_freedom = int(diagnostics.get("degrees_of_freedom", 0))
    sigma2 = diagnostics.get("reduced_chi_square", np.nan)
    n_varying_parameters = int(diagnostics.get("n_varying_parameters", 0))

    if degrees_of_freedom <= 0 or n_varying_parameters <= 0:
        return None

    sigma2 = float(sigma2)
    if not np.isfinite(sigma2):
        return None

    jacobian_array = np.asarray(jacobian, dtype=np.float64)
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

    information = jacobian_used.T @ jacobian_used
    covariance = sigma2 * np.linalg.pinv(information, hermitian=True)
    covariance = 0.5 * (covariance + covariance.T)
    return _freeze_covariance_matrix(covariance)


def _normalize_solver_meta(method, result, warnmess=None):
    """Normalize backend-specific solver information into stable diagnostics."""
    message = getattr(result, "message", warnmess)
    if isinstance(message, (list, tuple)):
        message = " ".join(str(item) for item in message)
    elif message is None:
        message = ""
    else:
        message = str(message)

    method_lower = method.lower()
    if method_lower in ["lm", "trf"]:
        return {
            "success": bool(getattr(result, "success", False)),
            "status": getattr(result, "status", None),
            "message": message,
        }

    if method_lower == "simplex":
        status = int(warnmess) if warnmess is not None else None
        if status == 0:
            success = True
            message = message or "Optimization terminated successfully."
        elif status == 1:
            success = False
            message = message or "Maximum number of function evaluations made."
        elif status == 2:
            success = False
            message = message or "Maximum number of iterations reached."
        else:
            success = False
            message = message or "Simplex optimization ended with an unknown status."
        return {
            "success": success,
            "status": status,
            "message": message,
        }

    if method_lower == "basinhopping":
        lowest = getattr(result, "lowest_optimization_result", None)
        success = getattr(result, "success", None)
        status = getattr(result, "status", None)
        if lowest is not None:
            if success is None:
                success = getattr(lowest, "success", None)
            if status is None:
                status = getattr(lowest, "status", None)
            if not message:
                lowest_message = getattr(lowest, "message", "")
                if isinstance(lowest_message, (list, tuple)):
                    lowest_message = " ".join(str(item) for item in lowest_message)
                message = str(lowest_message)
        if success is None:
            success = bool(status in (0, 1, True)) if status is not None else False
        return {
            "success": bool(success),
            "status": status,
            "message": message,
        }

    return {
        "success": False,
        "status": None,
        "message": message,
    }


def _extract_solver_artifacts(method, result):
    """Retain backend artifacts needed for future uncertainty features."""
    method_lower = method.lower()

    if method_lower in ["lm", "trf"]:
        return {
            "jacobian": _freeze_jacobian_artifact(getattr(result, "jac", None)),
            "jacobian_backend": method_lower,
        }

    return {
        "jacobian": None,
        "jacobian_backend": method_lower,
    }


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
    fp = FitParameters()
    errors = []

    modlabel = None
    common = False
    fixed = False
    reference = False
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
            if modlabel not in fp.models:
                fp.models.append(modlabel)
                if modlabel:
                    models_missing_shape.add(modlabel)
            else:
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        f"Duplicate model label: '{modlabel}'",
                    ),
                )
            common = False
            continue

        if key.startswith("common") or key.startswith("vars"):
            common = True
            modlabel = "common"
            continue

        if key.startswith("shape"):
            if modlabel is None or modlabel == "common":
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        "shape: must appear after a MODEL: declaration",
                    ),
                )
                continue
            models_missing_shape.discard(modlabel)
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
            fp.model[modlabel] = shape
            common = False
            continue

        if modlabel is None and not common:
            errors.append(
                ScriptError(
                    lc,
                    line,
                    "The first definition should be a label for a model"
                    " or a block of variables or constants.",
                ),
            )
            continue

        # parameter prefix --------------------------------------------------
        if key.startswith("*"):
            fixed = True
            reference = False
            key = key[1:].strip()
        elif key.startswith("$"):
            fixed = False
            reference = False
            key = key[1:].strip()
        elif key.startswith(">"):
            fixed = True
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

        # value / bounds ----------------------------------------------------
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
        if mini.strip().lower() in ["none", ""]:
            mini = str(-1.0 / sys.float_info.epsilon)
        if maxi.strip().lower() in ["none", ""]:
            maxi = str(+1.0 / sys.float_info.epsilon)
        if modlabel != "common":
            ks = f"{key}_{modlabel}"
            fp.common[key] = False
        else:
            ks = f"{key}"
            fp.common[key] = True
        fp.reference[ks] = reference
        if not reference:
            val = value.strip()
            try:
                val = eval(str(val))  # noqa: S307
            except Exception as exc:
                errors.append(
                    ScriptError(
                        lc,
                        line,
                        f"Cannot evaluate value for '{ks}': {exc}",
                    ),
                )
                continue
            fp[ks] = val, mini.strip(), maxi.strip(), fixed
        else:
            fp[ks] = value.strip()

    # Post-parse: check that every model has a shape definition
    for label in list(models_missing_shape):
        if label not in fp.model:
            errors.append(
                ScriptError(
                    0,
                    "",
                    f"Model '{label}' has no shape definition",
                ),
            )

    return fp, errors


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
    The current implementation supports only 1D fitting. Multi-dimensional
    datasets are not yet handled by :meth:`fit`.

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

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        warm_start=False,
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
        **kwargs : dict
            Additional keyword arguments.
        """
        # An empty __doc__ must be placed here, else Configurable.__doc__ will appear

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private methods ( overriding abstract methods)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):
        # NMR
        # sequence = kargs.get('sequence', 'ideal_pulse')
        # self.sequence = PulseSequence(type=sequence)

        # for now, we only work with 1D data
        if X.ndim > 1 and all(np.array(X.shape) > 1):
            raise NotImplementedError("Only 1D data are supported for now")

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

        def fun_residuals(params, X):
            global ncalls
            ncalls += 1

            # model
            modeldata = self._get_modeldata(X)[0]

            # baseline is already summed with modeldata[-1]
            mdata = modeldata[-1]  # modelsum

            # important to work with the real component of dataset
            # not the complex number
            data = X.real.squeeze()

            return data - mdata

        def fun_chi2(params, X):  # , *constraints):
            """Return sum((y - x)**2)."""
            global chi2

            # model
            res = fun_residuals(params, X)
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

        fp = self.fp  # starting parameters

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
        if not self.dry:
            fp, fopt, solver_meta, solver_artifacts = _optimize(
                func,
                fp,
                args=(X,),
                maxfun=self.max_fun_calls,
                maxiter=self.max_iter,
                method=self.method,
                constraints=self.constraints,
                callback=callback,
            )

        # Store solver metadata for the result object.
        # Created on every _fit call so it always reflects the last fit.
        self._fit_meta = {
            "cost": float(fopt) if fopt is not None else None,
            "niter": niter,
            "ncalls": ncalls,
            **solver_meta,
        }
        self._fit_config = fit_config
        self._solver_artifacts = solver_artifacts

        # replace the previous script with new fp parameters
        self.script = str(fp)

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
            # the keys must be with lower case
            # and the values must be a models_.usermodel instance
            if not isinstance(value, models_.usermodel):
                usermodel = models_.usermodel
                usermodel.f = staticmethod(value)
                usermodel.args = inspect.getfullargspec(value).args[1:]
            newdict[key.lower()] = usermodel
        return newdict

    @tr.validate("script")
    def _script_validate(self, proposal):
        fp, errors = _validate_script_content(proposal.value, self.usermodels)
        if errors:
            err = errors[0]
            raise ValueError(str(err))
        self.fp = fp
        return proposal.value

    @tr.validate("constraints")
    def _constraints_validate(self, proposal):
        constraints = proposal.value
        if constraints in (None, {}, []):
            return None

        fit_parameters = self.fp
        errors = _validate_constraints_content(constraints, fit_parameters)
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
        _, errors = _validate_script_content(script, self.usermodels)
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

        fit_parameters, script_errors = _validate_script_content(
            script, self.usermodels
        )
        if script_errors:
            return [
                ConstraintError(
                    0,
                    constraints,
                    "Cannot validate constraints because the fitting script is invalid. "
                    f"First script error: {script_errors[0].message}",
                ),
            ]

        return _validate_constraints_content(constraints, fit_parameters)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.default("_script")
    def _script_default(self):
        """Return a default script."""
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

    def _get_modeldata(self, X, exp_idx=1):
        # exp_idx is not used for the moment, but will be necessary for multidataset
        # fitting

        # Prepare parameters
        parameters = self._prepare(self.fp, exp_idx)

        # Get the list of models
        models = self.fp.models
        self._n_components = nbmodels = len(models)

        # Make an array 'modeldata' with the size of the dataset of data
        # which will contains the data produced by the models
        # This name must always be 'modeldata'
        # which will be returned to the main program.

        expedata = X.real.squeeze()

        # we need to calculate the model with the full unmasked coordinates

        if expedata.ndim > 1:
            # nD data
            raise NotImplementedError("Fit not implemented for nD data yet!")

        # we need to keep track of the x axis before masking
        axis, dim = self._X.get_axis(-1)
        _xaxis = self._X_coordset[dim].data

        x = _xaxis
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
            calc = getmodel(
                x,
                modelname=model,
                par=parameters,
                amplitude_mode=self.amplitude_mode,
                usermodels=self.usermodels,
            )
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
        if np.any(self._X_mask):
            masked_columns = np.all(self._X_mask, axis=-2)
            modeldata = modeldata[:, ~masked_columns]
            x = x[~masked_columns]

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
        keyword = r"\b([a-z]+[0-9]*)\b"  # match a whole word
        pat = re.compile(keyword)
        mo = pat.findall(str(expr))
        for kw in mo:
            if kw in param:
                expr = expr.replace(kw, str(param[kw]))
            elif kw in np.__dict__:  # check if it is a recognized math function
                expr = expr.replace(kw, f"np.{kw}")
        return expr

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
        sE = sum(expe)
        sF = sum(calc)
        sFI = sum(xi * calc)
        sFd = sum(calc * calc)
        sI = sum(xi)
        sEF = sum(expe * calc)
        sEI = sum(xi * expe)
        sId = sum(xi**2)

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
        return super().fit(X, Y=None)

    @property
    def jacobian(self):
        """
        Return the retained raw solver Jacobian when the backend provides one.

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
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before accessing the result",
            )

        # NOTE: a new FitResult is created on every access.
        # Caching is deliberately deferred to keep the implementation
        # simple and aligned with the PCA / SVD result behaviour.

        fitted = self.predict()
        residuals, fit_diagnostics = _compute_fit_diagnostics(
            self._X,
            fitted,
            getattr(self, "_fit_meta", {}),
            self.fp,
        )
        covariance = _compute_covariance_matrix(
            self._X,
            fitted,
            self.jacobian,
            fit_diagnostics,
        )
        parameter_values = _extract_varying_parameter_values(self.fp)
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

        return FitResult(
            estimator="Optimize",
            parameters=fit_config.copy(),
            outputs={
                "fitted": fitted,
                "components": self.components,
                "residuals": residuals,
            },
            diagnostics=fit_diagnostics,
            covariance=covariance,
            parameter_values=parameter_values,
        )


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

    def restore_external(fp, p, keys):
        # restore external parameters
        for key in list(fp.keys()):
            keysp = key.split("_")
            if keysp[0] in fp.expvars:
                ps = []
                for i in range(fp.expnumber):
                    ks = f"{key}_exp{i}"
                    if ks not in keys:
                        break
                    k = keys.index(ks)
                    ps.append(p[k])
                if ps:
                    fp.to_external(key, ps)
            else:
                if key not in keys:
                    continue
                k = keys.index(key)
                fp.to_external(key, p[k])
        return fp

    def internal_func(p, dat, fp, keys, *args):
        fp = restore_external(fp, p, keys)
        return func(fp, dat)

    def internal_callback(*args):
        if callback is None:
            return None
        return callback(*args)

    if not isinstance(fp0, FitParameters):
        raise TypeError("fp0 is not of FitParameter type")

    # make internal parameters
    par = []
    keys = []

    for key in sorted(fp0.keys()):
        if not fp0.fixed[key]:
            # we make internal parameters in case of bounding
            # We also take care of the multiple experiments
            keysp = key.split("_")[0]
            if keysp in fp0.expvars:
                for i in range(fp0.expnumber):
                    par.append(fp0.to_internal(key, i))
                    keys.append(f"{key}_exp{i}")
            else:
                par.append(fp0.to_internal(key))
                keys.append(key)

    args = list(args)
    args.append(fp0)
    args.append(keys)
    if constraints:
        args.append(constraints)

    if not maxfun:
        maxfun = 4 * maxiter

    if method in ["leastsq", "least_squares"]:
        method = "lm" if len(fp0) < 10 else "trf"

    if method.lower() in ["lm", "trf"]:
        result = optimize.least_squares(
            internal_func,
            par,
            args=tuple(args),
            method=method.lower(),
        )
        res, fopt, warnmess = result.x, result.cost, result.message
        solver_meta = _normalize_solver_meta(method, result, warnmess)
        solver_artifacts = _extract_solver_artifacts(method, result)

    elif method.lower() == "simplex":
        result = optimize.fmin(
            internal_func,
            par,
            args=tuple(args),
            maxfun=maxfun,
            maxiter=maxiter,
            ftol=ftol,
            xtol=xtol,
            full_output=True,
            disp=False,
            callback=internal_callback,
        )
        res, fopt, _, _, warnmess = result
        solver_meta = _normalize_solver_meta(method, None, warnmess)
        solver_artifacts = _extract_solver_artifacts(method, None)

    elif method.lower() == "basinhopping":
        result = optimize.basinhopping(
            internal_func,
            par,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={"args": tuple(args)},
            take_step=None,
            accept_test=None,
            callback=internal_callback,
            interval=50,
            disp=False,
            niter_success=None,
        )

        # fmin(func, par, args=args, maxfun=maxfun, maxiter=maxiter, ftol=ftol, xtol=xtol,
        #                                                full_output=True, disp=False, callback=callback)
        res, fopt, warnmess = result.x, result.fun, result.message
        solver_meta = _normalize_solver_meta(method, result, warnmess)
        solver_artifacts = _extract_solver_artifacts(method, result)

    elif method == "XXXX":
        raise NotImplementedError(f"method: {method}")
        # TODO: implement other algorithms
    else:
        raise NotImplementedError(f"method: {method}")

    # restore the external parameter
    fpe = restore_external(fp0, res, keys)
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

    x = np.array(x, dtype=np.float64)
    if y is not None:
        y = np.array(y, dtype=np.float64)

    val = a.f(x, *args, **kwargs) if y is None else a.f(x, y, *args, **kwargs)

    # Return amplitude or area ? return calc is scaled by area by default
    if amplitude_mode.lower() == "height":
        # in this case ampl parameter is the height, so we need to rescale
        # calc
        ampl = args[0]
        val = ampl * val / val.max()
    return val
