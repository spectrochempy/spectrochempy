# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Numerical fitting backends independent of model definitions and spectral data.

Clients prepare numeric objectives, parameter vectors and solver coordinates.
Raw solver results and backend metadata are returned without interpreting
physical covariance or residual statistics.
"""

import numpy as np
from scipy import optimize


def _least_squares(objective, initial, **options):
    """Run a numeric least-squares problem with the caller's solver options."""
    return optimize.least_squares(objective, initial, **options)


def _run_optimizer(
    objective,
    initial,
    *,
    method,
    args=(),
    maxfun=None,
    maxiter=1000,
    ftol=1e-8,
    xtol=1e-8,
    callback=None,
):
    """Run the historical solver configurations against a numeric objective."""
    method_lower = method.lower()
    if method_lower in ["lm", "trf"]:
        return _least_squares(objective, initial, args=args, method=method_lower)
    if method_lower == "simplex":
        return optimize.fmin(
            objective,
            initial,
            args=args,
            maxfun=maxfun,
            maxiter=maxiter,
            ftol=ftol,
            xtol=xtol,
            full_output=True,
            disp=False,
            callback=callback,
        )
    if method_lower == "basinhopping":
        return optimize.basinhopping(
            objective,
            initial,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={"args": args},
            take_step=None,
            accept_test=None,
            callback=callback,
            interval=50,
            disp=False,
            niter_success=None,
        )
    raise NotImplementedError(f"method: {method}")


def _freeze_jacobian_artifact(jacobian):
    """Return an immutable ndarray snapshot of a solver-provided Jacobian."""
    if jacobian is None:
        return None
    array = np.array(jacobian, copy=True)
    array.flags.writeable = False
    return array


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
