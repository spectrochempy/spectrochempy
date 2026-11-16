"""OSQP solves for independent IRIS regularization parameters."""

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from inspect import getmodule
from multiprocessing import get_context

import numpy as np

_solve_in_worker = None


def _initialize_worker(solve_lambda, error_policy):
    global _solve_in_worker
    np.seterr(**error_policy)
    _solve_in_worker = solve_lambda


def _solve_worker_lambda(lambda_):
    distribution = error = None
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        try:
            distribution = _solve_in_worker(lambda_)
        except Exception as exc:
            error = exc
    return (
        distribution,
        error,
        [(item.message, item.category, item.filename, item.lineno) for item in emitted],
    )


def solve_osqp_grid(lambdas, solve_lambda, workers):
    workers = min(workers, len(lambdas))
    if sys.platform == "win32":
        workers = min(workers, 61)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(solve_lambda, np.geterr()),
    ) as pool:
        warning_modules = {}
        for distribution, error, emitted in pool.map(_solve_worker_lambda, lambdas):
            for message, category, filename, lineno in emitted:
                if filename not in warning_modules:
                    module = getmodule(None, filename)
                    warning_modules[filename] = (
                        {}
                        if module is None
                        else {
                            "module": module.__name__,
                            "registry": vars(module).setdefault(
                                "__warningregistry__", {}
                            ),
                        }
                    )
                warnings.warn_explicit(
                    message, category, filename, lineno, **warning_modules[filename]
                )
            if error is not None:
                raise error
            yield distribution
