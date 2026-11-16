"""OSQP solves for independent IRIS regularization parameters."""

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from inspect import getmodule
from multiprocessing import get_context

import numpy as np
from scipy import sparse


def _penalized_matrix(lambda_, P0, S):
    return sparse.triu(sparse.csc_matrix(P0 + 2 * lambda_ * S), format="csc")


def solve_osqp_lambda(lambda_, problem, make_solver, legacy_warning=None):
    P0, q, A, lo, up, S = problem
    cached_matrix = None
    if np.geterr()["under"] == "ignore":
        cached_matrix = _penalized_matrix(lambda_, P0, S)
    distribution = np.empty((P0.shape[0], len(q)))
    for j, linear in enumerate(q):
        P = cached_matrix
        if P is None:
            P = _penalized_matrix(lambda_, P0, S)
        solver = make_solver()
        if legacy_warning is None:
            solver.setup(P, linear.squeeze(), A, lo, up, alpha=1.0, verbose=False)
        else:
            solver.setup(P, linear.squeeze(), A, lo, alpha=1.0, verbose=False)
            legacy_warning()
        distribution[:, j] = solver.solve().x
    return distribution


_solve_in_worker = None


def _initialize_worker(problem, make_solver, error_policy):
    global _solve_in_worker
    np.seterr(**error_policy)
    _solve_in_worker = partial(
        solve_osqp_lambda, problem=problem, make_solver=make_solver
    )


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


def solve_osqp_grid(lambdas, problem, make_solver, workers):
    workers = min(workers, len(lambdas))
    if sys.platform == "win32":
        workers = min(workers, 61)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(problem, make_solver, np.geterr()),
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
