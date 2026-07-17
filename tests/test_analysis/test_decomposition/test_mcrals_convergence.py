"""Tests for the public MCRALS convergence criteria."""

import inspect

import numpy as np
import pytest

from spectrochempy.analysis.decomposition.mcrals import MCRALS


def _exact_rank_one_problem():
    C = np.array([[1.0], [2.0], [3.0]])
    St = np.array([[2.0, 1.0, 4.0]])
    return C @ St, 1.1 * C


def test_convergence_tolerance_defaults_signature_and_docstring():
    mcr = MCRALS()
    signature = inspect.signature(MCRALS)

    assert mcr.tol_residual_change == 1.0e-3
    assert mcr.tol_reconstruction_error is None
    assert mcr.tol_profile_change is None
    assert signature.parameters["tol_residual_change"].default == 1.0e-3
    assert signature.parameters["tol_reconstruction_error"].default is None
    assert signature.parameters["tol_profile_change"].default is None
    assert "tol" not in signature.parameters
    for name in (
        "tol_residual_change",
        "tol_reconstruction_error",
        "tol_profile_change",
    ):
        assert name in MCRALS.__doc__


def test_legacy_tol_is_converted_from_percent():
    with pytest.warns(FutureWarning, match="tol_residual_change"):
        mcr = MCRALS(tol=0.25)

    assert mcr.tol == 0.25
    assert mcr.tol_residual_change == 2.5e-3

    with pytest.warns(FutureWarning, match="tol_residual_change"):
        mcr.tol = 2.0
    assert mcr.tol_residual_change == 2.0e-2


def test_legacy_and_modern_residual_tolerances_are_mutually_exclusive():
    with pytest.raises(ValueError, match="Cannot specify both"):
        MCRALS(tol=0.1, tol_residual_change=1.0e-3)


def test_profile_change_ignores_reciprocal_scale_and_sign():
    C = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    St = np.array([[2.0, 1.0], [4.0, 3.0]])
    scales = np.array([-2.0, 5.0])

    changed_C = C * scales
    changed_St = St / scales[:, None]

    assert MCRALS._relative_profile_change(C, St, changed_C, changed_St) < 1.0e-14


def test_reconstruction_error_can_stop_fit_and_is_reported():
    X, C0 = _exact_rank_one_problem()
    mcr = MCRALS(
        constraints=[],
        tol_residual_change=None,
        tol_reconstruction_error=1.0e-12,
        max_iter=5,
    )

    mcr.fit(X, C0)
    diagnostics = mcr.result.diagnostics

    assert diagnostics["converged"]
    assert diagnostics["convergence_reason"] == "reconstruction_error"
    assert diagnostics["reconstruction_error"] < 1.0e-12
    assert np.isfinite(diagnostics["residual_change"])
    assert "profile_change" in diagnostics


def test_info_log_uses_relative_stopping_diagnostics():
    X, C0 = _exact_rank_one_problem()
    mcr = MCRALS(
        constraints=[],
        tol_residual_change=None,
        tol_reconstruction_error=1.0e-12,
        max_iter=5,
        log_level="INFO",
    )

    mcr.fit(X, C0)

    assert "reconstruction_error" in mcr.log
    assert "residual_change" in mcr.log
    assert "profile_change" in mcr.log
    assert "trend" in mcr.log
    assert "RSE / PCA" not in mcr.log
    assert "%change" not in mcr.log
    assert "Converged on reconstruction_error:" in mcr.log
    assert "tol_reconstruction_error=1.000000e-12" in mcr.log


def test_profile_change_can_stop_fit():
    X, C0 = _exact_rank_one_problem()
    mcr = MCRALS(
        constraints=[],
        tol_residual_change=None,
        tol_profile_change=1.0e-12,
        max_iter=5,
    )

    mcr.fit(X, C0)

    assert mcr.result.diagnostics["converged"]
    assert mcr.result.diagnostics["convergence_reason"] == "profile_change"


def test_all_convergence_criteria_cannot_be_disabled():
    X, C0 = _exact_rank_one_problem()
    mcr = MCRALS(
        constraints=[],
        tol_residual_change=None,
        tol_reconstruction_error=None,
        tol_profile_change=None,
    )

    with pytest.raises(ValueError, match="At least one"):
        mcr.fit(X, C0)
