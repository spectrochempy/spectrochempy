"""Numerical clients of the private backend without spectral model definitions."""

import numpy as np
import pytest

from spectrochempy.analysis.curvefitting._fit_backend import _least_squares
from spectrochempy.analysis.curvefitting._fit_backend import _run_optimizer


@pytest.fixture(autouse=True)
def preserve_numpy_random_state():
    """Keep stochastic backend tests from changing other tests' random streams."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("method", ["lm", "trf", "simplex", "basinhopping"])
def test_numeric_objective_and_method_specific_callbacks(method):
    target = np.array([1.7, -0.3])
    evaluations = []
    callbacks = []

    def objective(vector, target):
        evaluations.append(vector.copy())
        residual = vector - target
        return residual if method in ("lm", "trf") else float(residual @ residual)

    def callback(*values):
        callbacks.append(values)

    result = _run_optimizer(
        objective,
        [0.0, 0.0],
        method=method,
        args=(target,),
        maxfun=2000,
        maxiter=1000,
        callback=callback,
    )
    if method == "simplex":
        vector, cost, _, _, status = result
        assert status == 0
        assert cost < 1e-12
    else:
        vector = result.x
        assert result.success
    np.testing.assert_allclose(vector, target, atol=1e-6)
    assert evaluations
    if method in ("simplex", "basinhopping"):
        assert callbacks


def test_bounded_kinetic_residual_retains_native_solver_artifacts():
    time = np.linspace(0, 5, 21)
    observed = np.exp(-0.7 * time)

    def residual(vector, time, observed):
        return (np.exp(-vector[0] * time) - observed) / 0.02

    result = _least_squares(
        residual,
        [0.2],
        args=(time, observed),
        method="trf",
        bounds=(0.01, 2.0),
        jac="3-point",
        max_nfev=200,
    )
    assert result.success
    assert result.x[0] == pytest.approx(0.7)
    assert result.cost < 1e-12
    assert result.jac.shape == (time.size, 1)
    assert result.active_mask[0] == 0


def test_unknown_method_is_rejected():
    with pytest.raises(NotImplementedError, match="method: unsupported"):
        _run_optimizer(lambda vector: vector, [0.0], method="unsupported")


@pytest.mark.parametrize(
    "target, initial", [(0.0, [0.0, 1.0]), (1.0, [1.0, 1.0]), (4.0, [2.0, 2.0])]
)
def test_native_exact_rank_deficient_residual_keeps_budget_status(target, initial):
    import warnings

    from spectrochempy.analysis.curvefitting._fit_backend import _native_least_squares

    x = np.linspace(0, 1, 21)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = _native_least_squares(
            lambda p: (p[0] * p[1] - target) * x,
            initial,
            (-np.inf, np.inf),
            max_nfev=100,
        )
    assert not emitted
    assert np.all(result.fun == 0)
    assert result.status in (0, 3)
    assert result.success == (result.status == 3)
    assert result.nfev <= 100


def test_native_does_not_hide_user_model_warnings():
    import warnings

    from spectrochempy.analysis.curvefitting._fit_backend import _native_least_squares

    def residual(p):
        warnings.warn("model warning", RuntimeWarning, stacklevel=1)
        return p - 1

    with pytest.warns(RuntimeWarning, match="model warning"):
        _native_least_squares(residual, [0.0], (-np.inf, np.inf), max_nfev=2)
