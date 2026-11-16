# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""Core IRIS tests — moved from core to plugin."""

import warnings
from os import environ
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import osqp
import pytest

import spectrochempy as scp
from spectrochempy.utils import docutils as chd

from spectrochempy_iris import IRIS
from spectrochempy_iris import IrisKernel
from spectrochempy_iris._core import _Smat
from spectrochempy_iris._core import _solve_osqp_lambda


@pytest.mark.skipif(not hasattr(osqp, "default_algebra"), reason="requires modern OSQP")
def test_iris_parallel_fixed_grid_matches_sequential_fit(monkeypatch):
    monkeypatch.setattr(osqp, "default_algebra", lambda: "builtin")
    observations = scp.Coord(np.arange(5))
    channels = scp.Coord(np.arange(6))
    grid = scp.Coord(np.arange(4))
    spectra = scp.NDDataset(
        np.arange(30.0).reshape(5, 6), coordset=[observations, channels]
    )
    kernel = scp.NDDataset(np.eye(5, 4), coordset=[observations, grid])
    with warnings.catch_warnings(record=True) as serial_warnings:
        warnings.simplefilter("always", PendingDeprecationWarning)
        serial = IRIS(reg_par=[-4, -2, 3], workers=1).fit(spectra, kernel)
    with warnings.catch_warnings(record=True) as parallel_warnings:
        warnings.simplefilter("always", PendingDeprecationWarning)
        parallel = IRIS(reg_par=[-4, -2, 3], workers=2, log_level="INFO").fit(
            spectra, kernel
        )
    assert "Starting OSQP workers for a 4-point kernel grid" in parallel.log
    serial_messages = [
        str(item.message)
        for item in serial_warnings
        if item.category is PendingDeprecationWarning
    ]
    assert [
        str(item.message)
        for item in parallel_warnings
        if item.category is PendingDeprecationWarning
    ] == serial_messages
    if serial_messages:
        with (
            warnings.catch_warnings(),
            pytest.raises(PendingDeprecationWarning),
        ):
            warnings.simplefilter("error", PendingDeprecationWarning)
            IRIS(reg_par=[-4, -2, 3], workers=2).fit(spectra, kernel)

    for actual, reference in (
        (parallel.f, serial.f),
        (parallel.lambdas, serial.lambdas),
        (parallel.RSS, serial.RSS),
        (parallel.SM, serial.SM),
    ):
        actual = np.asarray(actual.data)
        reference = np.asarray(reference.data)
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype
        assert actual.tobytes() == reference.tobytes()


@pytest.mark.skipif(not hasattr(osqp, "default_algebra"), reason="requires modern OSQP")
@pytest.mark.parametrize(
    ("points", "reg_par", "warning_is_error"),
    [
        (np.arange(4.0), [-4, 3, 3], False),
        (np.arange(4) * 1e-100, [9, 10, 3], True),
    ],
    ids=["negative-penalty", "overflowing-penalty"],
)
def test_iris_workers_keep_invalid_grid_failures_sequential(
    monkeypatch, capfd, points, reg_par, warning_is_error
):
    monkeypatch.setattr(osqp, "default_algebra", lambda: "builtin")
    if not warning_is_error:
        monkeypatch.setitem(IRIS._fit.__globals__, "_Smat", lambda q: -_Smat(q))
    observations = scp.Coord(np.arange(5))
    channels = scp.Coord(np.arange(6))
    grid = scp.Coord(points, rounding=False)
    spectra = scp.NDDataset(
        np.arange(30.0).reshape(5, 6), coordset=[observations, channels]
    )
    kernel = scp.NDDataset(np.eye(5, 4), coordset=[observations, grid])
    native_output = []

    for workers in (1, 2):
        with np.errstate(over="warn", invalid="warn"), warnings.catch_warnings():
            if warning_is_error:
                warnings.filterwarnings(
                    "error", "overflow encountered in multiply", RuntimeWarning
                )
            error = RuntimeWarning if warning_is_error else osqp.OSQPException
            with pytest.raises(error):
                IRIS(reg_par=reg_par, workers=workers).fit(spectra, kernel)
        native_output.append(capfd.readouterr())

    assert native_output[0] == native_output[1]


@pytest.mark.skipif(not hasattr(osqp, "default_algebra"), reason="requires modern OSQP")
@pytest.mark.parametrize("workers", [1, 2])
def test_iris_reports_penalty_underflow_for_each_channel(monkeypatch, workers):
    monkeypatch.setattr(osqp, "default_algebra", lambda: "builtin")
    observations = scp.Coord(np.arange(5))
    channels = scp.Coord(np.arange(6))
    grid = scp.Coord(np.arange(4) * 1e100, rounding=False)
    spectra = scp.NDDataset(
        np.arange(30.0).reshape(5, 6), coordset=[observations, channels]
    )
    kernel = scp.NDDataset(np.eye(5, 4), coordset=[observations, grid])

    with np.errstate(under="warn"), warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", RuntimeWarning)
        IRIS(reg_par=[-25, -24, 2], workers=workers).fit(spectra, kernel)

    underflow_warnings = sum(
        str(item.message) == "underflow encountered in multiply" for item in emitted
    )
    assert underflow_warnings == 12

    with np.errstate(under="warn"), warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            "underflow encountered in multiply",
            RuntimeWarning,
            module=r"^spectrochempy_iris\._core$",
        )
        with pytest.raises(RuntimeWarning, match="underflow encountered in multiply"):
            IRIS(reg_par=[-25, -24, 2], workers=workers).fit(spectra, kernel)


def test_osqp_lambda_reports_penalty_overflow_for_each_channel():
    grid = np.arange(4) * (1.55e306 ** (-1 / 3))
    problem = (
        np.eye(4),
        np.ones((6, 4)),
        np.eye(4),
        np.zeros(4),
        np.full(4, np.inf),
        _Smat(grid),
    )
    solver = Mock()
    solver.solve.return_value.x = np.ones(4)
    with (
        np.errstate(over="warn"),
        pytest.warns(
            RuntimeWarning, match="overflow encountered in multiply"
        ) as emitted,
    ):
        _solve_osqp_lambda(10.0, problem, lambda: solver)

    assert solver.solve.call_count == len(emitted) == 6


@pytest.mark.parametrize(
    ("points", "expected_sm"),
    [
        (np.arange(4, dtype=np.longdouble), 24.0),
        (np.arange(4) * (1 + 0.1j), 24.0 / abs(1 + 0.1j) ** 3),
        (np.array([0.0, 1.0, 2.0, 0.0]), np.nan),
        (np.arange(3.0, -1.0, -1.0), 24.0),
    ],
    ids=["longdouble", "complex", "coincident-endpoints", "descending"],
)
def test_iris_wide_dataset_sharpness_for_unusual_grids(points, expected_sm):
    observations = scp.Coord(np.arange(5))
    channels = scp.Coord(np.arange(6))
    grid = scp.Coord(points, dtype=points.dtype)
    spectra = scp.NDDataset(np.ones((5, 6)), coordset=[observations, channels])
    kernel = scp.NDDataset(np.eye(5, 4), coordset=[observations, grid])

    with np.errstate(divide="ignore", invalid="ignore"):
        model = IRIS(reg_par=None).fit(spectra, kernel)

    np.testing.assert_array_equal(model.f.data, np.ones((1, 4, 6)))
    np.testing.assert_allclose(model.SM.data, [expected_sm])


@pytest.mark.parametrize(
    ("grid_step", "amplitude"),
    [(5.8e107, 1e160), (5e-103, 10.0), (1.0, 1e-81)],
    ids=["subnormal-penalty", "overflowing-sharpness", "underflowing-sharpness"],
)
def test_iris_wide_dataset_sharpness_at_extreme_scales(grid_step, amplitude):
    observations = scp.Coord(np.arange(5))
    channels = scp.Coord(np.arange(6))
    grid = scp.Coord(np.arange(4) * grid_step, rounding=False)
    data = np.zeros((5, 6))
    data[:4] = amplitude
    spectra = scp.NDDataset(data, coordset=[observations, channels])
    kernel = scp.NDDataset(np.eye(5, 4), coordset=[observations, grid])

    with np.errstate(over="ignore", invalid="ignore"):
        model = IRIS(reg_par=None).fit(spectra, kernel)
        distribution = model.f.data[0]
        expected_sm = np.linalg.norm(distribution.T @ _Smat(grid.data) @ distribution)
    np.testing.assert_allclose(model.SM.data, [expected_sm], rtol=1e-12, atol=0)


def test_iris_kernel_and_penalty_are_invariant_to_grid_reversal():
    spectra = scp.NDDataset(
        np.ones((5, 2)),
        coordset=[scp.Coord(np.linspace(0.1, 1, 5)), scp.Coord(np.arange(2))],
    )
    ascending = IrisKernel(spectra, "langmuir", q=[-4, 0, 5]).kernel
    descending = IrisKernel(spectra, "langmuir", q=[0, -4, 5]).kernel

    np.testing.assert_array_equal(descending.x.data[::-1], ascending.x.data)
    np.testing.assert_array_equal(descending.data[:, ::-1], ascending.data)
    penalty = _Smat(descending.x.data)
    np.testing.assert_array_equal(penalty[::-1, ::-1], _Smat(ascending.x.data))
    assert np.all(np.linalg.eigvalsh(penalty) > 0)


def test_regularized_nnls_satisfies_optimality_on_both_grid_orders():
    pressure = np.linspace(0.1, 1, 5)
    spectra = scp.NDDataset(
        np.column_stack((pressure, -pressure)),
        coordset=[scp.Coord(pressure), scp.Coord(np.arange(2))],
    )
    ascending = IrisKernel(spectra, "langmuir", q=[-4, 0, 5]).kernel
    descending = IrisKernel(spectra, "langmuir", q=[0, -4, 5]).kernel
    options = {"qpsolver": "nnls", "reg_par": [-3, -1, 3]}
    forward = IRIS(**options).fit(spectra, ascending)
    reverse = IRIS(**options).fit(spectra, descending)
    np.testing.assert_allclose(forward.f.data, reverse.f.data[:, ::-1], atol=1e-12)

    kernel = ascending.data
    penalty = _Smat(ascending.x.data)
    for strength, coefficients in zip(
        forward.lambdas.data, forward.f.data, strict=True
    ):
        gradient = kernel.T @ (kernel @ coefficients - spectra.data)
        gradient += strength * (penalty @ coefficients)
        assert np.all(coefficients >= 0)
        assert np.all(gradient >= -1e-12)
        np.testing.assert_allclose(gradient[coefficients > 0], 0, atol=1e-12)


def test_adaptive_nnls_preserves_search_bounds_on_refit():
    pressure = np.linspace(0.1, 1, 5)
    spectra = scp.NDDataset(
        np.column_stack((pressure, -pressure)),
        coordset=[scp.Coord(pressure), scp.Coord(np.arange(2))],
    )
    kernel = IrisKernel(spectra, "langmuir", q=[-4, 0, 5])
    iris = IRIS(qpsolver="nnls", reg_par=[-4, -2]).fit(spectra, kernel)
    strengths = iris.lambdas.data.copy()
    np.testing.assert_allclose(strengths[[0, -1]], [1e-4, 1e-2], atol=0)

    iris.fit(spectra, kernel)
    np.testing.assert_array_equal(iris.lambdas.data, strengths)


def test_adaptive_nnls_rejects_zero_sharpness():
    pressure = np.linspace(0.1, 1, 5)
    spectra = scp.NDDataset(
        -pressure[:, None],
        coordset=[scp.Coord(pressure), scp.Coord([0])],
    )
    kernel = IrisKernel(spectra, "langmuir", q=[-4, 0, 5])
    with pytest.raises(ValueError, match="finite, positive residuals and sharpness"):
        IRIS(qpsolver="nnls", reg_par=[-4, -2]).fit(spectra, kernel)


def test_IRIS_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy_iris._core"
    chd.check_docstrings(
        module,
        obj=IRIS,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


def test_analysis_iris_kernel():
    X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2105.0:1995.0]
    p = [
        0.00300,
        0.00400,
        0.00900,
        0.01400,
        0.02100,
        0.02600,
        0.03600,
        0.05100,
        0.09300,
        0.15000,
        0.20300,
        0.30000,
        0.40400,
        0.50300,
        0.60200,
        0.70200,
        0.80100,
        0.90500,
        1.00400,
    ]
    X.coordset.update(y=scp.Coord(p, title="pressure", units="torr"))
    # Using the `update` method is mandatory because it will preserve the name.
    # Indeed, setting using X.coordset[0] = Coord(...) fails unless name is specified: Coord(..., name='y')

    # test the docstring example
    k1 = IrisKernel(X, K="langmuir", p=np.linspace(0, 1, 19), q=np.logspace(-10, 1, 10))
    assert str(k1.kernel) == "NDDataset: [float64] unitless (shape: (y:19, x:10))"

    # without specifying p
    k2 = IrisKernel(X, "langmuir", q=np.logspace(-10, 1, 10))
    assert str(k2.kernel) == "NDDataset: [float64] unitless (shape: (y:19, x:10))"

    # using a function F
    F = lambda p, q: np.exp(-q) * p[:, None] / (1 + np.exp(-q) * p[:, None])
    k3 = IrisKernel(X, F, p=np.linspace(0, 1, 19), q=np.logspace(-10, 1, 10))
    assert np.all(k3.kernel.data == k1.kernel.data)

    # p and q can also be passed as coordinates:
    p0 = scp.Coord(np.linspace(0, 1, 19), name="pressure", title="p", units="torr")
    q0 = scp.Coord(
        np.logspace(-10, 1, 10),
        name="reduced adsorption energy",
        title=r"$\Delta_{ads}G^{0}/RT$",
        units="",
    )
    k4 = IrisKernel(X, "langmuir", p=p0, q=q0)
    assert np.all(k4.kernel.data == k1.kernel.data)
    pass


def test_IRIS():
    # Define the dataset to fit with the IRIS model
    # we change timestamp coordinates with the corresponding measured pressures.
    X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2105.0:1995.0]
    # X[:, 2000.0:2004.0] = scp.MASKED  # Test with a mask
    p = [
        0.00300,
        0.00400,
        0.00900,
        0.01400,
        0.02100,
        0.02600,
        0.03600,
        0.05100,
        0.09300,
        0.15000,
        0.20300,
        0.30000,
        0.40400,
        0.50300,
        0.60200,
        0.70200,
        0.80100,
        0.90500,
        1.00400,
    ]

    X.coordset.update(y=scp.Coord(p, title="pressure", units="torr"))
    # Using the `update` method is mandatory because it will preserve the name.
    # Indeed, setting using X.coordset[0] = Coord(...) fails unless name is specified: Coord(..., name='y')

    # Define a kernel
    q = [-8, -1, 10]
    K = IrisKernel(X, "langmuir", q=q)

    # Create an IRIS object with no regularization parameters
    iris1 = IRIS()

    # Fit the IRIS model with a langmuir kernel
    iris1.fit(X, K)

    f1 = iris1.f.copy()
    assert f1.shape == (1, q[2], X.shape[1])
    assert f1.history_entries[-1]["operation"] is None
    assert f1.history_entries[-1]["message"] == (
        f"Computed 2D IRIS distribution from {X.name}"
    )

    X_hat = iris1.inverse_transform()
    assert X_hat.squeeze().shape == X.shape

    # test with callable
    def ker(p, q):
        return np.exp(-q) * p[:, None] / (1 + np.exp(-q) * p[:, None])

    K = IrisKernel(X, ker, q=iris1.q)
    iris1.fit(X, K)
    f2 = iris1.f.copy()
    assert f2.shape == (1, q[2], X.shape[1])
    assert f1 == f2
    K1 = iris1.K  # keep it for later

    # no regularization, ca, 1D, p and q passed as a ndarrays
    X2 = X[:, 2110.0]
    K1D = IrisKernel(X2, "ca", p=p, q=iris1.K.x.data)
    iris1.fit(X2, K1D)
    f1d = iris1.f.copy()
    assert f1d.shape == (1, q[2], 1)

    # manual regularization, keeping the previous langmuir kernel
    reg_par = [-4, -2, 3]
    iris2 = IRIS(reg_par=reg_par)
    iris2.fit(X, K1)
    f2 = iris2.f
    assert f2.shape == (reg_par[2], q[2], X.shape[1])
    sharpness_matrix = _Smat(iris2.q.data)
    np.testing.assert_allclose(
        iris2.SM.data,
        [np.linalg.norm(fi.T @ sharpness_matrix @ fi) for fi in f2.data],
        rtol=1e-11,
    )

    iris2.plotlcurve()
    _ = iris2.f[-2].plot(method="contour")
    _ = iris2.plot_merit(-2)

    # with automated search, keeping the previous kernel
    reg_par = [-4, -2]
    iris3 = IRIS(reg_par=reg_par)
    iris3.fit(X, K1)
    f3 = iris3.f
    assert (f3.shape[1], f3.shape[2]) == (q[2], X.shape[1])
    with patch.object(np.linalg, "cholesky", side_effect=np.linalg.LinAlgError):
        legacy_metric = IRIS(reg_par=reg_par).fit(X, K1)
    np.testing.assert_array_equal(f3.data, legacy_metric.f.data)
    np.testing.assert_array_equal(iris3.RSS.data, legacy_metric.RSS.data)
    np.testing.assert_array_equal(iris3.lambdas.data, legacy_metric.lambdas.data)
    np.testing.assert_allclose(iris3.SM.data, legacy_metric.SM.data, rtol=1e-11)

    iris_nnls = IRIS(qpsolver="nnls", reg_par=reg_par).fit(X, K1)
    assert iris_nnls.f.shape[1:] == f3.shape[1:]
    assert np.all(iris_nnls.f.data >= 0)
    assert iris_nnls.f.shape[0] > 4
    distributions = iris_nnls.f.data
    residuals = X.data - K1.data @ distributions
    penalty = _Smat(K1.x.data)
    np.testing.assert_allclose(iris_nnls.RSS, np.sum(residuals**2, axis=(1, 2)))
    np.testing.assert_allclose(
        iris_nnls.SM,
        np.linalg.norm(
            distributions.transpose(0, 2, 1) @ penalty @ distributions, axis=(1, 2)
        ),
    )

    # test other kernels (beware: it is chemical non-sense !)
    q = [5, 1, 10]
    iris4 = IRIS()
    K4 = IrisKernel(X, "diffusion", q=q)
    iris4.fit(X, K4)
    f4 = iris4.f
    assert f4.shape == (1, q[2], X.shape[1])

    q = [5, 1, 10]
    K5 = IrisKernel(X, "product-first-order", q=q)
    iris4.fit(X, K5)
    f5 = iris4.f
    assert f5.shape == (1, q[2], X.shape[1])

    q = [5, 1, 10]
    K6 = IrisKernel(X[::-1], "reactant-first-order", q=q)
    iris4.fit(X, K6)
    f6 = iris4.f
    assert f6.shape == (1, q[2], X.shape[1])

    q = [5, 1, 10]
    K7 = IrisKernel(X[::-1], "stejskal-tanner", q=q)
    iris4.fit(X, K7)
    f7 = iris4.f
    assert f7.shape == (1, q[2], X.shape[1])
