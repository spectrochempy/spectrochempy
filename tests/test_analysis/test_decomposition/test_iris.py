# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
from os import environ

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.utils import docstrings as chd


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_IRIS_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.iris"
    chd.check_docstrings(
        module,
        obj=scp.IRIS,
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
    k1 = scp.IrisKernel(
        X, K="langmuir", p=np.linspace(0, 1, 19), q=np.logspace(-10, 1, 10)
    )
    assert str(k1.kernel) == "NDDataset: [float64] unitless (shape: (y:19, x:10))"

    # without specifying p
    k2 = scp.IrisKernel(X, "langmuir", q=np.logspace(-10, 1, 10))
    assert str(k2.kernel) == "NDDataset: [float64] unitless (shape: (y:19, x:10))"

    # using a function F
    F = lambda p, q: np.exp(-q) * p[:, None] / (1 + np.exp(-q) * p[:, None])
    k3 = scp.IrisKernel(X, F, p=np.linspace(0, 1, 19), q=np.logspace(-10, 1, 10))
    assert np.all(k3.kernel.data == k1.kernel.data)

    # p and q can also be passed as coordinates:
    p0 = scp.Coord(np.linspace(0, 1, 19), name="pressure", title="p", units="torr")
    q0 = scp.Coord(
        np.logspace(-10, 1, 10),
        name="reduced adsorption energy",
        title=r"$\Delta_{ads}G^{0}/RT$",
        units="",
    )
    k4 = scp.IrisKernel(X, "langmuir", p=p0, q=q0)
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
    K = scp.IrisKernel(X, "langmuir", q=q)

    # Create an IRIS object with no regularization parameters
    iris1 = scp.IRIS()

    # Fit the IRIS model with a langmuir kernel
    iris1.fit(X, K)

    f1 = iris1.f.copy()
    assert f1.shape == (1, q[2], X.shape[1])

    X_hat = iris1.inverse_transform()
    assert X_hat.squeeze().shape == X.shape

    # test with callable
    def ker(p, q):
        return np.exp(-q) * p[:, None] / (1 + np.exp(-q) * p[:, None])

    K = scp.IrisKernel(X, ker, q=iris1.q)
    iris1.fit(X, K)
    f2 = iris1.f.copy()
    assert f2.shape == (1, q[2], X.shape[1])
    assert f1 == f2
    K1 = iris1.K  # keep it for later

    # no regularization, ca, 1D, p and q passed as a ndarrays
    X2 = X[:, 2110.0]
    K1D = scp.IrisKernel(X2, "ca", p=p, q=iris1.K.x.data)
    iris1.fit(X2, K1D)
    f1d = iris1.f.copy()
    assert f1d.shape == (1, q[2], 1)

    # manual regularization, keeping the previous langmuir kernel
    reg_par = [-4, -2, 3]
    iris2 = scp.IRIS(reg_par=reg_par)
    iris2.fit(X, K1)
    f2 = iris2.f
    assert f2.shape == (reg_par[2], q[2], X.shape[1])

    iris2.plotlcurve()
    iris2.plotdistribution(-2)
    _ = iris2.plotmerit(-2)

    # with automated search, keeping the previous kernel
    reg_par = [-4, -2]
    iris3 = scp.IRIS(reg_par=reg_par)
    iris3.fit(X, K1)
    f3 = iris3.f
    assert (f3.shape[1], f3.shape[2]) == (q[2], X.shape[1])

    # test other kernels (beware: it is chemical non-sense !)
    q = [5, 1, 10]
    iris4 = scp.IRIS()
    K4 = scp.IrisKernel(X, "diffusion", q=q)
    iris4.fit(X, K4)
    f4 = iris4.f
    assert f4.shape == (1, q[2], X.shape[1])

    q = [5, 1, 10]
    K5 = scp.IrisKernel(X, "product-first-order", q=q)
    iris4.fit(X, K5)
    f5 = iris4.f
    assert f5.shape == (1, q[2], X.shape[1])

    q = [5, 1, 10]
    K6 = scp.IrisKernel(X[::-1], "reactant-first-order", q=q)
    iris4.fit(X, K6)
    f6 = iris4.f
    assert f6.shape == (1, q[2], X.shape[1])

    q = [5, 1, 10]
    K7 = scp.IrisKernel(X[::-1], "stejskal-tanner", q=q)
    iris4.fit(X, K7)
    f7 = iris4.f
    assert f7.shape == (1, q[2], X.shape[1])
