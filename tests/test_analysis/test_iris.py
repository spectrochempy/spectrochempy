# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import numpy as np

import spectrochempy as scp


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
        title="$\Delta_{ads}G^{0}/RT$",
        units="",
    )
    k4 = scp.IrisKernel(X, "langmuir", p=p0, q=q0)
    assert np.all(k4.kernel.data == k1.kernel.data)
    pass


def test_IRIS():

    # Define the dataset to fit with the IRIS model
    # we change timestamp coordinates with the corresponding measured pressures.
    X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2105.0:1995.0]
    X[:, 2000.0:2004.0] = scp.MASKED  # Test with a mask
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
    iris1 = scp.IRIS2()

    # Fit the IRIS model with a langmuir kernel
    iris1.fit(X, K)

    f1 = iris1.f
    assert f1.shape == (1, q[2], X.shape[1])

    X_hat = iris1.reconstruct()
    assert X_hat.squeeze().shape == X.shape

    # test with callable
    def ker(p, q):
        import numpy as np

        return np.exp(-q) * p[:, None] / (1 + np.exp(-q) * p[:, None])

    iris1 = IRIS(X_, ker, q=iris1.K.x)
    f1 = iris1.f
    assert f1.shape == (1, q[2], X_.shape[1])

    # no regularization, ca, 1D, p and q passed as a ndarrays
    iris1b = IRIS(X_[:, 2110.0], "ca", p=p, q=iris1.K.x.data)

    f1b = iris1b.f
    assert f1b.shape == (1, q[2], 1)

    # manual regularization, keeping the previous kernel
    K = iris1.K
    reg_par = [-4, -2, 3]

    iris2 = IRIS(X_, K, reg_par=reg_par)
    f2 = iris2.f
    assert f2.shape == (reg_par[2], q[2], X_.shape[1])

    iris2.plotlcurve()
    iris2.plotdistribution(-2)
    _ = iris2.plotmerit(-2)

    # with automated search, keeping the previous kernel
    reg_par = [-4, -2]

    iris3 = IRIS(X_, K, reg_par=reg_par)
    f3 = iris3.f
    assert (f3.shape[1], f3.shape[2]) == (q[2], X_.shape[1])

    # test other kernels (beware: it is chemical non-sense !)
    q = [5, 1, 10]
    iris4 = IRIS(X_, "diffusion", q=q)
    f4 = iris4.f
    assert f4.shape == (1, q[2], X_.shape[1])

    q = [5, 1, 10]
    iris4 = IRIS(X_, "product-first-order", q=q)
    f4 = iris4.f
    assert f4.shape == (1, q[2], X_.shape[1])

    q = [5, 1, 10]
    iris4 = IRIS(X_[::-1], "reactant-first-order", q=q)
    f4 = iris4.f
    assert f4.shape == (1, q[2], X_.shape[1])
