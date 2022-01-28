# -*- coding: utf-8 -*-
# flake8: noqa

import os
import pytest

from spectrochempy.analysis.iris import IRIS
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import show

# pytestmark = pytest.mark.skip("WIP: iris dev on going")

# import pytest
# @pytest.mark.skip('do not work with workflow - need to solve this!')
def test_IRIS():
    X = NDDataset.read_omnic(os.path.join("irdata", "CO@Mo_Al2O3.SPG"))

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

    X.coordset.update(y=Coord(p, title="pressure", units="torr"))
    # Using the `update` method is mandatory because it will preserve the name.
    # Indeed, setting using X.coordset[0] = Coord(...) fails unless name is specified: Coord(..., name='y')

    # take a small region to reduce test time
    X_ = X[:, 2105.0:1995.0]

    # no regularization, langmuir
    q = [-8, -1, 10]
    iris1 = IRIS(X_, "langmuir", q=q)
    f1 = iris1.f
    assert f1.shape == (1, q[2], X_.shape[1])

    X_hat = iris1.reconstruct()
    assert X_hat.squeeze().shape == X_.shape

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
