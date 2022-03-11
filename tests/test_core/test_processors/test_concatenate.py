# -*- coding: utf-8 -*-
# flake8: noqa

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.processors.concatenate import concatenate, stack
from spectrochempy.core.units import ur
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.utils.testing import assert_dataset_almost_equal
from spectrochempy.utils import show
from spectrochempy.utils.exceptions import DimensionsCompatibilityError

import pytest


def test_concatenate(IR_dataset_2D):
    dataset = IR_dataset_2D

    # print(dataset)
    s1 = dataset[:, -10:]
    s2 = dataset[:, :-10]

    # specify axis
    dim = "x"
    s = concatenate(s1, s2, dims=dim)
    assert s.units == s1.units
    assert s.shape[-1] == (s1.shape[-1] + s2.shape[-1])
    assert s.x.size == (s1.x.size + s2.x.size)
    assert s.x != dataset.x
    s = s.sort(dims=dim, descend=True)  #
    assert_dataset_almost_equal(s.x, Coord(dataset.x, linear=False), decimal=3)

    # default concatenation in the last dimensions
    s = concatenate(s1, s2)
    assert s.units == s1.units
    assert s.shape[-1] == (s1.shape[-1] + s2.shape[-1])
    assert s.x.size == (s1.x.size + s2.x.size)
    assert s.x != dataset.x
    s = s.sort(descend=True)  #
    assert_dataset_almost_equal(s.x, Coord(dataset.x, linear=False), decimal=3)

    s1 = dataset[:10]
    s2 = dataset[20:]
    # check with derived units
    s1.to(ur.m, force=True)
    s2.to(ur.dm, force=True)
    s = concatenate(s1, s2, dim=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)
    s = s.sort(dim="y")
    s.plot()

    # second syntax
    s = s1.concatenate(s2, dim=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)

    # third syntax
    s = concatenate((s1, s2), dim=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)

    # concatenation in the first dimension using stack
    s = stack(s1, s2)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)

    # Stacking of datasets:
    # for nDimensional datasets (with the same shape), a new dimension is added
    ss = concatenate(s.copy(), s.copy(), force_stack=True)  # make a copy of s
    # (dataset cannot be concatenated to itself!)
    assert ss.shape == (2, 45, 5549)

    # If one of the dimensions is of size one, then this dimension is removed before stacking
    s0 = s[0]
    s1 = s[1]
    ss = s0.concatenate(s1, force_stack=True)
    assert s0.shape == (1, 5549)
    assert ss.shape == (2, 5549)

    # if incompatible dimensions
    s0 = s[0, :1000]
    s1 = s[1]
    with pytest.raises(DimensionsCompatibilityError):
        s0.concatenate(s1, force_stack=True)
    s0 = s[0]
    s1 = s[1].squeeze()
    with pytest.raises(DimensionsCompatibilityError):
        s0.concatenate(s1, force_stack=True)

    s0 = s[0]
    s1 = s[1]
    s0.author = "sdqe65g4rf"
    s2 = concatenate(s0, s1)
    assert "sdqe65g4rf" in s2.author and s1.author in s2.author

    # stack squeezed nD dataset
    s0 = s[0].copy().squeeze()
    assert s0.shape == (5549,)
    s1 = s[1].squeeze()
    assert s1.shape == (5549,)
    ss = concatenate(s0, s1, force_stack=True)
    assert ss.shape == (2, 5549)

    show()


def test_bug_243():
    import spectrochempy as scp

    D = scp.zeros((10, 100))

    x = scp.LinearCoord(offset=0.0, increment=1.0, size=100)
    y = scp.LinearCoord(offset=0.0, increment=1.0, size=10)

    D.set_coordset(x=x, y=y)
    D1 = D[:, 0.0:10.0]
    D2 = D[:, 20.0:40.0]

    D12 = scp.concatenate(D1, D2, axis=1)

    # D2.x.data[-1] is 40., as expected, but not D12.x.data[-1]:
    assert D12.x.data[-1] == D2.x.data[-1]


def test_bug_doctring():
    import spectrochempy as scp

    A = scp.read("irdata/nh4y-activation.spg", protocol="omnic")
    B = scp.read("irdata/nh4y-activation.scp")
    C = scp.concatenate(A[10:], B[3:5], A[:10], axis=0)
    assert (A[10:].shape, B[3:5].shape, A[:10].shape, C.shape) == (
        (45, 5549),
        (2, 5549),
        (10, 5549),
        (57, 5549),
    )

    D = A.concatenate(B, B, axis=0)
    assert (A.shape, B.shape, D.shape) == ((55, 5549), (55, 5549), (165, 5549))

    E = A.concatenate(B, axis=1)
    assert (A.shape, B.shape, E.shape) == ((55, 5549), (55, 5549), (55, 11098))

    F = A.concatenate(B, force_stack=True)
    assert (A.shape, B.shape, F.shape) == ((55, 5549), (55, 5549), (2, 55, 5549))

    G = A[0].concatenate(B[0], force_stack=True)
    assert (A[0].shape, B[0].shape, G.shape) == ((1, 5549), (1, 5549), (2, 5549))

    C = scp.stack(A, B)
    assert str(C) == "NDDataset: [float64] a.u. (shape: (z:2, y:55, x:5549))"
