# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy.processing.transformation.concatenate import concatenate, stack
from spectrochempy.utils.exceptions import (
    DimensionsCompatibilityError,
    UnitsCompatibilityError,
)
from spectrochempy.utils.testing import assert_dataset_almost_equal


def test_concatenate(IR_dataset_2D):
    dataset = IR_dataset_2D
    dim = "x"
    # print(dataset)
    s = dataset
    s1 = dataset[:, -10:]
    s2 = dataset[:, :-10]

    # specify axis
    s = concatenate(s1, s2, dims=dim)
    assert s.units == s1.units
    assert s.shape[-1] == (s1.shape[-1] + s2.shape[-1])
    assert s.x.size == (s1.x.size + s2.x.size)
    assert s.x != dataset.x
    s = s.sort(dims=dim, descend=True)  #
    assert_dataset_almost_equal(s.x, Coord(dataset.x, linear=False), decimal=3)

    # default concatenation in the last dimension
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
    s = concatenate(s1, s2, dims=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)
    s = s.sort(dim="y")

    # second syntax
    s = s1.concatenate(s2, dims=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)

    # third syntax
    s = concatenate((s1, s2), dims=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.y.size == (s1.y.size + s2.y.size)

    # coordset
    coord_2 = Coord(np.cos(s.y.data), title="cos_time")
    s.set_coordset(y=[s.y, coord_2], x=s.x)
    s1 = s[:2]
    s2 = s[-5:]
    s12 = concatenate(s1, s2, axis=0)
    assert (s2["y"].labels[1] == s12["y"].labels[1][-5:]).all()

    # authors
    s0 = s[0]
    s1 = s[1]
    s0.author = "sdqe65g4rf"
    s2 = concatenate(s0, s1)
    assert "sdqe65g4rf" in s2.author and s1.author in s2.author

    # titles
    s0.title = "new_title"
    assert concatenate(s0, s1).title == "new_title"

    # incompatible dimensions
    s0 = scp.NDDataset(np.zeros((10, 100)))
    s1 = scp.NDDataset(np.zeros((10, 100)))
    with pytest.raises(DimensionsCompatibilityError):
        s0.concatenate(s1[0].squeeze())

    with pytest.raises(DimensionsCompatibilityError):
        s0.concatenate(s1[:, :50], axis=0)

    # incompatible units
    s0 = scp.NDDataset(np.zeros((10, 100)), units="V")
    s1 = scp.NDDataset(np.zeros((10, 100)), units="A")
    with pytest.raises(UnitsCompatibilityError):
        scp.concatenate(s0, s1)

    s1 = scp.NDDataset(np.ones((10, 100)), units="mV")
    s01 = scp.concatenate(s0, s1)
    assert s01.data[-1, -1] == 0.001

    # ----------------------------------------------------------------------------------
    # Stack

    # concatenation using stack
    s1 = dataset[:10]
    s2 = dataset[-10:]
    s = stack(s1, s2)
    assert s.units == s1.units
    assert s.shape == (2, s1.shape[0], s1.shape[1])
    assert s.y.size == s1.y.size
    assert s.x.size == s1.x.size

    with pytest.warns(DeprecationWarning):
        concatenate(s1, s2, force_stack=True)

    # If one of the dimensions is of size one, then this dimension is NOT removed before stacking
    s0 = dataset[0]
    s1 = dataset[1]
    ss = stack(s0, s1)
    assert s0.shape == (1, 5549)
    assert ss.shape == (2, s1.shape[0], s1.shape[1])

    # # stack squeezed nD dataset
    s0 = dataset[0].copy().squeeze()
    assert s0.shape == (5549,)
    s1 = dataset[1].squeeze()
    assert s1.shape == (5549,)
    s = stack(s0, s1)
    assert s.shape == (2, 5549)

    # # stack squeezed nD dataset
    s2 = s1[0:100]
    with pytest.raises(DimensionsCompatibilityError):
        s = stack(s0, s2)


def test_bug_243():
    import spectrochempy as scp

    D = scp.zeros((10, 100))

    x = scp.Coord.arange(100)
    y = scp.Coord.arange(10)

    D.set_coordset(x=x, y=y)
    D1 = D[:, 0.0:10.0]
    D2 = D[:, 20.0:40.0]

    D12 = scp.concatenate(D1, D2, axis=1)

    # D2.x.data[-1] is 40., as expected, but not D12.x.data[-1]:
    assert D12.x.data[-1] == D2.x.data[-1]
