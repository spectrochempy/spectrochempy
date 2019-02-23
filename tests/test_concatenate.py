# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from spectrochempy.core.processors.concatenate import concatenate, stack
from spectrochempy.core.dataset.ndcoords import CoordRange
from spectrochempy.units import ur
from spectrochempy.utils import info_, MASKED, show

def test_concatenate(IR_dataset_2D):
    dataset = IR_dataset_2D

    # print(dataset)
    s1 = dataset[:, -10:]
    s2 = dataset[:, :-10]

    dim = 'x'
    s = concatenate(s1, s2, dims=dim)
    assert s.units == s1.units
    assert s.shape[-1] == (s1.shape[-1] + s2.shape[-1])
    assert s.coords(-1).size == (s1.coords(-1).size + s2.coords(-1).size)
    assert s.x != dataset.x
    s = s.sort(dims=dim, descend=True)  #
    assert s.x == dataset.x
    s.plot()

    s1 = dataset[:10]
    s2 = dataset[20:]
    # check with derived units
    s1.to(ur.m, force=True)
    s2.to(ur.dm, force=True)
    s = concatenate(s1, s2, dim=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.coords(0).size == (s1.coords(0).size + s2.coords(0).size)
    s = s.sort(dim='y')
    s.plot()

    # second syntax
    s = s1.concatenate(s2, dim=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.coords(0).size == (s1.coords(0).size + s2.coords(0).size)

    # third syntax
    s = concatenate((s1, s2), dim=0)
    assert s.units == s1.units
    assert s.shape[0] == (s1.shape[0] + s2.shape[0])
    assert s.coords(0).size == (s1.coords(0).size + s2.coords(0).size)


def test_concatenate_1D_along_axis1(IR_dataset_2D):

    # equivalent to stack

    dataset = IR_dataset_2D

    # make these data with a mask
    dataset[:,1] = MASKED
    info_(dataset)

    # split all rows
    rows = []
    for i in range(len(dataset)):
        rows.append(dataset[i])
    assert len(rows) == dataset.shape[0]

    # reconstruct
    new = stack(rows)
    assert new.shape == dataset.shape
    info_(new)

    # split by pairs of columns
    # need an even number of columns
    dataset = dataset[:54]
    rows = []
    for i in range(0, len(dataset), 2):
        rows.append(dataset[i:i + 2])
    assert len(rows) == dataset.shape[0] / 2

    # reconstruct
    new = stack(rows)
    assert new.shape == dataset.shape
    info_(new)


def test_concatenate_along_dim_x(IR_dataset_2D):

    dim = 'x'
    dataset = IR_dataset_2D
    dim = axis = dataset.get_axis(dim)
    coord = dataset.coord(dim=dim)

    # test along dim x
    ranges = ( [1500., 1800.], [3500., 6000.])

    ranges = CoordRange(*ranges, reversed=coord.reversed)
    assert ranges == [[6000.0, 3500.0], [1800.0, 1500.0]]  # rearranged according the reversed flag

    s = []
    for pair in ranges:
        # determine the slices
        sl = slice(*pair)
        s.append(dataset[..., sl])

    sbase = concatenate(*s, dim=dim)
    xbase = sbase.coord(dim=dim)

    assert sbase.shape[axis] == (s[0].shape[axis] + s[1].shape[axis])
    assert xbase.size == (s[0].coords(axis).size + s[1].coords(axis).size)

    sbase.plot_stack()
    show()
