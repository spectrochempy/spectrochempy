# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT  
# See full LICENSE agreement in the root directory
# =============================================================================


from spectrochempy import *

import pytest

def test_concatenate(IR_dataset_2D):

    dataset = IR_dataset_2D

    #print(dataset)
    s1 = dataset[:10]
    s2 = dataset[20:]

    # check with derived units
    s1.to(ur.m, force=True)
    s2.to(ur.dm, force=True)
    s = concatenate(s1, s2)
    assert s.units==s1.units
    assert s.shape[0]==(s1.shape[0]+s2.shape[0])
    assert s.coordset(0).size==(s1.coordset(0).size+s2.coordset(0).size)
    s = s.sort(axis=0)
    s.plot()

    # second syntax
    s = s1.concatenate(s2)
    assert s.units==s1.units
    assert s.shape[0]==(s1.shape[0]+s2.shape[0])
    assert s.coordset(0).size==(s1.coordset(0).size+s2.coordset(0).size)

    # third syntax
    s = concatenate((s1, s2))
    assert s.units==s1.units
    assert s.shape[0]==(s1.shape[0]+s2.shape[0])
    assert s.coordset(0).size==(s1.coordset(0).size+s2.coordset(0).size)

def test_concatenate_1D_along_axis0(IR_dataset_2D):
    # TODO: very long process - try to optimize this
    dataset = IR_dataset_2D[3:]

    # split all rows
    rows = []
    for i in range(len(dataset)):
        rows.append(dataset[i])

    assert len(rows)==dataset.shape[0]

    # reconstruct
    new = stack(rows)
    assert new.shape == dataset.shape

    # #TODO: fix bug when ndim=1 (squeezed data)
    # using stack we should have a concatenation along a new axis 0 in this case.
    # for now it doesnt work.

    rows = []
    for s in dataset:
        rows.append(s)
        assert s.shape == (5549,)
        print(s._mask)
        assert not s.is_masked

    # reconstruct from rows
    new = stack(rows)
    assert new.shape == dataset.shape

def test_concatenate_along_axis1(IR_dataset_2D):

    dataset = IR_dataset_2D

    coord = dataset.coordset(-1)

    # test along axis 1
    ranges = ([6000., 3500.], [1800., 1500.])

    ranges = CoordRange(*ranges, reversed=coord.is_reversed)

    s = []
    for pair in ranges:
        # determine the slices
        sl = slice(*pair)
        s.append(dataset[..., sl])

    sbase = concatenate( *s, axis=-1)
    xbase = sbase.coordset(-1)

    assert sbase.shape[-1] == (s[0].shape[-1] + s[1].shape[-1])
    assert xbase.size == (s[0].coordset(-1).size + s[1].coordset(-1).size)

    sbase.plot_stack()
    show()