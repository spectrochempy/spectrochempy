# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


from spectrochempy.api import ur, concatenate, stack,\
    CoordRange

import pytest

def test_concatenate(IR_source_1):

    source = IR_source_1

    #print(source)
    s1 = source[:10]
    s2 = source[20:]

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

def test_concatenate_1D_along_axis0(IR_source_1):

    source = IR_source_1

    # split all rows
    rows = []
    for s in source:
        rows.append(s)

    assert len(rows)==source.shape[0]

    # reconstruct
    new = stack(rows)
    assert new.shape == source.shape

    # #TODO: fix bug when ndim=1 (squeezed data)
    # using stack we should have a concatenation along a new axis 0 in this case.
    # for now it doesnt work.

    rows = []
    for s in source:
        rows.append(s.squeeze())

    # reconstruct from squeezed data
    new = stack(rows)
    assert new.shape == (source.size,)

def test_concatenate_along_axis1(IR_source_1):

    source = IR_source_1

    coord = source.coordset(-1)

    # test along axis 1
    ranges = ([6000., 3500.], [1800., 1500.])

    ranges = CoordRange(*ranges, reversed=coord.is_reversed)

    s = []
    for pair in ranges:
        # determine the slices
        sl = slice(*pair)
        s.append(source[..., sl])

    sbase = concatenate( *s, axis=-1)
    xbase = sbase.coordset(-1)

    assert sbase.shape[-1] == (s[0].shape[-1] + s[1].shape[-1])
    assert xbase.size == (s[0].coordset(-1).size + s[1].coordset(-1).size)

