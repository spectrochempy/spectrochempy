# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# =============================================================================

"""Tests for the ndmath module

"""
import pandas as pd
import pytest

#from spectrochempy import *
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.dataset.ndcoords import Coord, CoordSet
from spectrochempy.units.units import ur
from spectrochempy.utils import show, masked

from spectrochempy.extern.pint.errors import (UndefinedUnitError,
                                              DimensionalityError)
from spectrochempy.utils import Meta, SpectroChemPyWarning
from tests.utils import (assert_equal, assert_array_equal,
                         assert_array_almost_equal, assert_equal_units,
                         raises)
from tests.utils import RandomSeedContext
import numpy as np



def test_absolute_of_complex():
    ndd = NDDataset([1., 2. + 1j, 3.])

    val = np.abs(ndd)
    # print(val)

    val = ndd[1] * 1.2 - 10.
    val = np.abs(val)
    # print(val)

    na0 = np.array([[1., 2., 2., 0., 0., 0.],
                    [1.3, 2., 2., 0.5, 1., 1.],
                    [1, 4.2, 2., 3., 2., 2.],
                    [5., 4.2, 2., 3., 3., 3.]])
    nd = NDDataset(na0)
    coordset = CoordSet([np.linspace(-1, 1, 4), np.linspace(-10., 10., 6)])
    assert nd.shape == (4, 6)
    nd.coordset = coordset
    nd.set_complex(axis=0)
    # print(nd)

    val = np.abs(nd)  # this dimension (the last is ot complex)
    # print(val)

    val = np.fabs(nd)  # this dimension (the last is ot complex)
    # print(val)   # should work the same  (works only on the last dimension

    val = nd.abs(axis=0)  # the np.abs works only on the last dimension
    # print(val)

    # TODO: add more testings


def test_ufunc_method(nd):
    assert isinstance(nd, NDDataset)
    nd2 = np.sin(nd)
    nds = np.sqrt(nd)

    assert nd2 is not nd
    assert nd2._data is not nd._data
    sinnd = np.sin(nd._data)
    assert_array_equal(nd2._data, sinnd)

    nd2.units = 'm'
    nds = np.sqrt(nd2)
    assert nds.units == ur.m ** .5
    assert_array_equal(nds._data, np.sqrt(nd2._data))

    nds._data = nds._data[:4, :2]
    # print(nds.shape)
    ndsw = np.swapaxes(nds, 1, 0)
    # print(ndsw.shape)
    pass

def test_unary_ops():
    # UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
    d1 = NDDataset(np.ones((5, 5)))
    d2 = +d1  # pos
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == 1.)
    d2 = -d1  # neg
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == -1.)
    d3 = abs(d2)  # abs
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.)


def test_unary_ops_with_units():
    # UNARY_OPS = ['neg', 'pos', 'abs']
    d1 = NDDataset(np.ones((5, 5)), units='m')
    d2 = +d1  # pos
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == 1.)
    assert d2.units == ur.m
    d2 = -d1  # neg
    assert isinstance(d2, NDDataset)
    assert np.all(d2.data == -1.)
    assert d2.units == ur.m
    d3 = abs(d2)  # abs
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.)
    assert d2.units == ur("m")


def test_nddataset_add():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)))
    d3 = -d1
    d3 = d1 * .5 + d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)


def test_nddataset_add_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5))
    d3 = d1 * .5 + d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)
    # should commute!
    d3 = d2 + d1 * .5
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 1.5)


def test_nddataset_add_inplace():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)))
    d1 += d2 * .5
    assert np.all(d1.data == 1.5)


def test_nddataset_add_mismatch_coords():
    coord1 = Coord(np.arange(5.))
    coord2 = Coord(np.arange(1., 5.5, 1.))
    d1 = NDDataset(np.ones((5, 5)), coordset=[coord1, coord2])
    d2 = NDDataset(np.ones((5, 5)), coordset=[coord2, coord1])
    with pytest.raises(ValueError) as exc:
        d3 = d1 + d2
    assert exc.value.args[0] == "coordinate's values do not match"
    with pytest.raises(ValueError) as exc:
        d1 += d2
    assert exc.value.args[0] == "coordinate's values do not match"
    # TODO= make more tests like this for various functions

def test_nddataset_add_mismatch_units():
    d1 = NDDataset(np.ones((5, 5)), units='cm^2')
    d2 = NDDataset(np.ones((5, 5)), units='cm')
    with pytest.raises(DimensionalityError) as exc:
        d3 = d1 + d2
    assert exc.value.__str__() == \
           "Cannot convert from 'centimeter' ([length]) to 'centimeter ** 2' " \
           "([length] ** 2)"
    with pytest.raises(DimensionalityError) as exc:
        d1 += d2
    assert exc.value.__str__() == \
           "Cannot convert from 'centimeter ** 2' ([length] ** 2) " \
           "to 'centimeter' ([length])"


def test_nddataset_add_mismatch_shape():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((6, 6)))
    with pytest.raises(ArithmeticError) as exc:
        d1 += d2
    assert exc.value.args[0].startswith(
            "operands could not be broadcast together")


def test_nddataset_add_with_masks():
    # numpy masked arrays mask the result of binary operations if the
    # mask of either operand is set.
    # Does NDData?
    ndd1 = NDDataset(np.array([1, 2]))
    ndd2 = NDDataset(np.array([2, 1]))
    result = ndd1 + ndd2
    assert_array_equal(result.data, np.array([3, 3]))

    ndd1 = NDDataset(np.array([1, 2]), mask=np.array([True, False]))
    other_mask = ~ ndd1.mask
    ndd2 = NDDataset(np.array([2, 1]), mask=other_mask)
    result = ndd1 + ndd2
    # The result should have all entries masked...
    assert result.mask.all()


def test_nddataset_add_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)), uncertainty=u2)
    d3 = d1 + d2
    assert np.all(d3.data == 2.)
    assert_array_equal(d3.uncertainty, np.sqrt(10.))


def test_nddataset_subtract():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((5, 5)) * 2.)
    d3 = d1 - d2
    assert np.all(d3.data == -1.)


def test_nddataset_substract_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5))
    d3 = d1 * .5 - d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == -0.5)
    # should commute!
    d3 = d2 - d1 * .5
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 0.5)


def test_nddataset_subtract_mismatch_coords():
    coord1 = Coord(np.arange(5.))
    coord2 = Coord(np.arange(1., 5.5, 1.))
    d1 = NDDataset(np.ones((5, 5)), coordset=[coord1, coord2])
    d2 = NDDataset(np.ones((5, 5)), coordset=[coord2, coord1])
    with pytest.raises(ValueError) as exc:
        d1 -= d2
    assert exc.value.args[0] == "coordinate's values do not match"


def test_nddataset_subtract_mismatch_units():
    d1 = NDDataset(np.ones((5, 5)), units='m')
    d2 = NDDataset(np.ones((5, 5)) * 2., units='m/s')
    with pytest.raises(DimensionalityError) as exc:
        d1 -= d2
    assert exc.value.__str__() == "Cannot convert from 'meter' ([length]) " \
                                "to 'meter / second' ([length] / [time])"

def test_nddataset_subtract_mismatch_shape():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((6, 6)) * 2.)
    with pytest.raises(ArithmeticError) as exc:
        d1 -= d2
    assert exc.value.args[0].startswith(
            "operands could not be broadcast together")


def test_nddataset_subtract_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)) * 2., uncertainty=u2)
    d3 = d1 - d2
    assert np.all(d3.data == -1.)
    assert_array_equal(d3.uncertainty, np.sqrt(10.))


def test_nddataset_multiply_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)) * 2., uncertainty=u2)
    d3 = d1 * d2
    assert np.all(d3.data == 2.)
    assert_array_equal(d3.uncertainty, 2 * np.sqrt(9.25))


def test_nddataset_multiply_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5)) * 2.
    d3 = d1 * d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.)
    # should commute!
    d3 = d2 * d1
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.)


def test_nddataset_divide_uncertainties():
    u1 = np.ones((5, 5)) * 3
    u2 = np.ones((5, 5))
    d1 = NDDataset(np.ones((5, 5)), uncertainty=u1)
    d2 = NDDataset(np.ones((5, 5)) * 2., uncertainty=u2)
    d3 = d1 / d2
    assert np.all(d3.data == 0.5)
    assert_array_equal(d3.uncertainty, 0.5 * np.sqrt(9.25))


def test_nddataset_divide_with_numpy_array():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = np.ones((5, 5)) * 2.
    d3 = d1 / d2
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 0.5)
    # should commute!
    d3 = d2 / d1
    assert isinstance(d3, NDDataset)
    assert np.all(d3.data == 2.)


# first operand has units km, second has units m
@pytest.mark.parametrize(('operation', 'result_units'), [
    ('__add__', ur.km),
    ('__sub__', ur.km),
    ('__mul__', ur.km * ur.m),
    ('__truediv__', ur.km / ur.m)])
def test_uncertainty_unit_conversion_operators(operation, result_units):
    in_km = NDDataset(np.array([1, 1]), units=ur.km,
                      uncertainty=np.array([.1, .1]))
    in_m = NDDataset(in_km.data * 1000, units=ur.m)
    in_m.uncertainty = np.array(in_km.uncertainty * 1000)
    operator_km = in_km.__getattribute__(operation)
    combined = operator_km(in_m)
    assert_equal_units(combined.units, result_units)
    if operation in ['__add__', '__sub__']:
        # uncertainty is not scaled by result values
        assert_array_almost_equal(combined.uncertainty,
                                  np.sqrt(2) * in_km.uncertainty)
    else:
        # uncertainty is scaled by result
        assert_array_almost_equal(combined.uncertainty,
                                  np.sqrt(
                                          2) * in_km.uncertainty * combined.data)


@pytest.mark.parametrize(('unit1', 'unit2', 'op', 'result_units'), [
    (None, None, '__add__', None),
    (None, None, '__mul__', None),
    (None, ur.m, '__mul__', ur.m),
    (ur.dimensionless, None, '__mul__',
     ur.dimensionless),
    (ur.eV, ur.eV, '__add__', ur.eV),
    (ur.eV, ur.eV, '__sub__', ur.eV),
    (ur.eV, ur.eV, '__truediv__', ur.dimensionless),
    (ur.eV, ur.m, '__mul__', ur.m * ur.eV)
])
def test_arithmetic_unit_calculation(unit1, unit2, op, result_units):
    ndd1 = NDDataset(np.array([1]), units=unit1)
    ndd2 = NDDataset(np.array([1]), units=unit2)
    ndd1_method = ndd1.__getattribute__(op)
    result = ndd1_method(ndd2)
    assert result.units == result_units


# ----------------------------------------------------------------------------
# unary operations (ufuncs)
# ----------------------------------------------------------------------------

UNARY_MATH = ["fabs", "ceil", "floor", "negative", "reciprocal",
              "rint", "sqrt", "square"]

@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_simple_data(nd, name):
    assert nd.unitless
    f = getattr(np, name)
    r = f(nd)
    assert isinstance(r, NDDataset)


@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_data_w_uncertainties(nd, name):
    nd._uncertainty = np.abs(nd._data * .01)
    assert nd.unitless
    f = getattr(np, name)
    r = f(nd)
    assert isinstance(r, NDDataset)


@pytest.mark.parametrize('name', UNARY_MATH)
def test_unary_ufuncs_data_w_uncertainties(nd, name):
    nd.units = ur.m
    nd._uncertainty = np.abs(nd._data * .01)
    f = getattr(np, name)
    r = f(nd)
    assert isinstance(r, NDDataset)



# ----------------------------------------------------------------------------
# non ufuncs
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(('operation', 'result'),
                         [
                                ('sum', 46 ),
                                ('cumsum', [1,3,6,10,15,21,28,36,36,46] ),
                          ])
def test_non_ufunc_functions(operation, result):
    op = getattr(np, operation)
    print(op.__doc__)
    ds = NDDataset([1,2,3,4,5,6,7,8,9,10])
    coord0 = Coord(np.arange(10)*.1)
    ds.coordset = [coord0]
    ds[-2] = masked
    ds.units = ur.m
    dsy = op(ds, axis=-1)
    print(dsy)
    assert np.all(dsy.data==result)

@pytest.mark.parametrize(('operation', 'result'),
                         [
                                ('sum',    [3,7]        ),
                                ('cumsum', [[1,3],[3,7]]),

                          ])
def test_non_ufunc_functions_with_2D(operation, result):
    op = getattr(np, operation)
    print(op.__doc__)
    ds = NDDataset([[1,2],[3,4]])
    coord0 = Coord(np.arange(2) * .1)
    coord1 = Coord(np.arange(2) * .2)
    ds.coordset = [coord0,coord1]
    ds.units = ur.m
    dsy = op(ds, axis=-1)
    print(dsy)
    assert np.all(dsy.data==result)


# ----------------------------------------------------------------------------
# additional tests made following some bug fixes
# ----------------------------------------------------------------------------

def test_simple_arithmetic_on_full_dataset():
    # due to a bug in notebook with the following
    import os
    source = NDDataset.read_omnic(os.path.join('irdata', 'NH4Y-activation.SPG'))
    d = source - source[0]
    # suppress the first spectrum to all other spectra in the series

