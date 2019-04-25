# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT 
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""Tests for the ndmath module

"""
import pandas as pd
import pytest

# from spectrochempy import *
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndmath import unary_ufuncs, binary_ufuncs
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.units.units import ur, Quantity
from spectrochempy.utils import (MASKED, TYPE_FLOAT, TYPE_INTEGER,
                                 TYPE_COMPLEX)
from spectrochempy.core import info_, debug_, warning_, error_

from pint.errors import (UndefinedUnitError,
                         DimensionalityError)
from spectrochempy.utils import Meta, SpectroChemPyWarning
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                                         assert_array_almost_equal, assert_equal_units,
                                         raises, assert_approx_equal)
from spectrochempy.utils.testing import RandomSeedContext, catch_warnings
import numpy as np

from quaternion import quaternion

typequaternion = np.dtype(np.quaternion)


# ----------------------------------------------------------------------------------------------------------------------
# ufuncs
# ----------------------------------------------------------------------------------------------------------------------


def test_ndmath_show():
    info_()
    for item in unary_ufuncs().items():
        info_(*item)
    
    for item in binary_ufuncs().items():
        info_(*item)


# UNARY MATHS

@pytest.mark.parametrize(('name', 'comment'), unary_ufuncs().items())
def test_ndmath_unary_ufuncs_simple_data(nd2d, pnl, name, comment):
    nd1 = nd2d.copy() / 1.e+10  # divide to avoid some overflow in exp ufuncs
    print(f"\n{name}   # {comment}")
    
    # simple NDDataset
    # -----------------
    print(nd1)
    assert nd1.unitless
    
    f = getattr(np, name)
    r = f(nd1)
    print('after ', r)
    # assert isinstance(r, NDDataset)
    
    # NDDataset with units
    # ----------
    nd1.units = ur.absorbance
    print('units ', nd1)
    f = getattr(np, name)
    
    # TODO: some ufunc suppress the units! see pint.
    skip = False
    try:
        expected_units = f(Quantity(1., nd1.units)).units
    except AttributeError:
        if name in ['positive', 'fabs', 'cbrt', 'sign', 'spacing',
                    'signbit', 'isnan', 'isinf', 'isfinite', 'logical_not',
                    'log2', 'log10', 'log1p', 'exp2', 'expm1']:
            pass  # already solved
        else:
            print(f"\n =======> {name} remove units! \n")
    except DimensionalityError as e:
        error_(f"{name} :", e)
        skip = True
    
    if not skip:
        try:
            r = f(nd1)
            # assert isinstance(r, NDDataset)
            print('after units ', r)
            
            nd1 = nd2d.copy()  # reset nd
            
            # with units and mask
            nd1.units = ur.absorbance
            nd1[1, 1] = MASKED
            print('mask ', nd1)
            r = f(nd1)
            print('after mask', r)
            # assert isinstance(r, NDDataset)
        
        except DimensionalityError as e:
            error_(f"{name} : ", e)
    
    # NDPanel
    # -----------------
    if name not in ['sign', 'logical_not', 'isnan', 'isfinite', 'isinf', 'signbit', ]:
        print('panel before', pnl)
        
        f = getattr(np, name)
        try:
            r = f(pnl)
            print('panel after ', r)
        except TypeError as e:
            error_(e)
    
    print('-' * 60)


def test_bug_lost_dimensionless_units():
    from spectrochempy import print_
    import os
    dataset = NDDataset.read_omnic(os.path.join('irdata', 'NH4Y-activation.SPG'))
    assert dataset.units == 'absorbance'
    dataset = dataset - 2.  # artificially make negative some of the values
    assert dataset.units == 'absorbance'
    
    dataset = dataset.clip(-2., 2.)
    y = np.log2(dataset)
    y._repr_html_()
    print_(y)
    print(y)


# BINARY MATH

def test_logaddexp(nd2d):
    nd1 = nd2d.copy()  # divide to avoid some overflow in exp ufuncs
    nd2 = nd1.copy() + np.zeros_like(nd1)
    r = np.logaddexp(nd1, nd2)
    print('after ', r)

@pytest.mark.parametrize(('name', 'comment'), binary_ufuncs().items())
def test_ndmath_binary_ufuncs_simple_data(nd2d, pnl, name, comment):
    nd1 = nd2d.copy()  # divide to avoid some overflow in exp ufuncs
    nd2 = nd1.copy() + np.zeros_like(nd1)
    
    print(f"\n{name}   # {comment}")
    
    # simple NDDataset
    # -----------------
    
    f = getattr(np, name)
    r = f(nd1, nd2)
    print(r)
    assert isinstance(r, NDDataset)

    # NDDataset with units
    # -----------------------
    nd1.units = ur.absorbance
    f = getattr(np, name)
    r = f(nd1, nd2)
    print(r)
    assert isinstance(r, NDDataset)
    if name not in ['logaddexp', 'logaddexp2', 'true_divide', 'floor_divide', ]:
        assert r.units == nd1.units
    
    

    
    
def test():
    
    # TODO: some ufunc suppress the units! see pint.
    skip = False
    try:
        expected_units = f(Quantity(1., nd1.units)).units
    except AttributeError:
        if name in ['positive', 'fabs', 'cbrt', 'sign', 'spacing',
                    'signbit', 'isnan', 'isinf', 'isfinite', 'logical_not',
                    'log2', 'log10', 'log1p', 'exp2', 'expm1']:
            pass  # already solved
        else:
            print(f"\n =======> {name} remove units! \n")
    except DimensionalityError as e:
        error_(f"{name} :", e)
        skip = True
    
    if not skip:
        try:
            r = f(nd1)
            # assert isinstance(r, NDDataset)
            print('after units ', r)
            
            nd1 = nd2d.copy()  # reset nd
            
            # with units and mask
            nd1.units = ur.absorbance
            nd1[1, 1] = MASKED
            print('mask ', nd1)
            r = f(nd1)
            print('after mask', r)
            # assert isinstance(r, NDDataset)
        
        except DimensionalityError as e:
            error_(f"{name} : ", e)
    
    # NDPanel
    # -----------------
    if name not in ['sign', 'logical_not', 'isnan', 'isfinite', 'isinf', 'signbit', ]:
        print('panel before', pnl)
        
        f = getattr(np, name)
        try:
            r = f(pnl)
            print('panel after ', r)
        except TypeError as e:
            error_(e)
    
    print('-' * 60)


def test_ndmath_ufunc_method(nd2d):
    nd = nd2d.copy()
    assert isinstance(nd, NDDataset)
    
    nda = np.sin(nd)
    assert nda is not nd
    assert nda._data is not nd._data
    
    assert_array_equal(nda._data, np.sin(nd._data))
    
    nda.units = 'm'
    
    ndb = np.sqrt(nda)
    assert ndb.units == ur.m ** .5
    
    sqrt = np.sqrt
    if ndb.dtype in TYPE_COMPLEX:
        sqrt = np.emath.sqrt
    
    assert_array_equal(ndb._data, sqrt(nda._data))


"""
add(x1, x2, /[, out, where, casting, order, …])	Add arguments element-wise.
subtract(x1, x2, /[, out, where, casting, …])	Subtract arguments, element-wise.
multiply(x1, x2, /[, out, where, casting, …])	Multiply arguments element-wise.
divide(x1, x2, /[, out, where, casting, …])	Returns a true division of the inputs, element-wise.
logaddexp(x1, x2, /[, out, where, casting, …])	Logarithm of the sum of exponentiations of the inputs.
logaddexp2(x1, x2, /[, out, where, casting, …])	Logarithm of the sum of exponentiations of the inputs in base-2.
true_divide(x1, x2, /[, out, where, …])	Returns a true division of the inputs, element-wise.
floor_divide(x1, x2, /[, out, where, …])	Return the largest integer smaller or equal to the division of the inputs.
power(x1, x2, /[, out, where, casting, …])	First array elements raised to powers from second array, element-wise.
remainder(x1, x2, /[, out, where, casting, …])	Return element-wise remainder of division.
mod(x1, x2, /[, out, where, casting, order, …])	Return element-wise remainder of division.
fmod(x1, x2, /[, out, where, casting, …])	Return the element-wise remainder of division.
divmod(x1, x2[, out1, out2], / [[, out, …])	Return element-wise quotient and remainder simultaneously.
heaviside(x1, x2, /[, out, where, casting, …])	Compute the Heaviside step function.
gcd(x1, x2, /[, out, where, casting, order, …])	Returns the greatest common divisor of |x1| and |x2|
lcm(x1, x2, /[, out, where, casting, order, …])	Returns the lowest common multiple of |x1| and |x2|

"""

# ----------------------------------------------------------------------------------------------------------------------
# non ufuncs
# ----------------------------------------------------------------------------------------------------------------------

REDUCE_KEEPDIMS_METHODS = [
    'max',
    'min',
    'amax',
    'amin',
    'round',
    'around',
    'clip',
    'cumsum',
]

REDUCE_KEEPUNITS_METHODS = [
    'sum',
    'mean',
    'std',
    'ptp',
]

REDUCE_METHODS = [
    'all',
    'any',
    'argmax',
    'argmin',
]


# test if a method is implemented

@pytest.mark.parametrize('name', REDUCE_METHODS +
                         REDUCE_KEEPDIMS_METHODS +
                         REDUCE_KEEPUNITS_METHODS)
def test_ndmath_classmethod_implementation(nd2d, name):
    nd = nd2d.copy()
    try:
        method = getattr(NDDataset, name)
    except AttributeError:
        info_('\n{} is not yet implemented'.format(name))
    try:
        op = getattr(np.ma, name)
        x = getattr(np, name)(nd)
    except AttributeError:
        info_('\n{} is not a np.ma method'.format(name))
    except TypeError as e:
        if 'required positional' in e.args[0]:
            pass
        else:
            raise TypeError(*e.args)


@pytest.mark.parametrize(('operation', 'restype', 'args', 'kwargs'),
                         [('amax', NDDataset, None, {}),
                          ('amin', NDDataset, None, {}),
                          ('max', NDDataset, None, {}),
                          ('min', NDDataset, None, {}),
                          ('sum', (TYPE_FLOAT, Quantity), None, {}),
                          ('cumsum', NDDataset, None, {}),
                          ('cumsum', NDDataset, None, {'dim': 1}),
                          ('ptp', (TYPE_FLOAT, Quantity), None, {}),
                          ('all', np.bool_, None, {}),
                          ('any', np.bool_, None, {}),
                          ('mean', (TYPE_FLOAT, Quantity), None, {}),
                          ('std', (TYPE_FLOAT, Quantity), None, {}),
                          ('argmax', (TYPE_INTEGER, tuple), None, {}),
                          ('argmin', (TYPE_INTEGER, tuple), None, {}),
                          ('clip', NDDataset, (2., 5.), {}),
                          ('around', NDDataset, (-1,), {}),
                          ('round', NDDataset, (-1,), {}),

                          ('max', (TYPE_FLOAT, Quantity), None, {'keepdims': False}),

                          ])
def test_ndmath_non_ufunc_functions_with_masked(operation, restype, args, kwargs):
    def runop(a, args, kwargs):
        
        if not isinstance(a, NDDataset) and 'dim' in kwargs:  # dim not accepted by np
            kwargs['axis'] = kwargs.pop('dim')
        elif isinstance(a, NDDataset) and 'axis' in kwargs and operation != 'cumsum':
            kwargs['dim'] = kwargs.pop('axis')
        
        if args:
            if kwargs:
                res = op(a, *args, **kwargs)
            else:
                res = op(a, *args)
        else:
            if kwargs:
                res = op(a, **kwargs)
            else:
                res = op(a)
        return res
    
    if not isinstance(restype, tuple):
        restype = (restype, restype)
    
    op = getattr(np, operation)
    # info_(op.__doc__)
    
    # with 2D
    d2D = np.ma.array([[3, -1., 2, 4, 10],
                       [5, 60, 8, -7.7, 0],
                       [3.1, -1.5, 2.5, 4.3, 10.],
                       [1, 6., 8.5, 77., -200.]])
    d2D[3, 3] = np.ma.masked
    
    result = runop(d2D, args, kwargs)
    if 'arg' in operation:
        result = np.unravel_index(result, d2D.shape)
    
    ds2 = NDDataset(d2D, dtype='float64')
    coord0 = Coord(np.arange(4) * .1)
    coord1 = Coord(np.arange(5) * .2)
    ds2.set_coords(coord0, coord1)
    ds2.units = ur.m
    
    dsy = runop(ds2, args, kwargs)
    info_(ds2)
    info_("result operation {}: {}".format(operation, str(dsy)))
    
    assert isinstance(dsy, restype[1])
    if isinstance(dsy, NDDataset):
        assert_array_almost_equal(dsy.data, result)
        assert dsy.units == ds2.units
    elif isinstance(dsy, Quantity):
        assert_array_almost_equal(np.array(dsy.m), np.array(result))
        assert dsy.units == ds2.units
    else:
        assert_array_almost_equal(np.array(dsy), np.array(result))
    
    # with 1D
    
    d1D = np.ma.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      mask=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    
    kwargs.pop('dim', None)
    kwargs.pop('dims', None)
    kwargs.pop('axis', None)
    
    result = runop(d1D, args, kwargs)
    
    ds1 = NDDataset(d1D, dtype='float64')
    coord0 = Coord(np.arange(10) * .1)
    ds1.set_coords(coord0, )
    dsy = runop(ds1, args, kwargs)
    
    info_(str(ds1))
    info_("result operation {}: {}".format(operation, str(dsy)))
    
    assert isinstance(dsy, restype[0])
    if isinstance(dsy, NDDataset):
        assert_array_almost_equal(dsy.data, result)
    else:
        assert_approx_equal(dsy, result)


def test_ndmath_reduce_quantity(IR_dataset_2D):
    # must keep units
    ds = IR_dataset_2D.copy()
    s = ds.sum()
    assert ds.units == s.units


def test_ndmath_absolute_of_complex():
    ndd = NDDataset([1., 2. + 1j, 3.])
    
    val = np.abs(ndd)
    print(val)
    
    val = ndd[1] * 1.2 - 10.
    
    val = np.abs(val)
    print(val)


def test_ndmath_absolute_of_quaternion():
    na0 = np.array([[1., 2., 2., 0., 0., 0.],
                    [1.3, 2., 2., 0.5, 1., 1.],
                    [1, 4.2, 2., 3., 2., 2.],
                    [5., 4.2, 2., 3., 3., 3.]])
    nd = NDDataset(na0, dtype=quaternion)
    print(nd)
    coords = CoordSet(np.linspace(-1, 1, 2), np.linspace(-10., 10., 3))
    assert nd.shape == (2, 3)
    nd.set_coords(**coords)
    val = np.abs(nd)
    
    # TODO: add more testings


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
    d1 = NDDataset(np.ones((5, 5)), name='d1')
    d2 = NDDataset(np.ones((5, 5)), name='d2')
    d3 = -d1
    assert d3.name != d1
    
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
    d1 = NDDataset(np.ones((5, 5)), coords=[coord1, coord2])
    d2 = NDDataset(np.ones((5, 5)), coords=[coord2, coord1])
    with pytest.raises(ValueError) as exc:
        d1 -= d2
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
    assert str(exc.value).startswith("Cannot convert from '[length]' to '[length] ** 2', "
                                     "Units must be compatible for the `add` operator")
    
    with pytest.raises(DimensionalityError) as exc:
        d1 += d2
    assert str(exc.value).startswith("Cannot convert from '[length]' to '[length] ** 2', "
                                     "Units must be compatible for the `iadd` operator")


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
    d1 = NDDataset(np.ones((5, 5)), coords=[coord1, coord2])
    d2 = NDDataset(np.ones((5, 5)), coords=[coord2, coord1])
    with pytest.raises(ValueError) as exc:
        d1 -= d2
    assert exc.value.args[0] == "coordinate's values do not match"


def test_nddataset_binary_operation_with_other_1D():
    coord1 = Coord(np.linspace(0., 10., 10))
    coord2 = Coord(np.linspace(1., 5.5, 5))
    d1 = NDDataset(np.random.random((10, 5)), coords=[coord1, coord2])
    d2 = d1[0]
    # this should work independantly of the value of the coordinates on dimension y
    d3 = d1 - d2
    assert_array_equal(d3.data, d1.data - d2.data)


def test_nddataset_subtract_mismatch_units():
    d1 = NDDataset(np.ones((5, 5)), units='m')
    d2 = NDDataset(np.ones((5, 5)) * 2., units='m/s')
    with pytest.raises(DimensionalityError) as exc:
        d1 -= d2
    assert str(exc.value) == "Cannot convert from '[length] / [time]' to '[length]', " \
                                  "Units must be compatible for the `isub` operator"


def test_nddataset_subtract_mismatch_shape():
    d1 = NDDataset(np.ones((5, 5)))
    d2 = NDDataset(np.ones((6, 6)) * 2.)
    with pytest.raises(ArithmeticError) as exc:
        d1 -= d2
    assert exc.value.args[0].startswith(
        "operands could not be broadcast together")


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
def test_ndmath_unit_conversion_operators(operation, result_units):
    in_km = NDDataset(np.array([1, 1]), units=ur.km,
                      uncertainty=np.array([.1, .1]))
    in_m = NDDataset(in_km.data * 1000, units=ur.m)
    operator_km = in_km.__getattribute__(operation)
    combined = operator_km(in_m)
    assert_equal_units(combined.units, result_units)


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


# ----------------------------------------------------------------------------------------------------------------------
# additional tests made following some bug fixes
# ----------------------------------------------------------------------------------------------------------------------

def test_simple_arithmetic_on_full_dataset():
    # due to a bug in notebook with the following
    import os
    dataset = NDDataset.read_omnic(os.path.join('irdata',
                                                'nh4y-activation.spg'))
    d = dataset - dataset[0]
    # suppress the first spectrum to all other spectra in the series
