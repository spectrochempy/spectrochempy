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
from spectrochempy.core.dataset.ndcoords import Coord, CoordSet
from spectrochempy.units.units import ur, Quantity
from spectrochempy.utils import (MASKED, TYPE_FLOAT, TYPE_INTEGER,
                                 TYPE_COMPLEX)
from spectrochempy.core import log

from pint.errors import (UndefinedUnitError,
                                              DimensionalityError)
from spectrochempy.utils import Meta, SpectroChemPyWarning, info_
from spectrochempy.utils.testing import (assert_equal, assert_array_equal,
                                         assert_array_almost_equal, assert_equal_units,
                                         raises, assert_approx_equal)
from spectrochempy.utils.testing import RandomSeedContext
import numpy as np

from quaternion import quaternion

typequaternion = np.dtype(np.quaternion)

# ----------------------------------------------------------------------------------------------------------------------
# ufuncs
# ----------------------------------------------------------------------------------------------------------------------

ufunc_str = """

#Math operations

add(x1, x2, /[, out, where, casting, order, …])	Add arguments element-wise.
subtract(x1, x2, /[, out, where, casting, …])	Subtract arguments, element-wise.
multiply(x1, x2, /[, out, where, casting, …])	Multiply arguments element-wise.
divide(x1, x2, /[, out, where, casting, …])	Returns a true division of the inputs, element-wise.
logaddexp(x1, x2, /[, out, where, casting, …])	Logarithm of the sum of exponentiations of the inputs.
logaddexp2(x1, x2, /[, out, where, casting, …])	Logarithm of the sum of exponentiations of the inputs in base-2.
true_divide(x1, x2, /[, out, where, …])	Returns a true division of the inputs, element-wise.
floor_divide(x1, x2, /[, out, where, …])	Return the largest integer smaller or equal to the division of the inputs.
negative(x, /[, out, where, casting, order, …])	Numerical negative, element-wise.
positive(x, /[, out, where, casting, order, …])	Numerical positive, element-wise.
power(x1, x2, /[, out, where, casting, …])	First array elements raised to powers from second array, element-wise.
remainder(x1, x2, /[, out, where, casting, …])	Return element-wise remainder of division.
mod(x1, x2, /[, out, where, casting, order, …])	Return element-wise remainder of division.
fmod(x1, x2, /[, out, where, casting, …])	Return the element-wise remainder of division.
divmod(x1, x2[, out1, out2], / [[, out, …])	Return element-wise quotient and remainder simultaneously.
absolute(x, /[, out, where, casting, order, …])	Calculate the absolute value element-wise.
fabs(x, /[, out, where, casting, order, …])	Compute the absolute values element-wise.
rint(x, /[, out, where, casting, order, …])	Round elements of the array to the nearest integer.
sign(x, /[, out, where, casting, order, …])	Returns an element-wise indication of the sign of a number.
heaviside(x1, x2, /[, out, where, casting, …])	Compute the Heaviside step function.
conj(x, /[, out, where, casting, order, …])	Return the complex conjugate, element-wise.
exp(x, /[, out, where, casting, order, …])	Calculate the exponential of all elements in the input array.
exp2(x, /[, out, where, casting, order, …])	Calculate 2**p for all p in the input array.
log(x, /[, out, where, casting, order, …])	Natural logarithm, element-wise.
log2(x, /[, out, where, casting, order, …])	Base-2 logarithm of x.
log10(x, /[, out, where, casting, order, …])	Return the base 10 logarithm of the input array, element-wise.
expm1(x, /[, out, where, casting, order, …])	Calculate exp(x) - 1 for all elements in the array.
log1p(x, /[, out, where, casting, order, …])	Return the natural logarithm of one plus the input array, element-wise.
sqrt(x, /[, out, where, casting, order, …])	Return the non-negative square-root of an array, element-wise.
square(x, /[, out, where, casting, order, …])	Return the element-wise square of the input.
cbrt(x, /[, out, where, casting, order, …])	Return the cube-root of an array, element-wise.
reciprocal(x, /[, out, where, casting, …])	Return the reciprocal of the argument, element-wise.
gcd(x1, x2, /[, out, where, casting, order, …])	Returns the greatest common divisor of |x1| and |x2|
lcm(x1, x2, /[, out, where, casting, order, …])	Returns the lowest common multiple of |x1| and |x2|

# Trigonometric functions

sin(x, /[, out, where, casting, order, …])	Trigonometric sine, element-wise.
cos(x, /[, out, where, casting, order, …])	Cosine element-wise.
tan(x, /[, out, where, casting, order, …])	Compute tangent element-wise.
arcsin(x, /[, out, where, casting, order, …])	Inverse sine, element-wise.
arccos(x, /[, out, where, casting, order, …])	Trigonometric inverse cosine, element-wise.
arctan(x, /[, out, where, casting, order, …])	Trigonometric inverse tangent, element-wise.
arctan2(x1, x2, /[, out, where, casting, …])	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
hypot(x1, x2, /[, out, where, casting, …])	Given the “legs” of a right triangle, return its hypotenuse.
sinh(x, /[, out, where, casting, order, …])	Hyperbolic sine, element-wise.
cosh(x, /[, out, where, casting, order, …])	Hyperbolic cosine, element-wise.
tanh(x, /[, out, where, casting, order, …])	Compute hyperbolic tangent element-wise.
arcsinh(x, /[, out, where, casting, order, …])	Inverse hyperbolic sine element-wise.
arccosh(x, /[, out, where, casting, order, …])	Inverse hyperbolic cosine, element-wise.
arctanh(x, /[, out, where, casting, order, …])	Inverse hyperbolic tangent element-wise.
deg2rad(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
rad2deg(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.

# Comparison functions

greater(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 > x2) element-wise.
greater_equal(x1, x2, /[, out, where, …])	Return the truth value of (x1 >= x2) element-wise.
less(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 < x2) element-wise.
less_equal(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 =< x2) element-wise.
not_equal(x1, x2, /[, out, where, casting, …])	Return (x1 != x2) element-wise.
equal(x1, x2, /[, out, where, casting, …])	Return (x1 == x2) element-wise.
logical_and(x1, x2, /[, out, where, …])	Compute the truth value of x1 AND x2 element-wise.
logical_or(x1, x2, /[, out, where, casting, …])	Compute the truth value of x1 OR x2 element-wise.
logical_xor(x1, x2, /[, out, where, …])	Compute the truth value of x1 XOR x2, element-wise.
logical_not(x, /[, out, where, casting, …])	Compute the truth value of NOT x element-wise.
maximum(x1, x2, /[, out, where, casting, …])	Element-wise maximum of array elements.
minimum(x1, x2, /[, out, where, casting, …])	Element-wise minimum of array elements.
fmax(x1, x2, /[, out, where, casting, …])	Element-wise maximum of array elements.
fmin(x1, x2, /[, out, where, casting, …])	Element-wise minimum of array elements.

# Floating functions

isfinite(x, /[, out, where, casting, order, …])	Test element-wise for finiteness (not infinity or not Not a Number).
isinf(x, /[, out, where, casting, order, …])	Test element-wise for positive or negative infinity.
isnan(x, /[, out, where, casting, order, …])	Test element-wise for NaN and return result as a boolean array.
#isnat(x, /[, out, where, casting, order, …])	Test element-wise for NaT (not a time) and return result as a boolean array.
fabs(x, /[, out, where, casting, order, …])	Compute the absolute values element-wise.
signbit(x, /[, out, where, casting, order, …])	Returns element-wise True where signbit is set (less than zero).
copysign(x1, x2, /[, out, where, casting, …])	Change the sign of x1 to that of x2, element-wise.
nextafter(x1, x2, /[, out, where, casting, …])	Return the next floating-point value after x1 towards x2, element-wise.
spacing(x, /[, out, where, casting, order, …])	Return the distance between x and the nearest adjacent number.
modf(x[, out1, out2], / [[, out, where, …])	Return the fractional and integral parts of an array, element-wise.
ldexp(x1, x2, /[, out, where, casting, …])	Returns x1 * 2**x2, element-wise.
frexp(x[, out1, out2], / [[, out, where, …])	Decompose the elements of x into mantissa and twos exponent.
fmod(x1, x2, /[, out, where, casting, …])	Return the element-wise remainder of division.
floor(x, /[, out, where, casting, order, …])	Return the floor of the input, element-wise.
ceil(x, /[, out, where, casting, order, …])	Return the ceiling of the input, element-wise.
trunc(x, /[, out, where, casting, order, …])	Return the truncated value of the input, element-wise.

"""

def liste_ufunc():
    liste = ufunc_str.split("\n")
    ufuncs = []
    for item in liste:
        item = item.strip()
        if not item:
            continue
        if item.startswith('#'):
            continue
        item = item.split('(')
        unary = False
        if item[1].startswith('x, /['):
            unary = True
        ufuncs.append((item[0], unary ,item[1]))
    return ufuncs


UNARY_MATH = [(a[0],a[2]) for a in liste_ufunc() if a[1]]
print(UNARY_MATH)

@pytest.mark.parametrize(('name','comment'), UNARY_MATH)
def test_ndmath_unary_ufuncs_simple_data(nd2d, name, comment):

    nd1 = nd2d.copy()
    info_("{}   # {}".format(name, comment))

    # simple
    info_(nd1)
    assert nd1.unitless
    f = getattr(np, name)
    r = f(nd1)
    info_(r)
    assert isinstance(r, NDDataset)

    # with units
    nd1.units = ur.m
    info_(nd1)
    f = getattr(np, name)
    nounit= False
    try:
        r = f(nd1)
        info_(str(r))
        assert isinstance(r, NDDataset)
        # TODO: some ufunc suppress the units! see pint.
        try:
            expected_units = f(Quantity(1.,nd1.units)).units
            assert r.units == expected_units
        except AttributeError:
            log.error(" =======> {} remove units! ".format(name))

    except DimensionalityError:
        log.error("{} : Cannot convert from 'meter' ([length]) to 'dimensionless' (dimensionless)".format(name))
        nounit = True

    nd1 = nd2d.copy() # reset nd

    # with units and mask
    nd1[1,1]=MASKED
    info_(nd1)
    f = getattr(np, name)
    r = f(nd1)
    info_(r)
    assert isinstance(r, NDDataset)


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

def test_ndmath_ufunc_method(nd2d):
    nd = nd2d.copy()
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
                          ('sum', (TYPE_FLOAT,Quantity), None, {}),
                          ('cumsum', NDDataset, None, {}),
                          ('cumsum', NDDataset, None, {'dim':1}),
                          ('ptp', (TYPE_FLOAT,Quantity), None, {}),
                          ('all', np.bool_, None, {}),
                          ('any', np.bool_, None, {}),
                          ('mean', (TYPE_FLOAT,Quantity), None, {}),
                          ('std', (TYPE_FLOAT,Quantity), None, {}),
                          ('argmax', (TYPE_INTEGER,tuple), None, {}),
                          ('argmin', (TYPE_INTEGER,tuple), None, {}),
                          ('clip', NDDataset, (2., 5.), {}),
                          ('around', NDDataset, (-1,), {}),
                          ('round', NDDataset, (-1,), {}),

                          ('max', (TYPE_FLOAT, Quantity), None, {'keepdims':False}),

                          ])
def test_ndmath_non_ufunc_functions_with_masked(operation, restype, args, kwargs):


    def runop(a, args, kwargs):

        if not isinstance(a, NDDataset) and 'dim' in kwargs: # dim not accepted by np
            kwargs['axis']=kwargs.pop('dim')
        elif isinstance(a, NDDataset) and 'axis' in kwargs and operation!='cumsum':
            kwargs['dim']=kwargs.pop('axis')

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
        restype=(restype, restype)


    op = getattr(np, operation)
    #info_(op.__doc__)

    # with 2D
    d2D = np.ma.array([[3,   -1.,  2,   4,    10   ],
                       [5,   60,    8,   -7.7, 0    ],
                       [3.1, -1.5, 2.5, 4.3,  10.  ],
                       [1,   6.,   8.5, 77.,  -200.]])
    d2D[3,3] = np.ma.masked

    result = runop(d2D, args, kwargs)
    if 'arg' in operation:
            result = np.unravel_index(result, d2D.shape)

    ds2 = NDDataset(d2D, dtype='float64')
    coord0 = Coord(np.arange(4) * .1)
    coord1 = Coord(np.arange(5) * .2)
    ds2.coords = [coord0, coord1]
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
    ds1.coords = [coord0]
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
    coords = CoordSet([np.linspace(-1, 1, 2), np.linspace(-10., 10., 3)])
    assert nd.shape == (2, 3)
    nd.coords = coords
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
    assert exc.value.__str__() == "Cannot convert from 'meter' ([length]) " \
                                  "to 'meter / second' ([length] / [time])"

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
