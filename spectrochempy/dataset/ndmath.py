# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module implements the |NDMath| class.

"""

__all__ = ['NDMath', ]

__dataset_methods__ = []

# =============================================================================
# Standard python imports
# =============================================================================
import copy
import functools

# =============================================================================
# third-party imports
# =============================================================================
import numpy as np
from numpy.ma import MaskedArray

# =============================================================================
# Local imports
# =============================================================================
from ..extern.uncertainties import unumpy as unp
from ..units.units import Quantity
from .ndarray import NDArray
from ..utils import (interleave, interleaved2complex, getdocfrom,
                                 docstrings)
from spectrochempy.application import log


# =============================================================================
# utility
# =============================================================================

get_name = lambda x: str(x.name if hasattr(x, 'name') else x)

class NDMath(object):
    """
    This class provides the math functionality to |NDArray| or |Coord|.

    Below is a list of mathematical functions (numpy) implemented (or
    planned for implementation)

    Ufuncs
    ------

    These functions should work like for numpy-ndarray, except that they
    may be units-aware.

    For instance, `ds`  being a |NDDataset|, just call the np functions like
    this. Most of the time it returns a new NDDataset, while in some cases
    noted below, one get a |ndarray|.

    >>> np.sin(ds)

    In this particular case (*i.e.*, `np.sin` ufuncs) , the `ds` units must be
    `unitless`, `dimensionless` or angle-units : `radians` or `degrees`,
    or an exception will be raised.


    Examples
    --------

    >>> from spectrochempy import *
    >>> dataset = NDDataset.load('mydataset.scp')
    >>> dataset             #doctest: +ELLIPSIS
    NDDataset([[    2.06,...,     1.24]])
    >>> np.negative(dataset) #doctest: +ELLIPSIS
    NDDataset([[   -2.06,...,    -1.24]])


    """

    @property
    def __array_struct__(self):
        return self._data.__array_struct__

    # -------------------------------------------------------------------------
    # public methods
    # -------------------------------------------------------------------------

    def pipe(self, func, *args, **kwargs):
        """Apply func(self, \*args, \*\*kwargs)

        Parameters
        ----------
        func : function
            function to apply to the |NDDataset|.
            `\*args`, and `\*\*kwargs` are passed into `func`.
            Alternatively a `(callable, data_keyword)` tuple where
            `data_keyword` is a string indicating the keyword of
            `callable` that expects the array object.
        *args : positional arguments passed into `func`.
        **kwargs : keyword arguments passed into `func`.

        Returns
        -------
        object : the return type of `func`.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        a |NDDataset|.

        """
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise TypeError('%s is both the pipe target and a keyword '
                                 'argument' % target)
            kwargs[target] = self
            return func(*args, **kwargs)

        return func(self, *args, **kwargs)

    # .........................................................................
    @docstrings.dedent
    def abs(self, axis=-1, inplace=False):
        """
        Returns the absolute value of a complex array.

        Parameters
        ----------
        %(generic_method.parameters.axis)s
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        """
        new = self.copy()
        if not new.has_complex_dims or not new.is_complex[axis]:
            return np.fabs(new)  # not a complex, return fabs should be faster

        new.swapaxes(axis, -1, inplace=True)
        new = np.sqrt(new.real ** 2 + new.imag ** 2)
        new.swapaxes(axis, -1, inplace=True)
        new._is_complex[axis] = False

        if inplace:
            self = new

        return new

    absolute = abs


    # numpy functions which are not ufuncs
    #TODO: implement uncertainties!

    # sum, products...
    # ------------------------------------------------------------------------

    # ........................................................................
    @getdocfrom(np.sum)
    def sum(self, *args, **kwargs):
        """sum along axis"""

        new = self.copy()
        ma = np.sum(new._masked_data, *args, **kwargs)
        axis = kwargs.get('axis', None)
        if axis is None:
            return ma
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        # the data being reduce to only a single elements along the summed axis
        # we must reduce the corresponding coordinates
        new.coordset[axis]= None
        return new

    @getdocfrom(np.prod)
    def prod(self, *args, **kwargs):
        """product along axis"""

        new = self.copy()
        ma = np.prod(new._masked_data, *args, **kwargs)
        axis = kwargs.get('axis', None)
        if axis is None:
            return ma
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        # the data being reduce to only a single elements along the summed axis
        # we must reduce the corresponding coordinates
        new.coordset[axis]= None
        return new

    @getdocfrom(np.cumsum)
    def cumsum(self, *args, **kwargs):
        """umsum along axis"""

        new = self.copy()
        ma = np.cumsum(new._masked_data, *args, **kwargs)
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    @getdocfrom(np.cumprod)
    def cumprod(self, *args, **kwargs):
        """cumprod along axis"""

        new = self.copy()
        ma = np.cumprod(new._masked_data, *args, **kwargs)
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    # statistics
    # ------------------------------------------------------------------------
    @getdocfrom(np.mean)
    def mean(self, *args, **kwargs):
        """mean values along axis"""

        new = self.copy()
        ma = np.mean(new._masked_data, *args, **kwargs)
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    @getdocfrom(np.std)
    def std(self, *args, **kwargs):
        """Standard deviation values along axis"""

        new = self.copy()
        ma = np.std(new._masked_data, *args, **kwargs)
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    # utilities

    @getdocfrom(np.ptp)
    def ptp(self, *args, **kwargs):
        """amplitude of data along axis"""

        new = self.copy()
        ma = np.ptp(new._masked_data, *args, **kwargs)
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    @getdocfrom(np.min)
    def min(self, *args, **kwargs):
        """minimum of data along axis"""

        new = self.copy()
        ma = np.min(new._masked_data, *args, **kwargs)
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    @getdocfrom(np.max)
    def max(self, *args, **kwargs):
        """maximum of data along axis"""

        new = self.copy()
        ma = np.max(new._masked_data, *args, **kwargs)
        if isinstance(ma, MaskedArray):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma
        return new

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    # the following methods are to give NDArray based class
    # a behavior similar to np.ndarray regarding the ufuncs

    def __array_wrap__(self, *args):
        # called when element-wise ufuncs are applied to the array

        f, objs, huh = args[1]

        # case of complex data
        if self.has_complex_dims:

            if self._is_complex[-1] and \
                            f.__name__ in ['real', 'imag',
                                           'conjugate', 'absolute',
                                           'conj', 'abs']:
                return getattr(objs[0], f.__name__)()

            if self._is_complex[-1] and f.__name__ in ["fabs", ]:
                # fonction not available for complex data
                raise ValueError("{} does not accept complex data ".format(f))

        # not a complex data
        if f.__name__ in ['absolute', 'abs']:
            f = np.fabs

        data, uncertainty, units, mask, iscomplex = self._op(f, objs,
                                                             ufunc=True)
        history = 'ufunc %s applied.' % f.__name__

        return self._op_result(data, uncertainty, units, mask, history,
                               iscomplex)

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _op(f, objs, ufunc=False):
        # achieve an operation f on the objs

        fname = f.__name__  # name of the function to use
        objs = list(objs)  # work with a list of objs not tuples

        # determine if the function needs compatible units
        sameunits = False
        if fname in ['lt', 'le', 'ge', 'gt', 'add', 'sub']:
            sameunits = True

        # take the first object out of the objs list
        obj = copy.deepcopy(objs.pop(0))

        # Some flags to be set depending of the object
        isdataset = True
        iscomplex = False

        # this might be a limit of the program
        # but we will consider using uncertainty only if
        # the main object has uncertainty only
        isuncertain = False

        objcomplex = []  # to keep track of the complex nature of the obj
        # in the other dimensions tahn the last

        # case our first object is a NDArray
        # (Coord or NDDataset derive from NDArray)
        if isinstance(obj, NDArray):

            d = obj._data  # The underlying data

            # do we have units?
            if obj.has_units:
                q = Quantity(1., obj.units)  # create a Quantity from the units
            else:
                q = 1.

            if obj.is_uncertain:
                isuncertain = True

            # Check if our NDArray is actually a NDDataset
            # (it must have an attribute _coordset)
            if hasattr(obj, '_coordset'):

                # do we have uncertainties on our data ?
                # if any create an UFloat type if any uncertainty
                # else we use the faster numpy standard array
                if isuncertain:
                    d = obj._uarray(d, obj._uncertainty)

                # Our data may be complex
                iscomplex = False
                if obj.has_complex_dims:
                    iscomplex = obj._is_complex[-1]

                objcomplex.append(obj._is_complex)

            else:

                # Ok it's an NDArray but not a NDDataset, then it's an Coord.
                isdataset = False

            # mask?
            d = obj._umasked(d, obj._mask)

            if iscomplex:
                # pack to complex
                d = interleaved2complex(d)

        else:

            # obj is not a NDDarray
            # assume an array or a scalar (possibly a Quantity)
            isdataset = False

            if hasattr(obj, 'units'):
                if not obj.dimensionless:
                    q = Quantity(1.,
                                 obj.units)  # create a Quantity from the units
                else:
                    q = 1.
                d = obj.magnitude
            else:
                q = 1.
                d = obj

        # Now we analyse the other operands
        args = []
        argunits = []

        # TODO: check the units with respect to some ufuncs or ops
        for o in objs:
            other = copy.deepcopy(o)

            # is other a NDDataset or Coord?
            if isinstance(other, NDArray):

                # if the first arg (obj) is a nddataset
                if isdataset and other._coordset != obj._coordset:
                    # here it can be several situations
                    # One acceptable is that e.g., we suppress or add
                    # a row to the whole dataset
                    for i, (s1,s2) in enumerate(
                            zip(obj._data.shape, other._data.shape)):
                        # we obviously have to work on the real shapes
                        if s1!=1 and s2!=1:
                            if s1!=s2:
                                raise ValueError(
                                        "coordinate's sizes do not match")
                            elif not np.all(obj._coordset[i]._data==
                                      other._coordset[i]._data):
                                raise ValueError(
                                        "coordinate's values do not match")


                # rescale according to units
                if not other.unitless:
                    if hasattr(obj, 'units'):  # obj is a Quantity
                        if sameunits:
                            other.to(obj._units,
                                     inplace=True)  # must also rescale uncertainty
                        argunits.append(Quantity(1., other._units))
                    else:
                        argunits.append(1.)
                else:
                    argunits.append(1.)

                arg = other._data
                if isuncertain:
                    # whatever the existence of
                    # uncertainty here if the main object
                    # was not uncertain, we will not take into
                    # account those uncertainty
                    arg = other._uarray(arg, other._uncertainty)

                # mask?
                arg = other._umasked(arg, other._mask)

                # complex?
                if hasattr(other, '_is_complex') and \
                                other._is_complex is not None:
                    if other._is_complex[-1]:
                        # pack arg to complex
                        arg = interleaved2complex(arg)

                    objcomplex.append(other._is_complex)

            else:
                # Not a NDArray.
                # separate units and magnitude
                if isinstance(other, Quantity):
                    arg = other.magnitude
                    argunits.append(Quantity(1., other._units))
                else:
                    arg = other
                    argunits.append(1.)

            args.append(arg)

        # perform operations
        # ------------------
        if ufunc and isuncertain:
            # with use of the numpy functions of uncertainty package
            # some transformation to handle missing function in the unumpy module
            if fname == 'exp2':
                data = unp.exp(d * np.log(2.))
            elif fname == 'log2':
                data = unp.log(d) / np.log(2.)
            elif fname == 'negative':
                data = -d
            elif fname == 'reciprocal':
                data = 1. / d
            elif fname == 'square':
                data = d ** 2
            elif fname == 'rint':
                data = obj._uarray(np.rint(unp.nominal_values(d)),
                                   unp.std_devs(d))
            elif fname == 'sign':
                data = np.sign(unp.nominal_values(d))
            elif fname == 'isfinite':
                data = np.isfinite(unp.nominal_values(d))
            else:
                data = getattr(unp, fname)(d, *args)

        elif ufunc:
            # if not uncertain use the numpy package, not the unp
            data = getattr(np, fname)(d, *args)


                # TODO: check the complex nature of the result to return it

        else:
            # make a simple operation
            try:
                data = f(d, *args)
            except Exception as e:
                raise ArithmeticError(e.args[0])

            # restore interleaving of complex data
            data, iscomplex = interleave(data)

        # unpack the data (this process is long, so we bypass it if not needed)
        if isuncertain:
            uncertainty = unp.std_devs(data)
            data = unp.nominal_values(data).astype(float)
        else:
            uncertainty = None

        # get possible mask
        if isinstance(data, np.ma.MaskedArray):
            mask = data._mask
            data = data._data
        else:
            mask = np.zeros_like(data, dtype=bool)

        # redo the calculation with the units to found the final one
        q = f(q, *argunits)
        if hasattr(q, 'units'):
            units = q.units
        else:
            units = None

        # determine the is_complex parameter:
        data_iscomplex = [False] * data.ndim

        if iscomplex:
            # the resulting data are complex on the last dimension
            data_iscomplex[-1] = True

        # For the other dimension, this will depends on the history of the
        # objs:
        # TODO: The following will have to be carefully checked in many kind
        # of situation
        for i in range(data.ndim)[:-1]:

            for item in objcomplex:
                # dim is complex for this object
                # (should be also the case of the results)
                # of course this will work only if the array
                # doesn't change in ndim ...
                # TODO: is that possible? - To check
                # this also assume that compatible object have been
                # passed. If it is not the case,
                # some adaptation will be necessary
                # TODO: adapt array if necessary
                # for complex dimension
                if item:
                    data_iscomplex[i] |= item[i]  # `or` operation

        return data, uncertainty, units, mask, data_iscomplex


    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self):
            data, uncertainty, units, mask, iscomplex = self._op(f, [self])
            if hasattr(self, 'history'):
                history = 'unary operation %s applied' % f.__name__
            return self._op_result(data,
                                   uncertainty, units, mask, history, iscomplex)

        return func

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            if not reflexive:
                objs = [self, other]
            else:
                objs = [other, self]
            data, uncertainty, units, mask, iscomplex = self._op(f, objs)
            if hasattr(self, 'history'):
                history = 'binary operation ' + f.__name__ + \
                          ' with `%s` has been performed' % get_name(other)
            else:
                history = None
            return self._op_result(data, uncertainty, units, mask, history,
                                   iscomplex)

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            objs = [self, other]
            data, uncertainty, units, mask, iscomplex = self._op(f, objs)
            self._data = data
            self._uncertainty = uncertainty
            self._units = units
            self._mask = mask
            self._iscomplex = iscomplex

            self.history = 'inplace binary op : ' + f.__name__ + \
            ' with %s ' % get_name(other)
            return self

        return func

    def _op_result(self, data, uncertainty=None, units=None,
                   mask=None, history=None, is_complex=None):
        # make a new NDArray resulting of some operation

        new = self.copy()

        # update the data
        new._data = copy.deepcopy(data)

        # update the attributes
        if uncertainty is not None:
            new._uncertainty = copy.deepcopy(uncertainty)
        if units is not None:
            new._units = copy.copy(units)
        if mask is not None:
            new._mask = copy.copy(mask)
        if history is not None and hasattr(new, 'history'):
            new._history.append(history.strip())
        if is_complex is not None:
            new._is_complex = is_complex

        return new


if __name__ == '__main__':

    pass


_s = """

To be continued
#################################################################
Math operations
----------------
add(x1, x2, /[, out, where, casting, order, ...])	Add arguments element-wise.
subtract(x1, x2, /[, out, where, casting, ...])	Subtract arguments, element-wise.
multiply(x1, x2, /[, out, where, casting, ...])	Multiply arguments element-wise.
divide(x1, x2, /[, out, where, casting, ...])	Divide arguments element-wise.
logaddexp(x1, x2, /[, out, where, casting, ...])	Logarithm of the sum of exponentiations of the inputs.
logaddexp2(x1, x2, /[, out, where, casting, ...])	Logarithm of the sum of exponentiations of the inputs in base-2.
true_divide(x1, x2, /[, out, where, ...])	Returns a true division of the inputs, element-wise.
floor_divide(x1, x2, /[, out, where, ...])	Return the largest integer smaller or equal to the division of the inputs.
negative(x, /[, out, where, casting, order, ...])	Numerical negative, element-wise.
positive(x, /[, out, where, casting, order, ...])	Numerical positive, element-wise.
power(x1, x2, /[, out, where, casting, ...])	First array elements raised to powers from second array, element-wise.
remainder(x1, x2, /[, out, where, casting, ...])	Return element-wise remainder of division.
mod(x1, x2, /[, out, where, casting, order, ...])	Return element-wise remainder of division.
fmod(x1, x2, /[, out, where, casting, ...])	Return the element-wise remainder of division.
divmod(x1, x2[, out1, out2], / [[, out, ...])	Return element-wise quotient and remainder simultaneously.
absolute(x, /[, out, where, casting, order, ...])	Calculate the absolute value element-wise.
fabs(x, /[, out, where, casting, order, ...])	Compute the absolute values element-wise.
rint(x, /[, out, where, casting, order, ...])	Round elements of the array to the nearest integer.
sign(x, /[, out, where, casting, order, ...])	Returns an element-wise indication of the sign of a number.
heaviside(x1, x2, /[, out, where, casting, ...])	Compute the Heaviside step function.
conj(x, /[, out, where, casting, order, ...])	Return the complex conjugate, element-wise.
exp(x, /[, out, where, casting, order, ...])	Calculate the exponential of all elements in the input array.
exp2(x, /[, out, where, casting, order, ...])	Calculate 2**p for all p in the input array.
log(x, /[, out, where, casting, order, ...])	Natural logarithm, element-wise.
log2(x, /[, out, where, casting, order, ...])	Base-2 logarithm of x.
log10(x, /[, out, where, casting, order, ...])	Return the base 10 logarithm of the input array, element-wise.
expm1(x, /[, out, where, casting, order, ...])	Calculate exp(x) - 1 for all elements in the array.
log1p(x, /[, out, where, casting, order, ...])	Return the natural logarithm of one plus the input array, element-wise.
sqrt(x, /[, out, where, casting, order, ...])	Return the positive square-root of an array, element-wise.
square(x, /[, out, where, casting, order, ...])	Return the element-wise square of the input.
cbrt(x, /[, out, where, casting, order, ...])	Return the cube-root of an array, element-wise.
reciprocal(x, /[, out, where, casting, ...])	Return the reciprocal of the argument, element-wise.
Tip
The optional output arguments can be used to help you save memory for large calculations. If your arrays are large, complicated expressions can take longer than absolutely necessary due to the creation and (later) destruction of temporary calculation spaces. For example, the expression G = a * b + c is equivalent to t1 = A * B; G = T1 + C; del t1. It will be more quickly executed as G = A * B; add(G, C, G) which is the same as G = A * B; G += C.

Trigonometric functions
All trigonometric functions use radians when an angle is called for. The ratio of degrees to radians is 180^{\circ}/\pi.

sin(x, /[, out, where, casting, order, ...])	Trigonometric sine, element-wise.
cos(x, /[, out, where, casting, order, ...])	Cosine element-wise.
tan(x, /[, out, where, casting, order, ...])	Compute tangent element-wise.
arcsin(x, /[, out, where, casting, order, ...])	Inverse sine, element-wise.
arccos(x, /[, out, where, casting, order, ...])	Trigonometric inverse cosine, element-wise.
arctan(x, /[, out, where, casting, order, ...])	Trigonometric inverse tangent, element-wise.
arctan2(x1, x2, /[, out, where, casting, ...])	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
hypot(x1, x2, /[, out, where, casting, ...])	Given the “legs” of a right triangle, return its hypotenuse.
sinh(x, /[, out, where, casting, order, ...])	Hyperbolic sine, element-wise.
cosh(x, /[, out, where, casting, order, ...])	Hyperbolic cosine, element-wise.
tanh(x, /[, out, where, casting, order, ...])	Compute hyperbolic tangent element-wise.
arcsinh(x, /[, out, where, casting, order, ...])	Inverse hyperbolic sine element-wise.
arccosh(x, /[, out, where, casting, order, ...])	Inverse hyperbolic cosine, element-wise.
arctanh(x, /[, out, where, casting, order, ...])	Inverse hyperbolic tangent element-wise.
deg2rad(x, /[, out, where, casting, order, ...])	Convert angles from degrees to radians.
rad2deg(x, /[, out, where, casting, order, ...])	Convert angles from radians to degrees.
Bit-twiddling functions
These function all require integer arguments and they manipulate the bit-pattern of those arguments.

bitwise_and(x1, x2, /[, out, where, ...])	Compute the bit-wise AND of two arrays element-wise.
bitwise_or(x1, x2, /[, out, where, casting, ...])	Compute the bit-wise OR of two arrays element-wise.
bitwise_xor(x1, x2, /[, out, where, ...])	Compute the bit-wise XOR of two arrays element-wise.
invert(x, /[, out, where, casting, order, ...])	Compute bit-wise inversion, or bit-wise NOT, element-wise.
left_shift(x1, x2, /[, out, where, casting, ...])	Shift the bits of an integer to the left.
right_shift(x1, x2, /[, out, where, ...])	Shift the bits of an integer to the right.
Comparison functions
greater(x1, x2, /[, out, where, casting, ...])	Return the truth value of (x1 > x2) element-wise.
greater_equal(x1, x2, /[, out, where, ...])	Return the truth value of (x1 >= x2) element-wise.
less(x1, x2, /[, out, where, casting, ...])	Return the truth value of (x1 < x2) element-wise.
less_equal(x1, x2, /[, out, where, casting, ...])	Return the truth value of (x1 =< x2) element-wise.
not_equal(x1, x2, /[, out, where, casting, ...])	Return (x1 != x2) element-wise.
equal(x1, x2, /[, out, where, casting, ...])	Return (x1 == x2) element-wise.
Warning
Do not use the Python keywords and and or to combine logical array expressions. These keywords will test the truth value of the entire array (not element-by-element as you might expect). Use the bitwise operators & and | instead.

logical_and(x1, x2, /[, out, where, ...])	Compute the truth value of x1 AND x2 element-wise.
logical_or(x1, x2, /[, out, where, casting, ...])	Compute the truth value of x1 OR x2 element-wise.
logical_xor(x1, x2, /[, out, where, ...])	Compute the truth value of x1 XOR x2, element-wise.
logical_not(x, /[, out, where, casting, ...])	Compute the truth value of NOT x element-wise.
Warning
The bit-wise operators & and | are the proper way to perform element-by-element array comparisons. Be sure you understand the operator precedence: (a > 2) & (a < 5) is the proper syntax because a > 2 & a < 5 will result in an error due to the fact that 2 & a is evaluated first.

maximum(x1, x2, /[, out, where, casting, ...])	Element-wise maximum of array elements.
Tip
The Python function max() will find the maximum over a one-dimensional array, but it will do so using a slower sequence interface. The reduce method of the maximum ufunc is much faster. Also, the max() method will not give answers you might expect for arrays with greater than one dimension. The reduce method of minimum also allows you to compute a total minimum over an array.

minimum(x1, x2, /[, out, where, casting, ...])	Element-wise minimum of array elements.
Warning
the behavior of maximum(a, b) is different than that of max(a, b). As a ufunc, maximum(a, b) performs an element-by-element comparison of a and b and chooses each element of the result according to which element in the two arrays is larger. In contrast, max(a, b) treats the objects a and b as a whole, looks at the (total) truth value of a > b and uses it to return either a or b (as a whole). A similar difference exists between minimum(a, b) and min(a, b).

fmax(x1, x2, /[, out, where, casting, ...])	Element-wise maximum of array elements.
fmin(x1, x2, /[, out, where, casting, ...])	Element-wise minimum of array elements.
Floating functions
Recall that all of these functions work element-by-element over an array, returning an array output. The description details only a single operation.

isfinite(x, /[, out, where, casting, order, ...])	Test element-wise for finiteness (not infinity or not Not a Number).
isinf(x, /[, out, where, casting, order, ...])	Test element-wise for positive or negative infinity.
isnan(x, /[, out, where, casting, order, ...])	Test element-wise for NaN and return result as a boolean array.
fabs(x, /[, out, where, casting, order, ...])	Compute the absolute values element-wise.
signbit(x, /[, out, where, casting, order, ...])	Returns element-wise True where signbit is set (less than zero).
copysign(x1, x2, /[, out, where, casting, ...])	Change the sign of x1 to that of x2, element-wise.
nextafter(x1, x2, /[, out, where, casting, ...])	Return the next floating-point value after x1 towards x2, element-wise.
spacing(x, /[, out, where, casting, order, ...])	Return the distance between x and the nearest adjacent number.
modf(x[, out1, out2], / [[, out, where, ...])	Return the fractional and integral parts of an array, element-wise.
ldexp(x1, x2, /[, out, where, casting, ...])	Returns x1 * 2**x2, element-wise.
frexp(x[, out1, out2], / [[, out, where, ...])	Decompose the elements of x into mantissa and twos exponent.
fmod(x1, x2, /[, out, where, casting, ...])	Return the element-wise remainder of division.
floor(x, /[, out, where, casting, order, ...])	Return the floor of the input, element-wise.
ceil(x, /[, out, where, casting, order, ...])	Return the ceiling of the input, element-wise.
trunc(x, /[, out, where, casting, order, ...])	Return the truncated value of the input, element-wise.


Trigonometric functions
------------------------

sin(x)	Trigonometric sine, element-wise. (Ufuncs - ONLY for
dimensionless or x in radians)
cos(x)	Cosine element-wise. (Ufuncs - ONLY for dimensionless or x in
radians)
tan(x)	Compute tangent element-wise.
arcsin(x, /[, out, where, casting, order, ...])	Inverse sine, element-wise.
arccos(x, /[, out, where, casting, order, ...])	Trigonometric inverse cosine, element-wise.
arctan(x, /[, out, where, casting, order, ...])	Trigonometric inverse tangent, element-wise.
hypot(x1, x2, /[, out, where, casting, ...])	Given the “legs” of a right triangle, return its hypotenuse.
arctan2(x1, x2, /[, out, where, casting, ...])	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
degrees(x, /[, out, where, casting, order, ...])	Convert angles from radians to degrees.
radians(x, /[, out, where, casting, order, ...])	Convert angles from degrees to radians.
unwrap(p[, discont, axis])	Unwrap by changing deltas between values to 2*pi complement.
deg2rad(x, /[, out, where, casting, order, ...])	Convert angles from degrees to radians.
rad2deg(x, /[, out, where, casting, order, ...])	Convert angles from radians to degrees.
Hyperbolic functions
sinh(x, /[, out, where, casting, order, ...])	Hyperbolic sine, element-wise.
cosh(x, /[, out, where, casting, order, ...])	Hyperbolic cosine, element-wise.
tanh(x, /[, out, where, casting, order, ...])	Compute hyperbolic tangent element-wise.
arcsinh(x, /[, out, where, casting, order, ...])	Inverse hyperbolic sine element-wise.
arccosh(x, /[, out, where, casting, order, ...])	Inverse hyperbolic cosine, element-wise.
arctanh(x, /[, out, where, casting, order, ...])	Inverse hyperbolic tangent element-wise.
Rounding
around(a[, decimals, out])	Evenly round to the given number of decimals.
round_(a[, decimals, out])	Round an array to the given number of decimals.
rint(x, /[, out, where, casting, order, ...])	Round elements of the array to the nearest integer.
fix(x[, out])	Round to nearest integer towards zero.
floor(x, /[, out, where, casting, order, ...])	Return the floor of the input, element-wise.
ceil(x, /[, out, where, casting, order, ...])	Return the ceiling of the input, element-wise.
trunc(x, /[, out, where, casting, order, ...])	Return the truncated value of the input, element-wise.
Sums, products, differences
prod(a[, axis, dtype, out, keepdims])	Return the product of array elements over a given axis.
sum(a[, axis, dtype, out, keepdims])	Sum of array elements over a given axis.
nanprod(a[, axis, dtype, out, keepdims])	Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
nansum(a[, axis, dtype, out, keepdims])	Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
cumprod(a[, axis, dtype, out])	Return the cumulative product of elements along a given axis.
cumsum(a[, axis, dtype, out])	Return the cumulative sum of the elements along a given axis.
nancumprod(a[, axis, dtype, out])	Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.
nancumsum(a[, axis, dtype, out])	Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
diff(a[, n, axis])	Calculate the n-th discrete difference along given axis.
ediff1d(ary[, to_end, to_begin])	The differences between consecutive elements of an array.
gradient(f, *varargs, **kwargs)	Return the gradient of an N-dimensional array.
cross(a, b[, axisa, axisb, axisc, axis])	Return the cross product of two (arrays of) vectors.
trapz(y[, x, dx, axis])	Integrate along the given axis using the composite trapezoidal rule.
Exponents and logarithms
exp(x, /[, out, where, casting, order, ...])	Calculate the exponential of all elements in the input array.
expm1(x, /[, out, where, casting, order, ...])	Calculate exp(x) - 1 for all elements in the array.
exp2(x, /[, out, where, casting, order, ...])	Calculate 2**p for all p in the input array.
log(x, /[, out, where, casting, order, ...])	Natural logarithm, element-wise.
log10(x, /[, out, where, casting, order, ...])	Return the base 10 logarithm of the input array, element-wise.
log2(x, /[, out, where, casting, order, ...])	Base-2 logarithm of x.
log1p(x, /[, out, where, casting, order, ...])	Return the natural logarithm of one plus the input array, element-wise.
logaddexp(x1, x2, /[, out, where, casting, ...])	Logarithm of the sum of exponentiations of the inputs.
logaddexp2(x1, x2, /[, out, where, casting, ...])	Logarithm of the sum of exponentiations of the inputs in base-2.
Other special functions
i0(x)	Modified Bessel function of the first kind, order 0.
sinc(x)	Return the sinc function.
Floating point routines
signbit(x, /[, out, where, casting, order, ...])	Returns element-wise True where signbit is set (less than zero).
copysign(x1, x2, /[, out, where, casting, ...])	Change the sign of x1 to that of x2, element-wise.
frexp(x[, out1, out2], / [[, out, where, ...])	Decompose the elements of x into mantissa and twos exponent.
ldexp(x1, x2, /[, out, where, casting, ...])	Returns x1 * 2**x2, element-wise.
nextafter(x1, x2, /[, out, where, casting, ...])	Return the next floating-point value after x1 towards x2, element-wise.
spacing(x, /[, out, where, casting, order, ...])	Return the distance between x and the nearest adjacent number.
Arithmetic operations
add(x1, x2, /[, out, where, casting, order, ...])	Add arguments element-wise.
reciprocal(x, /[, out, where, casting, ...])	Return the reciprocal of the argument, element-wise.
negative(x, /[, out, where, casting, order, ...])	Numerical negative, element-wise.
multiply(x1, x2, /[, out, where, casting, ...])	Multiply arguments element-wise.
divide(x1, x2, /[, out, where, casting, ...])	Divide arguments element-wise.
power(x1, x2, /[, out, where, casting, ...])	First array elements raised to powers from second array, element-wise.
subtract(x1, x2, /[, out, where, casting, ...])	Subtract arguments, element-wise.
true_divide(x1, x2, /[, out, where, ...])	Returns a true division of the inputs, element-wise.
floor_divide(x1, x2, /[, out, where, ...])	Return the largest integer smaller or equal to the division of the inputs.
float_power(x1, x2, /[, out, where, ...])	First array elements raised to powers from second array, element-wise.
fmod(x1, x2, /[, out, where, casting, ...])	Return the element-wise remainder of division.
mod(x1, x2, /[, out, where, casting, order, ...])	Return element-wise remainder of division.
modf(x[, out1, out2], / [[, out, where, ...])	Return the fractional and integral parts of an array, element-wise.
remainder(x1, x2, /[, out, where, casting, ...])	Return element-wise remainder of division.
divmod(x1, x2[, out1, out2], / [[, out, ...])	Return element-wise quotient and remainder simultaneously.
Handling complex numbers
angle(z[, deg])	Return the angle of the complex argument.
real(val)	Return the real part of the complex argument.
imag(val)	Return the imaginary part of the complex argument.
conj(x, /[, out, where, casting, order, ...])	Return the complex conjugate, element-wise.
Miscellaneous
convolve(a, v[, mode])	Returns the discrete, linear convolution of two one-dimensional sequences.
clip(a, a_min, a_max[, out])	Clip (limit) the values in an array.
sqrt(x, /[, out, where, casting, order, ...])	Return the positive square-root of an array, element-wise.
cbrt(x, /[, out, where, casting, order, ...])	Return the cube-root of an array, element-wise.
square(x, /[, out, where, casting, order, ...])	Return the element-wise square of the input.
absolute(x, /[, out, where, casting, order, ...])	Calculate the absolute value element-wise.
fabs(x, /[, out, where, casting, order, ...])	Compute the absolute values element-wise.
sign(x, /[, out, where, casting, order, ...])	Returns an element-wise indication of the sign of a number.
heaviside(x1, x2, /[, out, where, casting, ...])	Compute the Heaviside step function.
maximum(x1, x2, /[, out, where, casting, ...])	Element-wise maximum of array elements.
minimum(x1, x2, /[, out, where, casting, ...])	Element-wise minimum of array elements.
fmax(x1, x2, /[, out, where, casting, ...])	Element-wise maximum of array elements.
fmin(x1, x2, /[, out, where, casting, ...])	Element-wise minimum of array elements.
nan_to_num(x[, copy])	Replace nan with zero and inf with finite numbers.
real_if_close(a[, tol])	If complex input returns a real array if complex parts are close to zero.
interp(x, xp, fp[, left, right, period])	One-dimensional linear interpolation.

"""