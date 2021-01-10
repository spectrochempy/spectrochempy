# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
This module implements the |NDMath| class.
"""

__all__ = ['NDMath', ]

__dataset_methods__ = []

# ======================================================================================================================
# Standard python imports
# ======================================================================================================================
import copy as cpy
import functools
import sys
import operator
from warnings import catch_warnings

# ======================================================================================================================
# third-party imports
# ======================================================================================================================
import numpy as np
from orderedset import OrderedSet
from quaternion import as_float_array

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from spectrochempy.units.units import ur, Quantity, DimensionalityError
from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.utils import MaskedArray, NOMASK, make_func_from, is_sequence, TYPE_COMPLEX
from spectrochempy.core import warning_, error_
from spectrochempy.utils.testing import assert_dataset_equal
from spectrochempy.utils.exceptions import CoordinateMismatchError


# ----------------------------------------------------------------------------------------------------------------------
# decorators
# ----------------------------------------------------------------------------------------------------------------------
#

class class_or_instance_method(object):
    """
    This decorator is designed as a replacement of @classmethod.

    It accept instance or class
    """

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, cls):

        def func(*args, **kwargs):
            # print(instance, cls, args, kwargs, self.method.__name__)

            if instance is not None:
                obj = instance

                # replace some of the attribute
                for k, v in kwargs.items():
                    if k in dir(obj) and k != 'units':
                        setattr(obj, k, v)
                    if k == 'units':
                        obj.ito(v, force=True)

            else:
                obj = cls(*args, **kwargs)

            return self.method(obj, **kwargs)  # *args,

        return func


# ======================================================================================================================
# utility
# ======================================================================================================================
thismodule = sys.modules[__name__]


def get_name(x):
    return str(x.name if hasattr(x, 'name') else x)


DIMENSIONLESS = ur('dimensionless').units
UNITLESS = None
TYPEPRIORITY = {'Coord': 2, 'NDDataset': 3, 'NDPanel': 4}

unary_str = """

negative(x [, out, where, casting, order, …])    Numerical negative, element-wise.
absolute(x [, out, where, casting, order, …])    Calculate the absolute value element-wise.
fabs(x [, out, where, casting, order, …])    Compute the absolute values element-wise.
rint(x [, out, where, casting, order, …])    Round elements of the array to the nearest integer.
conj(x [, out, where, casting, order, …])    Return the complex conjugate, element-wise.

floor(x [, out, where, casting, order, …])    Return the floor of the input, element-wise.
ceil(x [, out, where, casting, order, …])    Return the ceiling of the input, element-wise.
trunc(x [, out, where, casting, order, …])    Return the truncated value of the input, element-wise.

around(x [, decimals, out])                Evenly round to the given number of decimals.
round_(x [, decimals, out])                Round an array to the given number of decimals.
rint(x [, out, where, casting, order, …])  Round elements of the array to the nearest integer.
fix(x[, out])                              Round to nearest integer towards zero (Do not work on NDPanel)

exp(x [, out, where, casting, order, …])     Calculate the exponential of all elements in the input array.
exp2(x [, out, where, casting, order, …])    Calculate 2**p for all p in the input array.
log(x [, out, where, casting, order, …])     Natural logarithm, element-wise.
log2(x [, out, where, casting, order, …])    Base-2 logarithm of x.
log10(x [, out, where, casting, order, …])    Return the base 10 logarithm of the input array, element-wise.
expm1(x [, out, where, casting, order, …])    Calculate exp(x) - 1 for all elements in the array.
log1p(x [, out, where, casting, order, …])    Return the natural logarithm of one plus the input array, element-wise.

sqrt(x [, out, where, casting, order, …])      Return the non-negative square-root of an array, element-wise.
square(x [, out, where, casting, order, …])    Return the element-wise square of the input.
cbrt(x [, out, where, casting, order, …])      Return the cube-root of an array, element-wise.
reciprocal(x [, out, where, casting, …])       Return the reciprocal of the argument, element-wise.

sin(x [, out, where, casting, order, …])       Trigonometric sine, element-wise.
cos(x [, out, where, casting, order, …])       Cosine element-wise.
tan(x [, out, where, casting, order, …])       Compute tangent element-wise.
arcsin(x [, out, where, casting, order, …])    Inverse sine, element-wise.
arccos(x [, out, where, casting, order, …])    Trigonometric inverse cosine, element-wise.
arctan(x [, out, where, casting, order, …])    Trigonometric inverse tangent, element-wise.

sinh(x [, out, where, casting, order, …])       Hyperbolic sine, element-wise.
cosh(x [, out, where, casting, order, …])       Hyperbolic cosine, element-wise.
tanh(x [, out, where, casting, order, …])       Compute hyperbolic tangent element-wise.
arcsinh(x [, out, where, casting, order, …])    Inverse hyperbolic sine element-wise.
arccosh(x [, out, where, casting, order, …])    Inverse hyperbolic cosine, element-wise.
arctanh(x [, out, where, casting, order, …])    Inverse hyperbolic tangent element-wise.

degrees(x [, out, where, casting, order, …])     Convert angles from radians to degrees.
radians(x [, out, where, casting, order, …])     Convert angles from degrees to radians.
deg2rad(x [, out, where, casting, order, …])     Convert angles from degrees to radians.
rad2deg(x [, out, where, casting, order, …])     Convert angles from radians to degrees.

sign(x [, out, where, casting, order, …])       Returns an element-wise indication of the sign of a number.

isfinite(x [, out, where, casting, order, …])   Test element-wise for finiteness (not infinity or not Not a Number).
isinf(x [, out, where, casting, order, …])      Test element-wise for positive or negative infinity.
isnan(x [, out, where, casting, order, …])      Test element-wise for NaN and return result as a boolean array.

logical_not(x [, out, where, casting, …])       Compute the truth value of NOT x element-wise.

signbit(x, [, out, where, casting, order, …])   Returns element-wise True where signbit is set (less than zero).
"""

def unary_ufuncs():
    liste = unary_str.split("\n")
    ufuncs = {}
    for item in liste:
        item = item.strip()
        if item and not item.startswith('#'):
            item = item.split('(')
            string = item[1].split(')')
            ufuncs[item[0]] = f'({string[0]}) -> {string[1].strip()}'
    return ufuncs

binary_str = """

multiply(x1, x2 [, out, where, casting, …])    Multiply arguments element-wise.
divide(x1, x2 [, out, where, casting, …])    Returns a true division of the inputs, element-wise.

maximum(x1, x2 [, out, where, casting, …])    Element-wise maximum of array elements.
minimum(x1, x2 [, out, where, casting, …])    Element-wise minimum of array elements.
fmax(x1, x2 [, out, where, casting, …])    Element-wise maximum of array elements.
fmin(x1, x2 [, out, where, casting, …])    Element-wise minimum of array elements.

add(x1, x2 [, out, where, casting, order, …])    Add arguments element-wise.
subtract(x1, x2 [, out, where, casting, …])    Subtract arguments, element-wise.

copysign(x1, x2 [, out, where, casting, …])    Change the sign of x1 to that of x2, element-wise.
"""


def binary_ufuncs():
    liste = binary_str.split("\n")
    ufuncs = {}
    for item in liste:
        item = item.strip()
        if not item:
            continue
        if item.startswith('#'):
            continue
        item = item.split('(')
        ufuncs[item[0]] = item[1]
    return ufuncs


comp_str = """
# Comparison functions

greater(x1, x2 [, out, where, casting, …])         Return the truth value of (x1 > x2) element-wise.
greater_equal(x1, x2 [, out, where, …])            Return the truth value of (x1 >= x2) element-wise.
less(x1, x2 [, out, where, casting, …])            Return the truth value of (x1 < x2) element-wise.
less_equal(x1, x2 [, out, where, casting, …])      Return the truth value of (x1 =< x2) element-wise.
not_equal(x1, x2 [, out, where, casting, …])       Return (x1 != x2) element-wise.
equal(x1, x2 [, out, where, casting, …])           Return (x1 == x2) element-wise.
"""


def comp_ufuncs():
    liste = comp_str.split("\n")
    ufuncs = {}
    for item in liste:
        item = item.strip()
        if not item:
            continue
        if item.startswith('#'):
            continue
        item = item.split('(')
        ufuncs[item[0]] = item[1]
    return ufuncs


logical_binary_str = """

logical_and(x1, x2 [, out, where, …])          Compute the truth value of x1 AND x2 element-wise.
logical_or(x1, x2 [, out, where, casting, …])  Compute the truth value of x1 OR x2 element-wise.
logical_xor(x1, x2 [, out, where, …])          Compute the truth value of x1 XOR x2, element-wise.
"""


def logical_binary_ufuncs():
    liste = logical_binary_str.split("\n")
    ufuncs = {}
    for item in liste:
        item = item.strip()
        if not item:
            continue
        if item.startswith('#'):
            continue
        item = item.split('(')
        ufuncs[item[0]] = item[1]
    return ufuncs


class NDMath(object):
    """
    This class provides the math and some other array manipulation functionalities to |NDArray| or |Coord|.

    Below is a list of mathematical functions (numpy) implemented (or
    planned for implementation)

    **Ufuncs**

    These functions should work like for numpy-ndarray, except that they
    may be units-aware.

    For instance, `ds`  being a |NDDataset|, just call the np functions like
    this. Most of the time it returns a new NDDataset, while in some cases
    noted below, one get a |ndarray|.

    >>> from spectrochempy import *
    >>> ds = NDDataset([1.,2.,3.])
    >>> np.sin(ds)
    NDDataset: [float64] unitless (size: 3)

    In this particular case (*i.e.*, `np.sin` ufuncs) , the `ds` units must be
    `unitless`, `dimensionless` or angle-units : `radians` or `degrees`,
    or an exception will be raised.


    Examples
    --------

    >>> from spectrochempy import *
    >>> nd1 = NDDataset.read('wodger.spg')
    >>> nd1
    NDDataset: [float32]  a.u. (shape: (y:2, x:5549))
    >>> nd1.data
    array([[   2.005,    2.003, ...,    1.826,    1.831],
           [   1.983,    1.984, ...,    1.698,    1.704]], dtype=float32)
    >>> nd2 = np.negative(nd1)
    >>> nd2
    NDDataset: [float32]  a.u. (shape: (y:2, x:5549))
    >>> nd2.data
    array([[  -2.005,   -2.003, ...,   -1.826,   -1.831],
           [  -1.983,   -1.984, ...,   -1.698,   -1.704]], dtype=float32)
    """

    __radian = 'radian'
    __degree = 'degree'
    __require_units = {'cumprod': DIMENSIONLESS, 'arccos': DIMENSIONLESS, 'arcsin': DIMENSIONLESS,
                       'arctan': DIMENSIONLESS, 'arccosh': DIMENSIONLESS, 'arcsinh': DIMENSIONLESS,
                       'arctanh': DIMENSIONLESS, 'exp': DIMENSIONLESS, 'expm1': DIMENSIONLESS, 'exp2': DIMENSIONLESS,
                       'log': DIMENSIONLESS, 'log10': DIMENSIONLESS, 'log1p': DIMENSIONLESS, 'log2': DIMENSIONLESS,
                       'sin': __radian, 'cos': __radian, 'tan': __radian, 'sinh': __radian, 'cosh': __radian,
                       'tanh': __radian, 'radians': __degree, 'degrees': __radian, 'deg2rad': __degree,
                       'rad2deg': __radian, 'logaddexp': DIMENSIONLESS, 'logaddexp2': DIMENSIONLESS}
    __compatible_units = ['add', 'sub', 'iadd', 'isub', 'maximum', 'minimum', 'fmin', 'fmax', 'lt', 'le', 'ge', 'gt']
    __complex_funcs = ['real', 'imag', 'conjugate', 'absolute', 'conj', 'abs']
    __keep_title = ['negative', 'absolute', 'abs', 'fabs', 'rint', 'floor', 'ceil', 'trunc', 'add', 'subtract']
    __remove_title = ['multiply', 'divide', 'true_divide', 'floor_divide', 'mod', 'fmod', 'remainder', 'logaddexp',
                      'logaddexp2']
    __remove_units = ['logical_not', 'isfinite', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'iscomplex',
                      'signbit', 'sign']

    # the following methods are to give NDArray based class
    # a behavior similar to np.ndarray regarding the ufuncs

    @property
    def __array_struct__(self):
        if hasattr(self.umasked_data, 'mask'):
            self._mask = self.umasked_data.mask
        return self.data.__array_struct__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        fname = ufunc.__name__

        #        # case of complex or hypercomplex data
        #        if self.implements(NDComplexArray) and self.has_complex_dims:
        #
        #            if fname in self.__complex_funcs:
        #                return getattr(inputs[0], fname)()
        #
        #            if fname in ["fabs", ]:
        #                # fonction not available for complex data
        #                raise ValueError(f"Operation `{ufunc}` does not accept complex data!")
        #
        #        # If this reached, data are not complex or hypercomplex
        #        if fname in ['absolute', 'abs']:
        #            f = np.fabs

        # set history string
        history = f'Ufunc {fname} applied.'

        inputtype = type(inputs[0]).__name__

        if inputtype == 'NDPanel':

            # Some ufunc can not be applied to panels
            if fname in ['sign', 'logical_not', 'isnan', 'isfinite', 'isinf', 'signbit']:
                raise NotImplementedError(f'`{fname}` ufunc is not implemented for NDPanel objects.')

            # if we have a NDPanel, process the ufuncs on all datasets
            datasets = self._op(ufunc, inputs, isufunc=True)

            # recreate a panel object
            obj = type(inputs[0])
            panel = obj(*datasets, merge=True, align=None)
            panel.history = history

            # return it
            return panel

        else:
            # Some ufunc can not be applied to panels
            if fname in ['sign', 'logical_not', 'isnan', 'isfinite', 'isinf', 'signbit']:
                return (getattr(np, fname))(inputs[0].masked_data)

            # case of a dataset
            data, units, mask, returntype = self._op(ufunc, inputs, isufunc=True)
            new = self._op_result(data, units, mask, history, returntype)

            # make a new title depending on the operation
            if fname in self.__remove_title:
                new.title = f"<{fname}>"
            elif fname not in self.__keep_title and isinstance(new, NDArray):
                if hasattr(new, 'title') and new.title is not None:
                    new.title = f"{fname}({new.title})"
                else:
                    new.title = f"{fname}(data)"
            return new

    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def abs(self, inplace=False):
        """
        Returns the absolute value of a complex array.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Flag to say that the method return a new object (default)
            or not (inplace=True)

        Returns
        -------
        out
            Same object or a copy depending on the ``inplace`` flag.
        """
        if inplace:
            new = self
        else:
            new = self.copy()
        if not new.has_complex_dims:
            return np.fabs(new)  # not a complex, return fabs should be faster

        elif not new.is_quaternion:
            new = np.sqrt(new.real ** 2 + new.imag ** 2)
        else:
            new = np.sqrt(new.real ** 2 + new.part('IR') ** 2 + new.part('RI') ** 2 + new.part('II') ** 2)
            new._is_quaternion = False

        return new

    absolute = abs

    def pipe(self, func, *args, **kwargs):
        """Apply func(self, *args, **kwargs)

        Parameters
        ----------
        func : function
            function to apply to the |NDDataset|.
            `*args`, and `**kwargs` are passed into `func`.
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
                error_(f'{target} is both the pipe target and a keyword argument. Operation not applied!')
                return self
            kwargs[target] = self
            return func(*args, **kwargs)

        return func(self, *args, **kwargs)

    # ..................................................................................................................
    def sum(self, *args, **kwargs):
        """sum along axis"""

        return self._reduce_method('sum', *args, **kwargs)

    def prod(self, *args, **kwargs):
        """product along axis"""

        return self._reduce_method('prod', *args, **kwargs)

    product = prod

    def cumsum(self, *args, **kwargs):
        """cumsum along axis"""

        return self._method('cumsum', *args, **kwargs)

    def cumprod(self, *args, **kwargs):
        """cumprod along axis"""

        return self._method('cumprod', *args, **kwargs)

    cumproduct = cumprod

    # ..................................................................................................................
    def mean(self, *args, **kwargs):
        """mean values along axis"""

        return self._reduce_method('mean', *args, **kwargs)

    # ..................................................................................................................
    def var(self, *args, **kwargs):
        """variance values along axis"""

        return self._reduce_method('var', *args, **kwargs)

    # ..................................................................................................................
    def std(self, *args, **kwargs):
        """Standard deviation values along axis"""

        return self._reduce_method('std', *args, **kwargs)

    # ..................................................................................................................
    def ptp(self, *args, **kwargs):
        """
        Range of values (maximum - minimum) along a dimension.

        The name of the function comes from the acronym for 'peak to peak'.

        Parameters
        ----------
        axis : None or int, optional
            Dimension along which to find the peaks.
            If None, the operation is made on the first dimension
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input dataset.

        Returns
        -------
        ptp : nddataset
            A new dataset holding the result.
        """

        return self._reduce_method('ptp', *args, **kwargs)

    # ..................................................................................................................
    def all(self, *args, **kwargs):
        """Test whether all array elements along a given axis evaluate to True."""

        return self._reduce_method('all', *args, keepunits=False, **kwargs)

    # ..................................................................................................................
    def any(self, *args, **kwargs):
        """Test whether any array elements along a given axis evaluate to True."""

        return self._reduce_method('any', *args, keepunits=False, **kwargs)

    sometrue = any

    # ............................................................................
    @class_or_instance_method
    def diag(self, *args, **kwargs):
        """
        Extract a diagonal or construct a diagonal array.

        See the more detailed documentation for ``numpy.diagonal`` if you use this
        function to extract a diagonal and wish to write to the resulting array;
        whether it returns a copy or a view depends on what version of numpy you
        are using.

        Adapted from numpy (licence #TO ADD)

        Parameters
        ----------
        v : array_like
            If `v` is a 2-D array, return a copy of its `k`-th diagonal.
            If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
            diagonal.

        Returns
        -------
        out : ndarray
            The extracted diagonal or constructed diagonal array.
        """
        # TODO: fix this - other diagonals
        # k : int, optional
        # Diagonal in question. The default is 0. Use `k>0` for diagonals
        # above the main diagonal, and `k<0` for diagonals below the main
        # diagonal.

        new = self.copy()

        if new.ndim == 1:

            # construct a diagonal array
            # --------------------------

            data = np.diag(new.data)

            mask = NOMASK
            if new.is_masked:
                size = new.size
                m = np.repeat(new.mask, size).reshape(size, size)
                mask = m | m.T

            coordset = None
            if new.coordset is not None:
                coordset = (new.coordset[0], new.coordset[0])

            dims = ['y'] + new.dims

            new.data = data
            new.mask = mask
            new._dims = dims
            if coordset is not None:
                new.set_coordset(coordset)
            new.history = 'Diagonal array build from the 1D dataset'
            return new

        elif new.ndim == 2:
            # extract a diagonal
            # ------------------
            return new._diag(**kwargs)

        else:
            raise ValueError("Input must be 1- or 2-d.")

    # ..................................................................................................................
    def _diag(self, **kwargs):
        """take diagonal of a 2D array"""
        # As we reduce a 2D to a 1D we must specified which is the dimension for the coordinates to keep!

        if not kwargs.get("axis", kwargs.get("dims", kwargs.get("dim", None))):
            warning_('Dimensions to remove for coordinates must be specified. By default the first is kept. ')

        return self._reduce_method('diag', **kwargs)

    # ..................................................................................................................
    @classmethod
    def fromfunction(cls, function, *, shape=None, coordset=None, dtype=float, name=None, title=None, units=None,
                     **kwargs):
        """
        Construct a nddataset by executing a function over each coordinate.

        The resulting array therefore has a value ``fn(x, y, z)`` at
        coordinate ``(x, y, z)``.

        Parameters
        ----------
        function : callable
            The function is called with N parameters, where N is the rank of
            `shape` or from the provided ``coordset`.
        shape : (N,) tuple of ints, optional
            Shape of the output array, which also determines the shape of
            the coordinate arrays passed to `function`. It is optional only if
            `coordset` is None.
        name : str, optional
            Dataset name
        title : str, optional
            Dataset title (see |NDDataset.title|)
        units : str, optional
            Dataset units.
            When None, units will be determined from the function results
        coordset : |Coordset| instance, optional
            If provided, this determine the shape and coordinates of each dimension of
            the returned |NDDataset|. If shape is also passed it will be ignored.
        dtype : data-type, optional
            Data-type of the coordinate arrays passed to `function`.
            By default, `dtype` is float.

        Returns
        -------
        fromfunction : any
            The result of the call to `function` is passed back directly.
            Therefore the shape of `fromfunction` is completely determined by
            `function`.

        Notes
        -----
        Keywords **kwargs are passed to `function`.

        Examples
        --------
        >>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
        array([[ True, False, False],
               [False,  True, False],
               [False, False,  True]])

        >>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
        array([[0, 1, 2],
               [1, 2, 3],
               [2, 3, 4]])
        """
        from spectrochempy.core.dataset.coordset import CoordSet
        if coordset is not None:
            if not isinstance(coordset, CoordSet):
                coordset = CoordSet(*coordset)

            shape = coordset.sizes

        idx = np.indices(shape)

        args = [0] * len(shape)
        if coordset is not None:
            for i, co in enumerate(coordset):
                args[i] = co.data[idx[i]]
                if units is None and co.has_units:
                    args[i] = Quantity(args[i], co.units)

        data = function(*args, **kwargs)

        # argsunits = [1] * len(shape)
        # if coordset is not None:
        #     for i, co in enumerate(coordset):
        #         args[i] = co.data[idx[i]]
        #         if units is None and co.has_units:
        #             argsunits[i] = Quantity(1, co.units)
        #
        # kwargsunits = {}
        # for k, v in kwargs.items():
        #     if isinstance(v, Quantity) or hasattr(v, 'has_units'):
        #         kwargs[k] = v.m
        #         kwargsunits[k] = Quantity(1., v.u)
        #     else:
        #         kwargsunits[k] = 1
        #
        # data = function(*args, **kwargs)
        #
        # if units is None:
        #     q = function(*argsunits, **kwargsunits)
        #     if hasattr(q, 'units'):
        #         units = q.units

        data = data.T
        dims = coordset.names[::-1]
        return cls(data, coordset=coordset, dims=dims, name=name, title=title, units=units)

    # ..................................................................................................................
    def clip(self, *args, **kwargs):
        """
        Clip (limit) the values in a dataset.

        Given an interval, values outside the interval are clipped to
        the interval edges.  For example, if an interval of ``[0, 1]``
        is specified, values smaller than 0 become 0, and values larger
        than 1 become 1.

        No check is performed to ensure ``a_min < a_max``.

        Parameters
        ----------
        a_min : scalar or array_like or None
            Minimum value. If None, clipping is not performed on lower
            interval edge. Not more than one of `a_min` and `a_max` may be
            None.
        a_max : scalar or array_like or None
            Maximum value. If None, clipping is not performed on upper
            interval edge. Not more than one of `a_min` and `a_max` may be
            None. If `a_min` or `a_max` are array_like, then the three
            arrays will be broadcasted to match their shapes.

        Returns
        -------
        clipped_array : ndarray
            An array with the elements of `a`, but where values
            < `a_min` are replaced with `a_min`, and those > `a_max`
            with `a_max`.
        """
        if len(args) > 2 or len(args) == 0:
            raise ValueError('Clip requires at least one argument or at most two arguments')
        amax = kwargs.pop('a_max', args[0] if len(args) == 1 else args[1])
        amin = kwargs.pop('a_min', self.min() if len(args) == 1 else args[0])
        amin, amax = np.minimum(amin, amax), max(amin, amax)
        if self.has_units:
            if not isinstance(amin, Quantity):
                amin = amin * self.units
            if not isinstance(amax, Quantity):
                amax = amax * self.units
        res = self._method('clip', a_min=amin, a_max=amax, **kwargs)
        res.history = f'Clipped with limits between {amin} and {amax}'
        return res

    # ..................................................................................................................
    def round(self, *args, **kwargs):
        """Round an array to the given number of decimals.."""

        return self._method('round', *args, **kwargs)

    around = round_ = round

    # ..................................................................................................................
    def amin(self, *args, **kwargs):
        """
        Return the maximum of the dataset or maxima along given dimensions.

        Parameters
        ----------
        dim : None or int or dimension name or tuple of int or dimensions, optional
            dimension or dimensions along which to operate.  By default, flattened input is used.
            If this is a tuple, the minimum is selected over multiple dimensions,
            instead of a single dimension or all the dimensions as before.
        axis : alias for dim
            Compatibility with Numpy syntax
        out : nddataset, optional
            Alternative output dataset in which to place the result.  Must
            be of the same shape and buffer length as the expected output.
            See `ufuncs-output-type` for more details.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
        initial : scalar, optional
            The minimum value of an output element. Must be present to allow
            computation on empty slice.
        where : array_like of bool, optional
            Elements to compare for the minimum.

        Returns
        -------
        amin : ndarray or scalar
            Minimum of the data. If `dim` is None, the result is a scalar value.
            If `dim` is given, the result is an array of dimension ``ndim - 1``.

        See Also
        --------
        amax :
            The maximum value of a dataset along a given dimension, propagating any NaNs.
        nanmax :
            The maximum value of a dataset along a given dimension, ignoring any NaNs.
        maximum :
            Element-wise maximum of two datasets, propagating any NaNs.
        fmax :
            Element-wise maximum of two datasets, ignoring any NaNs.
        argmax :
            Return the indices or coordinates of the maximum values.
        nanmin, minimum, fmin
        """
        return self._reduce_method('amin', *args, **kwargs)

    min = amin

    # ..................................................................................................................
    def max(self, *args, **kwargs):
        """
        Return the maximum of the dataset or maxima along given dimensions.

        Parameters
        ----------
        dim : None or int or dimension name or tuple of int or dimensions, optional
            dimension or dimensions along which to operate.  By default, flattened input is used.
            If this is a tuple, the maximum is selected over multiple dimensions,
            instead of a single dimension or all the dimensions as before.
        axis : alias for dim
            Compatibility with Numpy syntax
        out : nddataset, optional
            Alternative output dataset in which to place the result.  Must
            be of the same shape and buffer length as the expected output.
            See `ufuncs-output-type` for more details.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
        initial : scalar, optional
            The minimum value of an output element. Must be present to allow
            computation on empty slice.
        where : array_like of bool, optional
            Elements to compare for the maximum. See `~numpy.ufunc.reduce`
            for details.

        Returns
        -------
        amax : ndarray or scalar
            Maximum of the data. If `dim` is None, the result is a scalar value.
            If `dim` is given, the result is an array of dimension ``ndim - 1``.

        See Also
        --------
        amin :
            The minimum value of a dataset along a given dimension, propagating any NaNs.
        nanmin :
            The minimum value of a dataset along a given dimension, ignoring any NaNs.
        minimum :
            Element-wise minimum of two datasets, propagating any NaNs.
        fmin :
            Element-wise minimum of two datasets, ignoring any NaNs.
        argmin :
            Return the indices or coordinates of the minimum values.
        nanmax, maximum, fmax
        """

        return self._reduce_method('max', *args, **kwargs)

    amax = max

    # ..................................................................................................................
    def argmin(self, *args, **kwargs):
        """indexes of minimum of data along axis"""

        return self._reduce_method('argmin', *args, **kwargs)

    # ..................................................................................................................
    def argmax(self, *args, **kwargs):
        """indexes of maximum of data along axis"""

        return self._reduce_method('argmax', *args, **kwargs)

    # ..................................................................................................................
    def coordmin(self, *args, **kwargs):
        """Coordinates of minimum of data along axis"""

        mi = self.min(keepdims=True)
        return mi.coordset

    # ..................................................................................................................
    def coordmax(self, *args, **kwargs):
        """Coordinates of maximum of data along axis"""

        ma = self.max(keepdims=True)
        return ma.coordset

    @classmethod
    def rand(cls, *args):

        return cls(np.random.rand(*args))

    # ..................................................................................................................
    @classmethod
    def arange(cls, start=0, stop=None, step=None, dtype=None, **kwargs):
        """
        """
        return cls(np.arange(start, stop, step, dtype=np.dtype(dtype)), **kwargs)

    @classmethod
    def linspace(cls, start, stop, num=50, endpoint=True, retstep=False, dtype=None, **kwargs):
        """
        Return evenly spaced numbers over a specified interval.

        Returns num evenly spaced samples, calculated over the interval [start, stop]. The endpoint of the interval
        can optionally be excluded.

        Parameters
        ----------
        start : array_like
            The starting value of the sequence.
        stop : array_like
            The end value of the sequence, unless endpoint is set to False.
            In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is
            excluded. Note that the step size changes when endpoint is False.
        num : int, optional
            Number of samples to generate. Default is 50. Must be non-negative.
        endpoint : bool, optional
            If True, stop is the last sample. Otherwise, it is not included. Default is True.
        retstep : bool, optional
            If True, return (samples, step), where step is the spacing between samples.
        dtype : dtype, optional
            The type of the array. If dtype is not given, infer the data type from the other input arguments.
        **kwargs : any
            keywords argument used when creating the returned object, such as units, name, title, ...

        Returns
        -------
        samples : ndarray
            There are num equally spaced samples in the closed interval [start, stop] or the half-open interval
            [start, stop) (depending on whether endpoint is True or False).
        step : float, optional
            Only returned if retstep is True
            Size of spacing between samples.
        """

        return cls(np.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype), **kwargs)

    @classmethod
    def logspace(cls, start, stop, num=50, endpoint=True, base=10.0, dtype=None, **kwargs):
        """
        Return numbers spaced evenly on a log scale.

        In linear space, the sequence starts at ``base ** start``
        (`base` to the power of `start`) and ends with ``base ** stop``
        (see `endpoint` below).

        Parameters
        ----------
        start : array_like
            ``base ** start`` is the starting value of the sequence.
        stop : array_like
            ``base ** stop`` is the final value of the sequence, unless `endpoint`
            is False.  In that case, ``num + 1`` values are spaced over the
            interval in log-space, of which all but the last (a sequence of
            length `num`) are returned.
        num : integer, optional
            Number of samples to generate.  Default is 50.
        endpoint : boolean, optional
            If true, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        base : float, optional
            The base of the log space. The step size between the elements in
            ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
            Default is 10.0.
        dtype : dtype
            The type of the output array.  If `dtype` is not given, infer the data
            type from the other input arguments.

        Returns
        -------
        samples : ndarray
            `num` samples, equally spaced on a log scale.
        See Also
        --------
        arange : Similar to linspace, with the step size specified instead of the
                 number of samples. Note that, when used with a float endpoint, the
                 endpoint may or may not be included.
        linspace : Similar to logspace, but with the samples uniformly distributed
                   in linear space, instead of log space.
        geomspace : Similar to logspace, but with endpoints specified directly.
        Notes
        -----
        Logspace is equivalent to the code
        >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)
        ... # doctest: +SKIP
        >>> power(base, y).astype(dtype)
        ... # doctest: +SKIP
        """
        return cls(np.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype), **kwargs)

    @classmethod
    def identity(cls, N, dtype=None, **kwargs):
        """
        Return the identity |NDDataset| of a given shape.

        The identity array is a square array with ones on
        the main diagonal.

        Parameters
        ----------
        N : int
            Number of rows (and columns) in `n` x `n` output.
        dtype : data-type, optional
            Data-type of the output.  Defaults to ``float``.

        Returns
        -------
        out : nddataset
            `n` x `n` array with its main diagonal set to one,
            and all other elements 0.

        Examples
        --------
        >>> import spectrochempy as scp
        >>> scp.identity(3).data
        array([[       1,        0,        0],
               [       0,        1,        0],
               [       0,        0,        1]])
        """
        return cls.eye(N, dtype=dtype, **kwargs)

    @classmethod
    def eye(cls, N, M=None, k=0, dtype=float, order='C', **kwargs):
        """
        Return a 2-D array with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        N : int
            Number of rows in the output.
        M : int, optional
            Number of columns in the output. If None, defaults to `N`.
        k : int, optional
            Index of the diagonal: 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal.
        dtype : data-type, optional
            Data-type of the returned array.
        order : {'C', 'F'}, optional
            Whether the output should be stored in row-major (C-style) or
            column-major (Fortran-style) order in memory.

        Returns
        -------
        I : NDDataset of shape (N,M)
            An array where all elements are equal to zero, except for the `k`-th
            diagonal, whose values are equal to one.

        See Also
        --------
        identity : equivalent function with k=0.
        diag : diagonal 2-D NDDataset from a 1-D array specified by the user.

        Examples
        --------
        >>> np.eye(2, dtype=int)
        array([[       1,        0],
               [       0,        1]])
        >>> np.eye(3, k=1)
        array([[       0,        1,        0],
               [       0,        0,        1],
               [       0,        0,        0]])
        """
        return cls(np.eye(N, M, k, dtype, order), **kwargs)

    @staticmethod
    def empty(shape, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, without initializing entries.

        Rhis is a wrapper to the numpy

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the empty array
        dtype : data-type, optional
            Desired output data-type.

        Returns
        -------
        out : |NDDataset|
            Array of uninitialized (arbitrary) data of the given shape, dtype, and
            order.  Object arrays will be initialized to None.

        See Also
        --------
        empty_like, zeros, ones

        Notes
        -----
        `empty`, unlike `zeros`, does not set the array values to zero,
        and may therefore be marginally faster.  On the other hand, it requires
        the user to manually set all the values in the array, and should be
        used with caution.

        Examples
        --------
        >>> from spectrochempy import *

        >>> NDDataset.empty([2, 2], dtype=int, units='s')
        NDDataset: [int64] s (shape: (y:2, x:2))
        """
        return NDMath._create(shape, fill_value=None, **kwargs)

    @staticmethod
    def zeros(shape, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with zeros.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is
            `numpy.float64`.
        **kwargs : keyword args to pass to the |NDDataset| constructor

        Returns
        -------
        out : |NDDataset|
            Array of zeros with the given shape, dtype.

        See Also
        --------
        ones, zeros_like

        Examples
        --------
        >>> import spectrochempy as scp
        >>> nd = scp.NDDataset.zeros(6)
        >>> nd
        NDDataset: [float64] unitless (size: 6)
        >>> nd = scp.zeros((5, ))
        >>> nd
        NDDataset: [float64] unitless (size: 5)
        >>> nd.values
        array([       0,        0,        0,        0,        0])
        >>> nd = scp.zeros((5, 10), dtype=np.int, units='absorbance')
        >>> nd
        NDDataset: [int64] a.u. (shape: (y:5, x:10))
        """
        return NDMath._create(shape, fill_value=0.0, **kwargs)

    @staticmethod
    def ones(shape, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with ones.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is
            `numpy.float64`.
        **kwargs : keyword args to pass to the |NDDataset| constructor

        Returns
        -------
        out : |NDDataset|
            Array of ones with the given shape, dtype.

        See Also
        --------
        zeros, ones_like

        Examples
        --------
        >>> import spectrochempy as scp
        >>> nd = scp.ones(5, units='km')
        >>> nd
        NDDataset: [float64] km (size: 5)
        >>> nd.values
        <Quantity([       1        1        1        1        1], 'kilometer')>
        >>> nd = scp.ones((5,), dtype=np.int, mask=[True, False, False, False, True])
        >>> nd
        NDDataset: [int64] unitless (size: 5)
        >>> nd.values
        masked_array(data=[  --,        1,        1,        1,   --],
                     mask=[  True,   False,   False,   False,   True],
               fill_value=999999)
        >>> nd = scp.ones((5,), dtype=np.int, mask=[True, False, False, False, True], units='joule')
        >>> nd
        NDDataset: [int64] J (size: 5)
        >>> nd.values
        <Quantity([  --        1        1        1   --], 'joule')>
        >>> scp.ones((2, 2)).values
        array([[       1,        1],
               [       1,        1]])
        """
        return NDMath._create(shape, fill_value=1.0, **kwargs)

    @staticmethod
    def full(shape, fill_value=0.0, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with `fill_value`.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        fill_value : scalar
            Fill value.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `np.int8`.  Default is fill_value.dtype.
        **kwargs : keyword args to pass to the |NDDataset| constructor

        Returns
        -------
        out : |NDDataset|
            Array of `fill_value` with the given shape, dtype, and order.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.

        Examples
        --------
        >>> from spectrochempy import *
        >>> Coord.full((2, ), np.inf)
        Coord: [float64] unitless (size: 2)
        >>> NDDataset.full((2, 2), 10, dtype=np.int)
        NDDataset: [int64] unitless (shape: (y:2, x:2))
        """
        return NDMath._create(shape, fill_value=fill_value, **kwargs)

    @classmethod
    def empty_like(cls, *args, **kwargs):
        """
        Return a new array with the same shape and type as a given array.

        Parameters
        ----------
        a : array_like
            The shape and data-type of `a` define these same attributes of the
            returned array.
        dtype : data-type, optional
            Overrides the data type of the result.

        Returns
        -------
        out : ndarray
            Array of uninitialized (arbitrary) data with the same
            shape and type as `a`.

        See Also
        --------
        ones_like : Return an array of ones with shape and type of input.
        zeros_like : Return an array of zeros with shape and type of input.
        empty : Return a new uninitialized array.
        ones : Return a new array setting values to one.
        zeros : Return a new array setting values to zero.

        Notes
        -----
        This function does *not* initialize the returned array; to do that use
        for instance `zeros_like`, `ones_like` or `full_like` instead.  It may be
        marginally faster than the functions that do set the array values.
        """

        return NDMath._like(cls, *args, **kwargs)

    @classmethod
    def zeros_like(cls, *args, **kwargs):
        """
        Return a |NDDataset| of zeros with the same shape and type as a given
        array.

        Parameters
        ----------
        a : |NDDataset|
        dtype : data-type, optional
            Overrides the data type of the result.

        Returns
        -------
        out : |NDDataset|
            Array of zeros with the same shape and type as `a`.

        See Also
        --------
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.

        Examples
        --------
        >>> import spectrochempy as scp
        >>> x = np.arange(6)
        >>> x = x.reshape((2, 3))
        >>> nd = scp.NDDataset(x, units='s')
        >>> nd
        NDDataset: [int64] s (shape: (y:2, x:3))
        >>> nd.values
         <Quantity([[       0        1        2]
         [       3        4        5]], 'second')>
        >>> nd = scp.zeros_like(nd)
        >>> nd
        NDDataset: [int64] s (shape: (y:2, x:3))
        >>> nd.values
            <Quantity([[       0        0        0]
         [       0        0        0]], 'second')>
        """
        return NDMath._like(cls, *args, fill_value=0.0, **kwargs)

    @classmethod
    def ones_like(cls, *args, **kwargs):
        """
        Return |NDDataset| of ones with the same shape and type as a given array.

        It preserves original mask, units, and coordset

        Parameters
        ----------
        a : |NDDataset|
        dtype : data-type, optional
            Overrides the data type of the result.

        Returns
        -------
        out : |NDDataset|
            Array of ones with the same shape and type as `a`.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.

        Examples
        --------
        >>> import spectrochempy as scp
        >>> x = np.arange(6)
        >>> x = x.reshape((2, 3))
        >>> x = scp.NDDataset(x, units='s')
        >>> x
        NDDataset: [int64] s (shape: (y:2, x:3))
        >>> scp.ones_like(x, dtype=float, units='J')
        NDDataset: [float64] J (shape: (y:2, x:3))
        """

        return NDMath._like(cls, *args, fill_value=1.0, **kwargs)

    @classmethod
    def full_like(cls, *args, **kwargs):
        """
        Return a |NDDataset| with the same shape and type as a given array.

        Parameters
        ----------
        a : |NDDataset| or array-like
        fill_value : scalar
            Fill value.
        dtype : data-type, optional
            Overrides the data type of the result.

        Returns
        -------
        array-like
            Array of `fill_value` with the same shape and type as `a`.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------
        >>> from spectrochempy import *

        >>> x = np.arange(6, dtype=int)
        >>> nd = full_like(x, 1)
        >>> nd
        NDDataset: [int64] unitless (size: 6)
        >>> nd.values
        array([       1,        1,        1,        1,        1,        1])
        >>> x = NDDataset(x, units='m')
        >>> NDDataset.full_like(x, 0.1).values
        <Quantity([       0        0        0        0        0        0], 'meter')>
        >>> full_like(x, 0.1, dtype=np.double).values
        <Quantity([     0.1      0.1      0.1      0.1      0.1      0.1], 'meter')>
        >>> full_like(x, np.nan, dtype=np.double).values
        <Quantity([     nan     nan      nan      nan      nan      nan], 'meter')>
        """
        return NDMath._like(cls, *args, **kwargs)

    # ----------------------------------------------------------------------------------------------------------------------
    # Private methods
    #

    @staticmethod
    def _create(*args, **kwargs):

        from spectrochempy.core.dataset.nddataset import NDDataset

        args = list(args)
        shape = args.pop(0)
        fill_value = kwargs.pop('fill_value', 0.0)
        dtype = kwargs.pop('dtype', None)

        if fill_value is not None:
            return NDDataset(np.full(shape, fill_value=fill_value, dtype=dtype), **kwargs)
        else:
            return NDDataset(np.empty(shape, dtype=dtype), **kwargs)

    @classmethod
    def _like(cls, *args, **kwargs):

        from spectrochempy.core.dataset.nddataset import NDDataset

        args = list(args)

        if isinstance(args[0], NDArray):
            a = args.pop(0)
            new = type(a)(np.empty_like(a.data))

        elif issubclass(args[0], NDArray):
            new = args.pop(0)()  # copy type
            ds = args.pop(0)  # get the template object
            if isinstance(ds, NDArray):
                new._data = np.empty_like(ds.data)
                new._dims = ds.dims.copy()
                new._mask = ds.mask.copy()
                if hasattr(ds, 'coordset'):
                    new._coordset = ds.coordset
                new._units = ds.units
                new._title = ds.title
            elif is_sequence(ds):
                # by default we produce a NDDataset
                new._data = NDDataset(np.empty_like(ds))

        fill_value = kwargs.pop('fill_value', args.pop(0) if args else None)
        dtype = kwargs.pop('dtype', None)
        units = kwargs.pop('units', None)
        coordset = kwargs.pop('coordset', None)
        if dtype is not None:
            new = new.astype(np.dtype(dtype))
        if fill_value is not None:
            new._data = np.full_like(new.data, fill_value=fill_value)
        if units is not None:
            new.ito(units, force=True)
        if coordset is not None:
            new._coordset = coordset
        return new

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------

    # Methods without dataset reduction
    def _method(self, op, *args, **kwargs):

        new = self.copy()

        if new.implements('NDPanel'):
            # if we have a NDPanel, iterate on all internal dataset of the panel
            datasets = []
            for k, v in new.datasets.items():
                v._coordset = new.coordset
                v.name = k
                datasets.append(getattr(np, op)(v))

            # recreate a panel object
            panel = type(new)(*datasets, merge=True, align=None)
            panel.history = f'Panel resulting from application of `{op}` method'

            # return it
            return panel

        if args:
            kwargs['dim'] = args[0]
            args = []

        # handle the various syntax to pass the axis
        dims = self._get_dims_from_args(*args, **kwargs)
        axis = self._get_dims_index(dims)
        axis = axis[0] if axis and not self.is_1d else None
        if axis is not None:
            kwargs['axis'] = axis

        # dim and dims keyword not accepted by the np function, so remove it
        kwargs.pop('dims', None)
        kwargs.pop('dim', None)

        # they may be a problem if some kwargs have units
        for k, v in kwargs.items():
            if hasattr(v, 'units'):
                kwargs[k] = v.m

        # apply the numpy operator on the masked data
        arr = getattr(np, op)(self.masked_data, *args, **kwargs)

        if isinstance(arr, MaskedArray):
            new._data = arr.data
            new._mask = arr.mask

        elif isinstance(arr, np.ndarray):
            new._data = arr
            new._mask = NOMASK

        # particular case of functions that returns Dataset with no coordinates
        if axis is None and op in ['cumsum', 'cumprod']:
            # delete all coordinates
            new._coordset.data = None

        # Here we must reduce the corresponding coordinates
        elif axis is not None:
            dim = new._dims[axis]
            if op not in ['cumsum', 'cumprod']:
                del new._dims[axis]
            if new.implements('NDDataset') and new._coordset and (dim in new._coordset.names):
                idx = new._coordset.names.index(dim)
                del new._coordset.coords[idx]

        new.history = f'Dataset resulting from application of `{op}` method'
        return new

    # Methods with dataset reduction
    def _reduce_method(self, op, *args, **kwargs):
        # TODO: make change to handle complex and quaternion
        new = self.copy()

        keepdims = kwargs.get('keepdims', False)
        keepunits = kwargs.pop('keepunits', True)

        if args:
            kwargs['dim'] = args[0]
            args = []

        # handle the various syntax to pass the axis
        dims = self._get_dims_from_args(*args, **kwargs)
        axis = self._get_dims_index(dims)
        axis = axis[0] if axis and not self.is_1d else None
        kwargs['axis'] = axis

        # dim and dims keyword not accepted by the np function, so remove it
        kwargs.pop('dims', None)
        kwargs.pop('dim', None)
        if op in ['diag']:
            # also remove axis
            kwargs.pop('axis', None)

        # particular case of ptp
        if axis is None and not self.is_1d and op in ['ptp']:
            kwargs['axis'] = axis = -1

        # particular case of argmax and argmin that only return indexes
        if op in ['argmin', 'argmax']:
            kwargs.pop('keepdims', None)
            idx = getattr(np, op)(self.real.masked_data, **kwargs)
            idx = np.unravel_index(idx, self.shape)
            if self.ndim == 1:
                idx = idx[0]
            return idx

        # particular case of max and min
        if axis is None and keepdims and op in ['max', 'amax', 'min', 'amin']:
            if op.startswith('a'):
                op = op[1:]
            idx = getattr(np, "arg" + op)(self.real.masked_data)
            idx = np.unravel_index(idx, self.shape)
            new = self[idx]

        else:
            # apply the numpy operator on the masked data
            arr = getattr(np, op)(self.real.masked_data, *args, **kwargs)

            # simpler case where we return a scalar value or a quantity

            if isinstance(arr, MaskedArray):
                new._data = arr.data
                new._mask = arr.mask

            elif isinstance(arr, np.ndarray):
                new._data = arr
                new._mask = NOMASK

            else:
                # simpler case for a returned scalar or quantity
                if new.has_units and keepunits:
                    new = arr * new._units
                else:
                    new = arr
                if not keepdims:
                    return new

            # particular case of functions that returns Dataset with no coordinates
            if axis is None and op in ['sum', 'trapz', 'prod', 'mean', 'var', 'std']:
                # delete all coordinates
                new._coordset = None

            # Here we must reduce the corresponding coordinates
            elif axis is not None:
                dim = new._dims[axis]
                if not keepdims:
                    del new._dims[axis]
                if new.implements('NDDataset') and new._coordset and (dim in new._coordset.names):
                    idx = new._coordset.names.index(dim)
                    del new._coordset.coords[idx]

        new.history = f'Reduced dataset resulting from application of `{op}` method'
        return new

    # ..................................................................................................................
    def _op(self, f, inputs, isufunc=False):
        # Achieve an operation f on the objs

        fname = f.__name__
        inputs = list(inputs)  # work with a list of objs not tuples
        # print(fname)

        # By default the type of the result is set regarding the first obj in inputs
        # (except for some ufuncs that can return numpy arrays or masked numpy arrays
        # but sometimes we have something such as 2 * nd where nd is a NDDataset: In this case we expect a dataset.
        # Actually, the if there is at least a NDPanel in the calculation, we expect a NDPanel as a results,
        # if there is a NDDataset, but no NDPanel, then it should be a NDDataset and so on with Coords.

        # For binary function, we also determine if the function needs object with compatible units.
        # If the object are not compatible then we raise an error

        # Take the objects out of the input list and get their types and units. Additionally determine if we need to
        # use operation on masked arrays and/or on quaternion

        objtypes = []
        objunits = OrderedSet()
        returntype = None
        # isquaternion = False     (  # TODO: not yet used)
        ismasked = False
        compatible_units = (fname in self.__compatible_units)
        remove_units = (fname in self.__remove_units)

        for i, obj in enumerate(inputs):
            # type
            objtype = type(obj).__name__
            objtypes.append(objtype)
            # units
            if objtype != 'NDPanel' and hasattr(obj, 'units'):
                objunits.add(ur.get_dimensionality(obj.units))
                if len(objunits) > 1 and compatible_units:
                    objunits = list(objunits)
                    raise DimensionalityError(*objunits[::-1],
                                              extra_msg=f", Units must be compatible for the `{fname}` operator")
            # returntype
            if objtype == 'NDPanel':
                returntype = 'NDPanel'
            elif objtype == 'NDDataset' and returntype != 'NDPanel':
                returntype = 'NDDataset'
            elif objtype == 'Coord' and returntype not in ['NDPanel', 'NDDataset']:
                returntype = 'Coord'
            else:
                # only the three above type have math capabilities in spectrochempy.
                pass

            # If one of the input is hypercomplex, this will demand a special treatment
            # if objtype != 'NDPanel' and hasattr(obj, 'is_quaternion'):
            #    isquaternion = obj.is_quaternion     # TODO: not yet used
            # elif   #TODO: check if it is a quaternion scalar

            # Do we have to deal with mask?
            if hasattr(obj, 'mask') and np.any(obj.mask):
                ismasked = True

        # it may be necessary to change the object order regarding the types
        if returntype in ['NDPanel', 'NDDataset', 'Coord'] and objtypes[0] != returntype:

            inputs.reverse()
            objtypes.reverse()

            if fname in ['mul', 'multiply', 'add', 'iadd']:
                pass
            elif fname in ['truediv', 'divide', 'true_divide']:
                fname = 'multiply'
                inputs[0] = np.reciprocal(inputs[0])
            elif fname in ['isub', 'sub', 'subtract']:
                fname = 'add'
                inputs[0] = np.negative(inputs[0])
            else:
                raise NotImplementedError()

        # Now we can proceed
        obj = cpy.copy(inputs.pop(0))
        objtype = objtypes.pop(0)
        other = None
        if inputs:
            other = cpy.copy(inputs.pop(0))
            othertype = objtypes.pop(0)

        # If our first objet is a NDPanel ------------------------------------------------------------------------------
        if objtype == 'NDPanel':

            # Some ufunc can not be applied to panels
            if fname in ['sign', 'logical_not', 'isnan', 'isfinite', 'isinf', 'signbit']:
                raise TypeError(f'`{fname}` ufunc is not implemented for NDPanel objects.')

            # Iterate on all internal dataset of the panel
            datasets = []
            for k, v in obj.datasets.items():
                v._coordset = obj.coordset
                v.name = k
                if other is not None:
                    datasets.append(f(v, other))
                else:
                    datasets.append(f(v))

            # Return a list of datasets
            return datasets

        # Our first object is a NDdataset ------------------------------------------------------------------------------
        isdataset = (objtype == 'NDDataset')

        # Get the underlying data: If one of the input is masked, we will work with masked array
        if ismasked and isdataset:
            d = obj._umasked(obj.data, obj.mask)
        else:
            d = obj.data

        # Do we have units?
        # We create a quantity q that will be used for unit calculations (without dealing with the whole object)
        def reduce_(magnitude):
            if hasattr(magnitude, 'dtype'):
                if magnitude.dtype in TYPE_COMPLEX:
                    magnitude = magnitude.real
                elif magnitude.dtype == np.quaternion:
                    magnitude = as_float_array(magnitude)[..., 0]
                magnitude = magnitude.max()
            return magnitude

        q = reduce_(d)
        if hasattr(obj, 'units') and obj.units is not None:
            q = Quantity(q, obj.units)
            q = q.values if hasattr(q, 'values') else q  # case of nddataset, coord,

        # Now we analyse the other operands ---------------------------------------------------------------------------
        args = []
        otherqs = []

        # If other is None, then it is a unary operation we can pass the following
        if other is not None:

            # First the units may require to be compatible, and if thet are sometimes they may need to be rescales
            if othertype in ['NDDataset', 'Coord']:

                # rescale according to units
                if not other.unitless:
                    if hasattr(obj, 'units'):
                        # obj is a Quantity
                        if compatible_units:
                            # adapt the other units to that of object
                            other.ito(obj.units)

            # If all inputs are datasets BUT coordset mismatch.
            if isdataset and (othertype == 'NDDataset') and (other._coordset != obj._coordset):

                obc = obj.coordset
                otc = other.coordset

                # here we can have several situations:
                # -----------------------------------
                # One acceptable situation could be that we have a single value
                if other._squeeze_ndim == 0 or ((obc is None or obc.is_empty) and (otc is None or otc.is_empty)):
                    pass

                # Another acceptable situation is that we suppress the other NDDataset is 1D, with compatible
                # coordinates in the x dimension
                elif other._squeeze_ndim >= 1:
                    try:
                        assert_dataset_equal(obc[obj.dims[-1]], otc[other.dims[-1]])
                    except AssertionError as e:
                        raise CoordinateMismatchError(str(e))

                # if other is multidimentional and as we are talking about element wise operation, we assume
                # tha all coordinates must match
                if other._squeeze_ndim > 1:
                    for idx in range(obj.ndim - 2):
                        try:
                            assert_dataset_equal(obc[obj.dims[idx]], otc[other.dims[idx]])
                        except AssertionError as e:
                            raise CoordinateMismatchError(str(e))

            if othertype in ['NDDataset', 'Coord']:

                # mask?
                if ismasked:
                    arg = other._umasked(other._data, other._mask)
                else:
                    arg = other._data

            else:
                # Not a NDArray.

                # if it is a quantity than separate units and magnitude
                if isinstance(other, Quantity):
                    arg = other.m

                else:
                    # no units
                    arg = other

            args.append(arg)

            otherq = reduce_(arg)
            if hasattr(other, 'units') and other.units is not None:
                otherq = Quantity(otherq, other.units)
                otherq = otherq.values if hasattr(otherq, 'values') else otherq  # case of nddataset, coord,
            otherqs.append(otherq)

        # Calculate the resulting units (and their compatibility for such operation)
        # --------------------------------------------------------------------------------------------------------------
        # Do the calculation with the units to find the final one

        def check_require_units(fname, _units):
            if fname in self.__require_units:
                requnits = self.__require_units[fname]
                if (requnits == DIMENSIONLESS or requnits == 'radian' or requnits == 'degree') and _units.dimensionless:
                    # this is compatible:
                    _units = DIMENSIONLESS
                else:
                    if requnits == DIMENSIONLESS:
                        s = 'DIMENSIONLESS input'
                    else:
                        s = f'`{requnits}` units'
                    raise DimensionalityError(_units, requnits, extra_msg=f'\nFunction `{fname}` requires {s}')

            return _units

        # define an arbitrary quantity `q` on which to perform the units calculation

        units = UNITLESS

        if not remove_units:

            if hasattr(q, 'units'):
                q = q.m * check_require_units(fname, q.units)

            for i, otherq in enumerate(otherqs[:]):
                if hasattr(otherq, 'units'):
                    if np.ma.isMaskedArray(otherq):
                        otherqm = otherq.m.data
                    else:
                        otherqm = otherq.m
                    otherqs[i] = otherqm * check_require_units(fname, otherq.units)
                else:

                    # here we want to change the behavior a pint regarding the addition of scalar to quantity
                    #         # in principle it is only possible with dimensionless quantity, else a dimensionerror is
                    #         raised.
                    if fname in ['add', 'sub', 'iadd', 'isub', 'and', 'xor', 'or'] and hasattr(q, 'units'):
                        otherqs[i] = otherq * q.units  # take the unit of the first obj

            # some functions are not handled by pint regardings units, try to solve this here
            f_u = f
            if compatible_units:
                f_u = np.add  # take a similar function handled by pint

            try:
                res = f_u(q, *otherqs)

            except Exception as e:
                if not otherqs:
                    # in this case easy we take the units of the single argument except for some function where units
                    # can be dropped
                    res = q
                else:

                    raise e

            if hasattr(res, 'units'):
                units = res.units

        # perform operation on magnitudes
        # --------------------------------------------------------------------------------------------------------------
        if isufunc:

            with catch_warnings(record=True) as ws:

                # try to apply the ufunc
                if fname == 'log1p':
                    fname = 'log'
                    d = d + 1.
                if fname in ['arccos', 'arcsin', 'arctanh']:
                    if np.any(np.abs(d) > 1):
                        d = d.astype(np.complex128)
                elif fname in ['sqrt']:
                    if np.any(d < 0):
                        d = d.astype(np.complex128)

                if fname == "sqrt":  # do not work with masked array
                    data = d ** (1. / 2.)
                elif fname == 'cbrt':
                    data = np.sign(d) * np.abs(d) ** (1. / 3.)
                else:
                    data = getattr(np, fname)(d, *args)

                # if a warning occurs, let handle it with complex numbers or return an exception:
                if ws and 'invalid value encountered in ' in ws[-1].message.args[0]:
                    ws = []  # clear
                    # this can happen with some function that do not work on some real values such as log(-1)
                    # then try to use complex
                    data = getattr(np, fname)(d.astype(np.complex128),
                                              *args)  # data = getattr(np.emath, fname)(d, *args)
                    if ws:
                        raise ValueError(ws[-1].message.args[0])
                elif ws and 'overflow encountered' in ws[-1].message.args[0]:
                    warning_(ws[-1].message.args[0])
                elif ws:
                    raise ValueError(ws[-1].message.args[0])

            # TODO: check the complex nature of the result to return it

        else:
            # make a simple operation
            try:
                # if not isquaternion:
                data = f(d, *args)
            # else:
            # TODO: handle hypercomplex quaternion
            #    print(fname, d, args)
            #    raise NotImplementedError('operation {} not yet implemented '
            #                              'for quaternion'.format(fname))

            except Exception as e:
                raise ArithmeticError(e.args[0])

        # get possible mask
        if isinstance(data, np.ma.MaskedArray):
            mask = data._mask
            data = data._data
        else:
            mask = np.zeros_like(data, dtype=bool)

        # return calculated data, units and mask
        return data, units, mask, returntype

    # ..................................................................................................................
    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self):
            fname = f.__name__
            if hasattr(self, 'history'):
                history = f'Unary operation {fname} applied'
            else:
                history = None

            inputtype = type(self).__name__
            if inputtype == 'NDPanel':
                # if we have a NDPanel, process the ufuncs on all datasets
                datasets = self._op(f, [self])

                # recreate a panel object
                obj = type(self)
                panel = obj(*datasets, merge=True, align=None)
                panel.history = history

                # return it
                return panel

            else:

                data, units, mask, returntype = self._op(f, [self])
                return self._op_result(data, units, mask, history, returntype)

        return func

    # ..................................................................................................................
    def _check_order(self, fname, inputs):
        objtypes = []
        returntype = None
        for i, obj in enumerate(inputs):
            # type
            objtype = type(obj).__name__
            objtypes.append(objtype)
            if objtype == 'NDPanel':
                returntype = 'NDPanel'
            elif objtype == 'NDDataset' and returntype != 'NDPanel':
                returntype = 'NDDataset'
            elif objtype == 'Coord' and returntype not in ['NDPanel', 'NDDataset']:
                returntype = 'Coord'
            else:
                # only the three above type have math capabilities in spectrochempy.
                pass

        # it may be necessary to change the object order regarding the types
        if returntype in ['NDPanel', 'NDDataset', 'Coord'] and objtypes[0] != returntype:

            inputs.reverse()
            objtypes.reverse()

            if fname in ['mul', 'add', 'iadd']:
                pass
            elif fname in ['truediv', 'divide', 'true_divide']:
                fname = 'mul'
                inputs[0] = np.reciprocal(inputs[0])
            elif fname in ['isub', 'sub', 'subtract']:
                fname = 'add'
                inputs[0] = np.negative(inputs[0])
            else:
                raise NotImplementedError()

        f = getattr(operator, fname)
        return f, inputs

    # ..................................................................................................................
    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            fname = f.__name__
            if not reflexive:
                objs = [self, other]
            else:
                objs = [other, self]
            fm, objs = self._check_order(fname, objs)

            if hasattr(self, 'history'):
                history = f'Binary operation {fm.__name__} with `{get_name(objs[-1])}` has been performed'
            else:
                history = None

            inputtype = objs[0].implements()
            if inputtype == 'NDPanel':
                # if we have a NDPanel, process the ufuncs on all datasets
                datasets = self._op(fm, objs)

                # recreate a panel object
                obj = type(objs[0])
                panel = obj(*datasets, merge=True, align=None)
                panel.history = history

                # return it
                return panel

            else:
                data, units, mask, returntype = self._op(fm, objs)
                new = self._op_result(data, units, mask, history, returntype)
                return new

        return func

    # ..................................................................................................................
    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            fname = f.__name__
            if hasattr(self, 'history'):
                self.history = f'Inplace binary op: {fname}  with `{get_name(other)}` '
            # else:
            #    history = None
            objs = [self, other]
            fm, objs = self._check_order(fname, objs)

            inputtype = type(objs[0]).__name__
            if inputtype == 'NDPanel':
                # if we have a NDPanel, process the ufuncs on all datasets
                datasets = self._op(fm, objs)

                # recreate a panel object
                obj = type(objs[0])
                panel = obj(*datasets, merge=True, align=None)

                # return it
                self = panel

            else:
                data, units, mask, returntype = self._op(fm, objs)
                self._data = data
                self._units = units
                self._mask = mask

            return self

        return func

    # ..................................................................................................................
    def _op_result(self, data, units=None, mask=None, history=None, returntype=None):
        # make a new NDArray resulting of some operation

        new = self.copy()
        if returntype == 'NDDataset' and not new.implements('NDDataset'):
            from spectrochempy.core.dataset.nddataset import NDDataset

            new = NDDataset(new)

        new._data = cpy.deepcopy(data)

        # update the attributes
        new._units = cpy.copy(units)
        new._mask = cpy.copy(mask)
        if history is not None and hasattr(new, 'history'):
            new._history.append(history.strip())

        # case when we want to return a simple masked ndarray
        if returntype == 'masked_array':
            return new.masked_data

        return new


# ----------------------------------------------------------------------------------------------------------------------
# ARITHMETIC ON NDArray
# ----------------------------------------------------------------------------------------------------------------------

# unary operators
UNARY_OPS = ['neg', 'pos', 'abs']

# binary operators
CMP_BINARY_OPS = ['lt', 'le', 'ge', 'gt']

NUM_BINARY_OPS = ['add', 'sub', 'and', 'xor', 'or', 'mul', 'truediv', 'floordiv', 'pow']


# ..................................................................................................................
def _op_str(name):
    return f'__{name}__'


# ..................................................................................................................
def _get_op(name):
    return getattr(operator, _op_str(name))


class _ufunc:

    def __init__(self, name):
        self.name = name
        self.ufunc = getattr(np, name)

    def __call__(self, *args, **kwargs):
        return self.ufunc(*args, **kwargs)

    @property
    def __doc__(self):
        doc = f"""
            {unary_ufuncs()[self.name].split('->')[-1].strip()}

            wrapper of the numpy.ufunc function ``np.{self.name}'(*args, **kwargs)``.

            Parameters
            ----------
            *args : NDDataset
                |NDDataset| to pass to the numpy function.
            **kwargs : dict
                See other parameters.
                
            See Also
            --------
            np.{self.name} : The corresponding numpy ufunc.
            
            Examples
            --------
            See `np.{self.name} <https://numpy.org/doc/stable/reference/generated/numpy.{self.name}.html>`_
            """
        return doc #.strip()

def set_ufuncs(cls):

    for func in unary_ufuncs():
        setattr(cls, func, _ufunc(func))
        setattr(thismodule, func, _ufunc(func))
        thismodule.__all__ += [func]

# ..................................................................................................................
def set_operators(cls, priority=50):

    cls.__array_priority__ = priority

    # unary ops
    for name in UNARY_OPS:
        setattr(cls, _op_str(name), cls._unary_op(_get_op(name)))

    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, _op_str(name), cls._binary_op(_get_op(name)))

    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, _op_str('r' + name), cls._binary_op(_get_op(name), reflexive=True))

        setattr(cls, _op_str('i' + name), cls._inplace_binary_op(_get_op('i' + name)))


# ----------------------------------------------------------------------------------------------------------------------
# module functions
# ----------------------------------------------------------------------------------------------------------------------
# make some NDMath operation accessible from the spectrochempy API

abs = make_func_from(NDMath.abs, first='dataset')

amax = make_func_from(NDMath.amax, first='dataset')

amin = make_func_from(NDMath.amin, first='dataset')

argmax = make_func_from(NDMath.argmax, first='dataset')

argmin = make_func_from(NDMath.argmin, first='dataset')

array = make_func_from(np.ma.array, first='dataset')
array.__doc__ = """
Return a numpy masked array (i.e., other NDDataset attributes are lost.

Examples
========

>>> a = array(dataset)

equivalent to:

>>> a = np.ma.array(dataset)
or
>>> a= dataset.masked_data
"""

clip = make_func_from(NDMath.clip, first='dataset')

cumsum = make_func_from(NDMath.cumsum, first='dataset')

# diag = NDMath.diag

mean = make_func_from(NDMath.mean, first='dataset')

pipe = make_func_from(NDMath.pipe, first='dataset')

ptp = make_func_from(NDMath.ptp, first='dataset')

round = make_func_from(NDMath.round, first='dataset')

std = make_func_from(NDMath.std, first='dataset')

sum = make_func_from(NDMath.sum, first='dataset')

var = make_func_from(NDMath.var, first='dataset')

__all__ += ['abs', 'amax', 'amin', 'argmin', 'argmax', 'array', 'clip', 'cumsum',  # 'diag',
            'mean', 'pipe', 'ptp', 'round', 'std', 'sum', 'var']

# make some API functions
__all__ += ['empty_like', 'zeros_like', 'ones_like', 'full_like', 'empty', 'zeros', 'ones', 'full']

empty_like = make_func_from(NDMath.empty_like, first='dataset')
zeros_like = make_func_from(NDMath.zeros_like, first='dataset')
ones_like = make_func_from(NDMath.ones_like, first='dataset')
full_like = make_func_from(NDMath.full_like, first='dataset')
empty = make_func_from(NDMath.empty)
zeros = make_func_from(NDMath.zeros)
ones = make_func_from(NDMath.ones)
full = make_func_from(NDMath.full)


def set_api_methods(cls, methods):
    import spectrochempy as scp
    for method in methods:
        setattr(scp, method, getattr(cls, method))


# ======================================================================================================================
if __name__ == '__main__':
    pass
