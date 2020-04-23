# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
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

# ======================================================================================================================
# third-party imports
# ======================================================================================================================
import numpy as np
from warnings import catch_warnings

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from ...units.units import ur, Quantity, DimensionalityError
from .ndarray import NDArray
from .ndcomplex import NDComplexArray
from ...utils import docstrings, MaskedArray, NOMASK
from ...core import warning_, error_
from ...extern.orderedset import OrderedSet

# ======================================================================================================================
# utility
# ======================================================================================================================
thismodule = sys.modules[__name__]

get_name = lambda x: str(x.name if hasattr(x, 'name') else x)

DIMENSIONLESS = ur('dimensionless').units
UNITLESS = None
TYPEPRIORITY = {'Coord': 2, 'NDDataset': 3, 'NDPanel': 4}

# ======================================================================================================================
# function signature
# ======================================================================================================================

import types


def change_func_args(func, new_args):
    """
    Create a new func with its arguments renamed to new_args.

    """
    # based on:
    # https://stackoverflow.com/questions/20712403/creating-a-python-function-at-runtime-with-specified-argument-names
    # https://stackoverflow.com/questions/16064409/how-to-create-a-code-object-in-python
    
    code_obj = func.__code__
    new_varnames = tuple(list(new_args))
    
    new_code_obj = types.CodeType(
        code_obj.co_argcount,  # integer
        code_obj.co_kwonlyargcount,  # integer
        code_obj.co_nlocals,  # integer
        code_obj.co_stacksize,  # integer
        code_obj.co_flags,  # integer
        code_obj.co_code,  # bytes
        code_obj.co_consts,  # tuple
        code_obj.co_names,  # tuple
        new_varnames,  # tuple
        code_obj.co_filename,  # string
        code_obj.co_name,  # string
        code_obj.co_firstlineno,  # integer
        code_obj.co_lnotab,  # bytes
        code_obj.co_freevars,  # tuple
        code_obj.co_cellvars  # tuple
    )
    modified = types.FunctionType(new_code_obj, func.__globals__)
    func.__code__ = modified.__code__  # replace code portion of original


def change_first_func_args(func, new_arg):
    """ This will change the first argument of function
     to the new_arg. This is essentially useful for documentation process

    """
    code_obj = func.__code__
    new_varnames = tuple([new_arg] +
                         list(code_obj.co_varnames[
                              1:code_obj.co_argcount]))
    change_func_args(func, new_varnames)


def make_func_from(func, first=None):
    """
    Create a new func with its arguments from another func ansd a new signature

    """
    code_obj = func.__code__
    new_varnames = list(code_obj.co_varnames)
    if first:
        new_varnames[0] = first
    new_varnames = tuple(new_varnames)
    
    new_code_obj = types.CodeType(
        code_obj.co_argcount,  # integer
        code_obj.co_kwonlyargcount,  # integer
        code_obj.co_nlocals,  # integer
        code_obj.co_stacksize,  # integer
        code_obj.co_flags,  # integer
        code_obj.co_code,  # bytes
        code_obj.co_consts,  # tuple
        code_obj.co_names,  # tuple
        new_varnames,  # tuple
        code_obj.co_filename,  # string
        code_obj.co_name,  # string
        code_obj.co_firstlineno,  # integer
        code_obj.co_lnotab,  # bytes
        code_obj.co_freevars,  # tuple
        code_obj.co_cellvars  # tuple
    )
    modified = types.FunctionType(new_code_obj,
                                  func.__globals__,
                                  func.__name__,
                                  func.__defaults__,
                                  func.__closure__)
    modified.__doc__ = func.__doc__
    return modified


unary_str = """

# Unary Math operations

negative(x, [, out, where, casting, order, …])    Numerical negative, element-wise.
absolute(x, [, out, where, casting, order, …])    Calculate the absolute value element-wise.
fabs(x, [, out, where, casting, order, …])    Compute the absolute values element-wise.
rint(x, [, out, where, casting, order, …])    Round elements of the array to the nearest integer.
sign(x, [, out, where, casting, order, …])    Returns an element-wise indication of the sign of a number.
conj(x, [, out, where, casting, order, …])    Return the complex conjugate, element-wise.
exp(x, [, out, where, casting, order, …])    Calculate the exponential of all elements in the input array.
exp2(x, [, out, where, casting, order, …])    Calculate 2**p for all p in the input array.
log(x, [, out, where, casting, order, …])    Natural logarithm, element-wise.
log2(x, [, out, where, casting, order, …])    Base-2 logarithm of x.
log10(x, [, out, where, casting, order, …])    Return the base 10 logarithm of the input array, element-wise.
expm1(x, [, out, where, casting, order, …])    Calculate exp(x) - 1 for all elements in the array.
log1p(x, [, out, where, casting, order, …])    Return the natural logarithm of one plus the input array, element-wise.
sqrt(x, [, out, where, casting, order, …])    Return the non-negative square-root of an array, element-wise.
square(x, [, out, where, casting, order, …])    Return the element-wise square of the input.
cbrt(x, [, out, where, casting, order, …])    Return the cube-root of an array, element-wise.
reciprocal(x, [, out, where, casting, …])    Return the reciprocal of the argument, element-wise.
sin(x, [, out, where, casting, order, …])    Trigonometric sine, element-wise.
cos(x, [, out, where, casting, order, …])    Cosine element-wise.
tan(x, [, out, where, casting, order, …])    Compute tangent element-wise.
arcsin(x, [, out, where, casting, order, …])    Inverse sine, element-wise.
arccos(x, [, out, where, casting, order, …])    Trigonometric inverse cosine, element-wise.
arctan(x, [, out, where, casting, order, …])    Trigonometric inverse tangent, element-wise.
sinh(x, [, out, where, casting, order, …])    Hyperbolic sine, element-wise.
cosh(x, [, out, where, casting, order, …])    Hyperbolic cosine, element-wise.
tanh(x, [, out, where, casting, order, …])    Compute hyperbolic tangent element-wise.
arcsinh(x, [, out, where, casting, order, …])    Inverse hyperbolic sine element-wise.
arccosh(x, [, out, where, casting, order, …])    Inverse hyperbolic cosine, element-wise.
arctanh(x, [, out, where, casting, order, …])    Inverse hyperbolic tangent element-wise.
deg2rad(x, [, out, where, casting, order, …])    Convert angles from degrees to radians.
rad2deg(x, [, out, where, casting, order, …])    Convert angles from radians to degrees.
floor(x, [, out, where, casting, order, …])    Return the floor of the input, element-wise.
ceil(x, [, out, where, casting, order, …])    Return the ceiling of the input, element-wise.
trunc(x, [, out, where, casting, order, …])    Return the truncated value of the input, element-wise.

"""


def unary_ufuncs():
    liste = unary_str.split("\n")
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


for func in unary_ufuncs():
    setattr(thismodule, func, getattr(np, func))
    __all__ += [func]

binary_str = """

# Binary Math operations

multiply(x1, x2, [, out, where, casting, …])    Multiply arguments element-wise.
divide(x1, x2, [, out, where, casting, …])    Returns a true division of the inputs, element-wise.

maximum(x1, x2, [, out, where, casting, …])    Element-wise maximum of array elements.
minimum(x1, x2, [, out, where, casting, …])    Element-wise minimum of array elements.
fmax(x1, x2, [, out, where, casting, …])    Element-wise maximum of array elements.
fmin(x1, x2, [, out, where, casting, …])    Element-wise minimum of array elements.

add(x1, x2, [, out, where, casting, order, …])    Add arguments element-wise.
subtract(x1, x2, [, out, where, casting, …])    Subtract arguments, element-wise.

copysign(x1, x2, [, out, where, casting, …])    Change the sign of x1 to that of x2, element-wise.
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

greater(x1, x2, [, out, where, casting, …])    Return the truth value of (x1 > x2) element-wise.
greater_equal(x1, x2, [, out, where, …])    Return the truth value of (x1 >= x2) element-wise.
less(x1, x2, [, out, where, casting, …])    Return the truth value of (x1 < x2) element-wise.
less_equal(x1, x2, [, out, where, casting, …])    Return the truth value of (x1 =< x2) element-wise.
not_equal(x1, x2, [, out, where, casting, …])    Return (x1 != x2) element-wise.
equal(x1, x2, [, out, where, casting, …])    Return (x1 == x2) element-wise.

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
        NDDataset: [   0.841,    0.909,    0.141] unitless

    In this particular case (*i.e.*, `np.sin` ufuncs) , the `ds` units must be
    `unitless`, `dimensionless` or angle-units : `radians` or `degrees`,
    or an exception will be raised.


    Examples
    --------

    >>> from spectrochempy import *
    >>> dataset = NDDataset.load('mydataset.scp')
    >>> dataset             # doctest: +ELLIPSIS
    NDDataset: [[   2.057,    2.061, ...,    2.013,    2.012],
                [   2.033,    2.037, ...,    1.913,    1.911],
                ...,
                [   1.794,    1.791, ...,    1.198,    1.198],
                [   1.816,    1.815, ...,    1.240,    1.238]] a.u.
    >>> np.negative(dataset) # doctest: +ELLIPSIS
    NDDataset: [[  -2.057, ... -1.238]] a.u.


    """
    
    # copy function properties regarding units from pint.Quantity
    # This works only for pint 0.9
    
    __handled = Quantity._Quantity__handled
    __copy_units = Quantity._Quantity__copy_units
    __require_units = Quantity._Quantity__require_units
    __same_units = Quantity._Quantity__same_units
    __set_units = Quantity._Quantity__set_units
    __prod_units = Quantity._Quantity__prod_units
    __skip_other_args = Quantity._Quantity__skip_other_args
    __keep_title = ['negative', 'absolute', 'abs', 'fabs', 'rint', 'floor', 'ceil', 'trunc',
                    'add', 'subtract']
    __remove_title = ['multiply', 'divide', 'true_divide', 'floor_divide', 'mod', 'fmod', 'remainder',
                      'logaddexp', 'logaddexp2']
    _compatible_units = ['lt', 'le', 'ge', 'gt', 'add', 'sub', 'iadd', 'isub', 'maximum', 'minimum', 'fmin', 'fmax']
    __complex_funcs = ['real', 'imag', 'conjugate', 'absolute', 'conj', 'abs']
    
    # the following methods are to give NDArray based class
    # a behavior similar to np.ndarray regarding the ufuncs
    
    @property
    def __array_struct__(self):
        self._mask = self.umasked_data.mask
        return self._data.__array_struct__
    
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
            # if we have a NDPanel, process the ufuncs on all datasets
            datasets = self._op(ufunc, inputs, isufunc=True)
            
            # recreate a panel object
            obj = type(inputs[0])
            panel = obj(*datasets, merge=True, align=None)
            panel.history = history
            
            # return it
            return panel
        
        else:
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
    @docstrings.dedent
    def abs(self, inplace=False):
        """
        Returns the absolute value of a complex array.

        Parameters
        ----------
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        """
        new = self.copy()
        if not new.has_complex_dims:
            return np.fabs(new)  # not a complex, return fabs should be faster
        
        elif not new.is_quaternion:
            new = np.sqrt(new.real ** 2 + new.imag ** 2)
        else:
            new = np.sqrt(new.real ** 2 + new.part('IR') ** 2 + new.part('RI') ** 2 + new.part('II') ** 2)
            new._is_quaternion = False
        if inplace:
            self = new
        
        return new
    
    absolute = abs
    
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
    
    # ..................................................................................................................
    def diag(self, **kwargs):
        """take diagonal of a 2D array"""
        # As we reduce a 2D to a 1D we must specified which is the dimension for the coordinates to keep!
        
        if not kwargs.get("axis", kwargs.get("dims", kwargs.get("dim", None))):
            warning_('dimensions to remove for coordinates must be specified. By default the fist is kept. ')
        
        return self._reduce_method('diag', **kwargs)
    
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
            if not isinstance(amin, Quantity) :
                amin = amin  * self.units
            if not isinstance(amax, Quantity) :
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
        return mi.coords
    
    # ..................................................................................................................
    def coordmax(self, *args, **kwargs):
        """Coordinates of maximum of data along axis"""

        ma = self.max(keepdims=True)
        return ma.coords
        
    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------

    # Methods without dataset reduction
    def _method(self, op, *args, **kwargs):
    
        new = self.copy()

        if args:
            kwargs['dim']=args[0]
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
                new._coords.data = None
        
        # Here we must reduce the corresponding coordinates
        elif axis is not None:
            dim = new._dims[axis]
            if op not in ['cumsum', 'cumprod']:
                del new._dims[axis]
            if new.implements('NDDataset') and new._coords and (dim in new._coords.names):
                idx = new._coords.names.index(dim)
                del new._coords.coords[idx]
    
        new.history = f'Dataset resulting from application of `{op}` method'
        return new
    
    
    # Methods with dataset reduction
    def _reduce_method(self, op, *args, **kwargs):
        # TODO: make change to handle complex and quaternion
        new = self.copy()
        
        keepdims = kwargs.get('keepdims', False)
        keepunits = kwargs.pop('keepunits', True)
        
        if args:
            kwargs['dim']=args[0]
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
            idx =  getattr(np, op)(self.real.masked_data, **kwargs)
            idx = np.unravel_index(idx, self.shape)
            if self.ndim==1:
                idx = idx[0]
            return idx
           
        # particular case of max and min
        if axis is None and keepdims and op in ['max', 'amax', 'min', 'amin']:
            if op.startswith('a'):
                op = op[1:]
            idx =  getattr(np, "arg"+op)(self.real.masked_data)
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
            if axis is None and op in ['sum', 'prod', 'mean', 'var', 'std']:
                # delete all coordinates
                new._coords = None
            
            # Here we must reduce the corresponding coordinates
            elif axis is not None:
                dim = new._dims[axis]
                if not keepdims:
                    del new._dims[axis]
                if new.implements('NDDataset') and new._coords and (dim in new._coords.names):
                    idx = new._coords.names.index(dim)
                    del new._coords.coords[idx]

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
        isquaternion = False
        ismasked = False
        compatible_units = (fname in self._compatible_units)
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
            if objtype != 'NDPanel' and hasattr(obj, 'is_quaternion'):
                isquaternion = obj.is_quaternion
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
            if fname in ['sign', 'logical_not', 'isnan', 'isfinite', 'isinf', 'signbit', ]:
                raise TypeError(f'`{fname}` ufunc can not be applied to NDPanel objects.')
            
            # Iterate on all internal dataset of the panel
            datasets = []
            for k, v in obj.datasets.items():
                v._coords = obj.coords
                v.name = k
                if other is not None:
                    datasets.append(f(v, other))
                else:
                    datasets.append(f(v))
            
            # Return a list of datasets
            return datasets
        
        # Our first object is a NDdataset or a Coord ------------------------------------------------------------------
        
        isdataset = (objtype == 'NDDataset')
        
        # Do we have units?
        if not obj.unitless:
            units = obj.units
        else:
            units = UNITLESS
        
        # Get the underlying data: If one of the input is masked, we will work with masked array
        if ismasked and isdataset:
            d = obj._umasked(obj._data, obj.mask)
        else:
            d = obj._data
        
        # Now we analyse the other operands ---------------------------------------------------------------------------
        args = []
        argunits = []
        
        if other is not None:
            
            otherunits = UNITLESS
            
            # If inputs are all datasets
            if isdataset and (othertype == 'NDDataset') and (other._coords != obj._coords):
                # here it can be several situations:
                # One acceptable situation could be that we have a single value
                if other._squeeze_ndim == 0:
                    pass
                
                
                # or that we suppress or add a row to the whole dataset
                elif other._squeeze_ndim == 1 and obj._data.shape[-1] != other._data.size:
                    raise ValueError(
                        "coordinate's sizes do not match")
                
                
                elif other._squeeze_ndim > 1 and obj.coords and other.coords and \
                        not (obj._coords[0].is_empty and obj._coords[0].is_empty) and \
                        not np.all(obj._coords[0]._data == other._coords[0]._data):
                    raise ValueError(
                            "coordinate's values do not match")
            
            if othertype in ['NDDataset', 'Coord']:

                # rescale according to units
                if not other.unitless:
                    if hasattr(obj, 'units'):
                        # obj is a Quantity
                        if compatible_units:
                            other.ito(obj.units)
                        otherunits = other.units
                    else:
                        # obj has no dimension so we get the units of the other quantity
                        otherunits = other.units
                else:
                    otherunits = UNITLESS
                
                arg = other._data
                # mask?
                if ismasked:
                    arg = other._umasked(arg, other._mask)
            
            else:
                # Not a NDArray.
                
                # if it is a quantity than separate units and magnitude
                if isinstance(other, Quantity):
                    arg = other.magnitude
                    otherunits = other.units
                
                # no units
                else:
                    arg = other
                    otherunits = UNITLESS
            
            argunits.append(otherunits)
            args.append(arg)
        
        # Calculate the resulting units (and their compatibility for such operation)
        # --------------------------------------------------------------------------------------------------------------
        # Do the calculation with the units to found the final one
        
        def check_require_units(fname, _units):
            if fname in self.__require_units:
                requnits = self.__require_units[fname]
                if (requnits or requnits == 'radian') and _units.dimensionless:
                    # this is compatible:
                    _units = ur(requnits)
            return _units
        
        # define an arbitrary quantity `q` on which to perform the units calculation
        factor = 1.
        if units is not None:
            q = 0.999 * check_require_units(fname, units)
        else:
            q = 0.999
        
        for i, argunit in enumerate(argunits[:]):
            if argunit is not None:
                argunits[i] = 0.998 * check_require_units(fname, argunit)
            else:
                # here we want to change the behavior a pint regarding the addition of scalar to quantity
                # in principle it is only possible with dimensionless quantity, else a dimensionerror is raised.
                argunits[i] = 0.998
                if fname in ['add', 'sub', 'iadd', 'isub', 'and', 'xor', 'or'] and units is not None:
                    argunits[i] = 0.998 * check_require_units(fname, units)  # take the unit of the first obj
        
        if fname in ['fabs']:
            # units is lost for these operations: attempt to correct this behavior
            pass
        
        elif fname in ['sign', 'isnan', 'isfinite', 'isinf', 'signbit']:
            # should return a simple masked array
            units = None
            returntype = 'masked_array'
        
        else:
            if fname == 'cbrt':  # ufunc missing in pint
                q = q ** (1. / 3.)
            elif fname in ['maximum', 'minimum', 'fmax', 'fmin']:
                q = q
            else:
                if fname.startswith('log'):
                    f = np.log  # all similar regardings units
                elif fname.startswith('exp'):
                    f = np.exp  # all similar regardings units
                
                # print(f, q, *argunits)
                q = f(q, *argunits)
                
            if hasattr(q, 'units'):
                if not np.isfinite(q):
                    q = 1. * q.units
                if q==0.0:
                    q=(q.m+.1) * q.units
                qr = q.to_base_units()
                factor = np.abs(qr.m)/np.abs(q.m)
                if not np.isfinite(factor):
                    raise ZeroDivisionError
                units = qr.units
            else:
                units = UNITLESS
        
        # perform operation on magnitudes
        # --------------------------------------------------------------------------------------------------------------
        if isufunc:
            
            with catch_warnings(record=True) as ws:
                
                # try to apply the ufunc
                if fname == 'log1p':
                    fname = 'log'
                    d = d + 1.
                if fname in ['arccos', 'arcsin', 'arctanh' ]:
                    if np.any(np.abs(d) > 1):
                        d = d.astype(np.complex128)
                elif fname in ['log', 'log10', 'log2', 'sqrt']:
                    if np.any(d < 0):
                        d = d.astype(np.complex128)
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
                #if not isquaternion:
                    data = f(d, *args)
                #else:
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
        return data * factor, units, mask, returntype
    
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
                datasets = self._op(f,[self])
    
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

            inputtype = type(objs[0]).__name__
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
            else:
                history = None
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

NUM_BINARY_OPS = ['add', 'sub', 'and', 'xor', 'or',
                  'mul', 'truediv', 'floordiv', 'pow']


# ..................................................................................................................
def _op_str(name):
    return '__%s__' % name


# ..................................................................................................................
def _get_op(name):
    return getattr(operator, _op_str(name))


# ..................................................................................................................
def set_operators(cls, priority=50):
    # adapted from Xarray
    
    cls.__array_priority__ = priority
    
    # unary ops
    for name in UNARY_OPS:
        setattr(cls, _op_str(name), cls._unary_op(_get_op(name)))
    
    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, _op_str(name), cls._binary_op(_get_op(name)))
    
    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, _op_str('r' + name),
                cls._binary_op(_get_op(name), reflexive=True))
        
        setattr(cls, _op_str('i' + name),
                cls._inplace_binary_op(_get_op('i' + name)))


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

diag = make_func_from(NDMath.diag, first='dataset')

mean = make_func_from(NDMath.mean, first='dataset')

pipe = make_func_from(NDMath.pipe, first='dataset')

ptp = make_func_from(NDMath.ptp, first='dataset')

round = make_func_from(NDMath.round, first='dataset')

std = make_func_from(NDMath.std, first='dataset')

sum = make_func_from(NDMath.sum, first='dataset')

var = make_func_from(NDMath.var, first='dataset')

__all__ += ['abs',
            'amax',
            'amin',
            'argmin',
            'argmax',
            'array',
            'clip',
            'cumsum',
            'diag',
            'mean',
            'pipe',
            'ptp',
            'round',
            'std',
            'sum',
            'var',
            ]

# ======================================================================================================================
if __name__ == '__main__':
    pass
