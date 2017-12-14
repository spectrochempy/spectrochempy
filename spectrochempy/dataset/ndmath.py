# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version 2.1)
# See full LICENSE agreement in the root directory
# =============================================================================




"""
This module implements the base `NDDataset` class.

It is largely adapted from xarray (...)

"""

# =============================================================================
# Standard python imports
# =============================================================================
import copy
import functools

# =============================================================================
# third-party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.extern.uncertainties import unumpy as unp
from spectrochempy.units.units import Quantity
from spectrochempy.dataset.ndarray import NDArray
from spectrochempy.utils import (interleave, interleaved2complex)
from spectrochempy.application import app

log = app.log

# =============================================================================
# Constants
# =============================================================================

__all__ = ['NDMath', ]

# =============================================================================
# utility
# =============================================================================

get_name = lambda x: str(x.name if hasattr(x, 'name') else x)

class NDMath(object):
    """

    Examples
    --------

    >>> from spectrochempy.api import *
    >>> source = NDDataset.load('mydataset.scp')
    >>> source             #doctest: +ELLIPSIS
    NDDataset([[    2.06,...,     1.24]])
    >>> np.negative(source) #doctest: +ELLIPSIS
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

            function to apply to the NDDataset.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the NADArray object.

        args : positional arguments passed into ``func``.

        kwargs : a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Notes
        -----

        Use ``.pipe`` when chaining together functions that expect
        on `NDDataset`.

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
    def abs(self, axis=-1):
        """
        Returns the absolute value of a complex array.

        Parameters
        ----------
        axis : int

            Optional, default: 1.

            The axis along which the absolute value should be calculated.

        Returns
        -------
        array : same type,

            Output array.


        """
        new = self.copy()
        if not new.has_complex_dims or not new.is_complex[axis]:
            return np.fabs(new)  # not a complex, return fabs should be faster

        new.swapaxes(axis, -1, inplace=True)
        new = np.sqrt(new.real ** 2 + new.imag ** 2)
        new.swapaxes(axis, -1, inplace=True)
        new._is_complex[axis] = False

        return new

    absolute = abs

    # -------------------------------------------------------------------------
    # special methods
    # -------------------------------------------------------------------------

    # the following methods are to give NDArray based class
    # a behavior similar to np.ndarray regarding the ufuncs

    # def __array_prepare(self, *args, **kwargs):
    #    pass

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

            # Check if our NDArray is actually a NDDataset
            # (it must have an attribute _coordset)
            if hasattr(obj, '_coordset'):

                # do we have uncertainties on our data ?
                # if any create an UFloat type if any uncertainty
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
        if ufunc:
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

                # TODO: check the complex nature of the result to return it

        else:
            # make a simple operation
            try:
                data = f(d, *args)
            except Exception as e:
                raise ArithmeticError(e.args[0])

            # restore interleaving of complex data
            data, iscomplex = interleave(data)

        # unpack the data
        uncertainty = unp.std_devs(data)
        data = unp.nominal_values(data).astype(float)

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

