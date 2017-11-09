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


"""
This module implements the base `NDDataset` class.

It is largely adapted from xarray (...)

"""

# =============================================================================
# Standard python imports
# =============================================================================
import copy
import functools
import logging
import operator

# =============================================================================
# third-party imports
# =============================================================================
import numpy as np

# =============================================================================
# Local imports
# =============================================================================
from spectrochempy.extern.uncertainties import unumpy as unp
from spectrochempy.core.units import Quantity
from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.utils import interleave, interleaved2complex

# =============================================================================
# Constants
# =============================================================================

__all__ = ['NDMath', ]

_classes = ['NDMath']

from spectrochempy.application import log


class NDMath(object):
    """
    Examples
    --------

    >>> from spectrochempy.api import *
    >>> source = NDDataset.load('mydataset.scp')
    >>appource             #doctest: +ELLIPSIS
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
                raise ValueError('%s is both the pipe target and a keyword '
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

            if self.is_complex[-1] and \
                            f.__name__ in ['real', 'imag',
                                           'conjugate', 'absolute',
                                           'conj', 'abs']:
                return getattr(objs[0], f.__name__)()

            if self.is_complex[-1] and f.__name__ in ["fabs", ]:
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
                    iscomplex = obj.is_complex[-1]

                objcomplex.append(obj.is_complex)

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

                # if isaxe:
                #     raise TypeError('when the first argument is an Coord, '
                #                      'second argument cannot be an Coord or '
                #                      'NDDataset instance')
                #
                # if isdataset and not hasattr(other, '_axes'):
                #     raise TypeError('when the first argument is a NDDataset, '
                #                     'second argument cannot be an Coord'
                #                     ' instance')
                # if the first arg (obj) is a nddataset
                if isdataset and other._coordset != obj._coordset:
                    # here it can be several situations
                    # One acceptable is that e.g., we suppress or add
                    # a row to the whole dataset
                    #TODO: go a little further on these checking
                    if not( other.squeeze().ndim < obj.squeeze().ndim  and \
                        other.shape[-1] == obj.shape[-1]):
                        raise ValueError("coordset properties do not match")

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
                                other.is_complex is not None:
                    if other.is_complex[-1]:
                        # pack arg to complex
                        arg = interleaved2complex(arg)

                    objcomplex.append(other.is_complex)

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
            # make a simple opration
            data = f(d, *args)

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
                          ' with `%s` has been performed' % str(other)
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
                           ' with %s ' % str(other)
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


# =============================================================================
# ARITHMETIC ON NDDATASET
# =============================================================================

# unary operators
UNARY_OPS = ['neg', 'pos', 'abs']

# binary operators
CMP_BINARY_OPS = ['lt', 'le', 'ge', 'gt']

NUM_BINARY_OPS = ['add', 'sub', 'and', 'xor', 'or',
                  'mul', 'truediv', 'floordiv', 'pow']


def _op_str(name):
    return '__%s__' % name


def _get_op(name):
    return getattr(operator, _op_str(name))


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
