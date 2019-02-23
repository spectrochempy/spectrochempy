# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
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
import copy
import functools

# ======================================================================================================================
# third-party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from ...units.units import Quantity
from .ndarray import NDArray
from .ndcomplex import NDComplexArray
from ...utils import getdocfrom, docstrings, MaskedArray, NOMASK, info_, debug_, warning_, error_, make_new_object

# ======================================================================================================================
# utility
# ======================================================================================================================

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

    # the following methods are to give NDArray based class
    # a behavior similar to np.ndarray regarding the ufuncs

    @property
    def __array_struct__(self):
        return self._data.__array_struct__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        # case of complex or hypercomplex data
        if self.implements(NDComplexArray) and self.has_complex_dims:

            if ufunc.__name__ in ['real',
                                  'imag',
                                  'conjugate',
                                  'absolute',
                                  'conj',
                                  'abs']:
                return getattr(inputs[0], ufunc.__name__)()

            if ufunc.__name__ in ["fabs", ]:
                # fonction not available for complex data
                error_(f"Operation `{ufunc}` does not accept complex data. Operation not applied!")
                return self

        # not a complex data
        if ufunc.__name__ in ['absolute', 'abs']:
            f = np.fabs

        data, units, mask = self._op(ufunc, inputs, isufunc=True)
        history = 'ufunc %s applied.' % ufunc.__name__

        return self._op_result(data, units, mask, history)

    def __array_wrap__(self, *args):
        # not sure we still need this
        print(args)
        raise NotImplementedError()

    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------

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

        elif not new._is_quaternion:
            new = np.sqrt(new.real ** 2 + new.imag ** 2)
        else:
            new = np.sqrt(new.real ** 2 + new.part('IR') ** 2 + new.part('RI') ** 2 + new.part('II') ** 2)
            new._is_quaternion = False
        if inplace:
            self = new

        return new

    absolute = abs

    # ..................................................................................................................
    # Non ufunc reduce functions
    # ..................................................................................................................
    def _funcs_reduce(self, op, *args, **kwargs):

        new = self.copy()

        keepdims = kwargs.pop('keepdims', True)
        keepunits = kwargs.pop('keepunits', True)

        # handle the various syntax to pass the axis
        dims = self._get_dims_from_args(*args, **kwargs)
        axis = self._get_dims_index(dims)
        axis = axis[0] if axis else None
        kwargs['axis'] = axis

        # dim and dims keyword not accepted by the np function, so remove it
        kwargs.pop('dims', None)
        kwargs.pop('dim', None)
        if op in ['diag', 'round', 'clip']:
            # also remove axis
            kwargs.pop('axis', None)

        arr = getattr(np, op)(self.masked_data, *args, **kwargs)

        if isinstance(arr, MaskedArray):
            new._data = arr.data
            new._mask = arr.mask

        elif isinstance(arr, np.ndarray):
            new._data = arr
            new._mask = NOMASK

        else:
            if new.has_units and keepunits:
                new = arr * new._units
            else:
                new = arr

        # particular case of functions that returns flatten array
        if self.ndim > 1 and axis is None and op in ['cumsum', ]:
            # delete all coordinates
            new._coords = None

        # we must reduce the corresponding coordinates
        if axis is not None and (not keepdims or op == 'diag'):
            dim = new._dims[axis]
            del new._dims[axis]
            if new.implements('NDDataset') and new._coords and (dim in new._coords.names):
                idx = new._coords.names.index(dim)
                del new._coords[idx]

        return new

    # ..................................................................................................................
    @getdocfrom(np.sum)
    def sum(self, *args, **kwargs):
        """sum along axis"""

        return self._funcs_reduce('sum', *args, **kwargs)

    @getdocfrom(np.cumsum)
    def cumsum(self, *args, **kwargs):
        """cumsum along axis"""

        return self._funcs_reduce('cumsum', *args, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.mean)
    def mean(self, *args, **kwargs):
        """mean values along axis"""

        return self._funcs_reduce('mean', *args, keepdims=False, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.var)
    def var(self, *args, **kwargs):
        """variance values along axis"""

        return self._funcs_reduce('var', *args, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.std)
    def std(self, *args, **kwargs):
        """Standard deviation values along axis"""

        return self._funcs_reduce('std', *args, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.ptp)
    def ptp(self, *args, **kwargs):
        """amplitude of data along axis"""

        return self._funcs_reduce('ptp', *args, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.all)
    def all(self, *args, **kwargs):
        """Test whether all array elements along a given axis evaluate to True."""

        return self._funcs_reduce('all', *args, keepunits=False, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.any)
    def any(self, *args, **kwargs):
        """Test whether any array elements along a given axis evaluate to True."""

        return self._funcs_reduce('any', *args, keepunits=False, **kwargs)

    sometrue = any

    # ..................................................................................................................
    @getdocfrom(np.diag)
    def diag(self, **kwargs):
        """take diagonal of a 2D array"""
        # As we reduce a 2D to a 1D we must specified which is the dimension for the coordinates to keep!

        if not kwargs.get("axis", kwargs.get("dims", kwargs.get("dim", None))):
            warning_('dimensions to remove for coordinates must be specified. By default the fist is kept. ')

        return self._funcs_reduce('diag', **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.clip)
    def clip(self, *args, **kwargs):
        """Clip (limit) the values in an array."""
        if len(args) > 2 or len(args) == 0:
            raise ValueError('Clip requires at least one or two arguments at most')
        elif len(args) == 1:
            kwargs['a_max'] = args[0]
        else:
            kwargs['a_min'], kwargs['a_max'] = args
        args = ()  # reset args
        return self._funcs_reduce('clip', *args, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.round)
    def round(self, *args, **kwargs):
        """Round an array to the given number of decimals.."""

        return self._funcs_reduce('round', *args, **kwargs)

    around = round_ = round

    # ..................................................................................................................
    # Find extrema
    # ..................................................................................................................
    def _extrema(self, op, *args, only_index=False, **kwargs):

        # as data can be complex or quaternion, it is worth to note that min and
        # max refer to the real part of the object. If one wants an extremum for
        # the absolute or the imaginary part, the input data must have been
        # prepared for this.

        # by defaut we do return a dataset
        # but if keepdims is False,  then a scalar value will be returned for
        # 1D array and a squeezed NDDataset for multidimensional arrays
        keepdims = kwargs.pop('keepdims', True)
        keepunits = kwargs.pop('keepunits', True)

        # handle the various syntax to pass the axis
        axis = self.get_axis(*args, **kwargs)
        kwargs['axis'] = axis

        # dim or dims keyword not accepted by the np function, so remove it
        kwargs.pop('dims', None)
        kwargs.pop('dim', None)

        # get the location of the extremum
        if op.startswith('a'):  # deal with the amax, amin name
            op = op[1:]
        idx = getattr(self.real.masked_data, f"arg{op}")(**kwargs)
        if axis is None:
            # unravel only when axis=None
            # (as the search was done on the flatten array)
            idx = np.unravel_index(idx, self.shape)

        # if we wants only the indexes of the extremum, return it now
        if only_index:
            if self.ndim == 1:
                idx = idx[0]
            return idx

        # now slice the array according to this indexes
        if axis is None:
            new = self[idx]
        else:
            # a little more complicated
            if self.ndim > 2:
                # TODO: for now I did not find a way to use the idx
                #      for fancy indexing of the NDDataset with ndim > 2
                raise NotImplementedError
            new = self.take(idx, dim=axis)
            new = new.diag(dim=axis)

        # return the results according to the keepdims and keepunits parameter
        if not keepdims:
            new.squeeze(inplace=True)
            arr = new.data
            if new.ndim == 0:
                arr = arr.data[()]  # keep only value
            if new.has_units and keepunits:
                new = arr * new._units
            else:
                new = arr

        return new  #

    # ..................................................................................................................
    @getdocfrom(np.amin)
    def amin(self, *args, **kwargs):
        """minimum of data along axis"""

        return self._extrema('amin', *args, **kwargs)

    min = amin

    # ..................................................................................................................
    @getdocfrom(np.amax)
    def amax(self, *args, **kwargs):
        """maximum of data along axis"""

        return self._extrema('amax', *args, **kwargs)

    max = amax

    # ..................................................................................................................
    @getdocfrom(np.argmin)
    def argmin(self, *args, **kwargs):
        """indexes of minimum of data along axis"""

        return self._extrema('min', *args, only_index=True, **kwargs)

    # ..................................................................................................................
    @getdocfrom(np.argmax)
    def argmax(self, *args, **kwargs):
        """indexes of maximum of data along axis"""

        return self._extrema('max', *args, only_index=True, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _op(f, inputs, isufunc=False):
        # achieve an operation f on the objs

        fname = f.__name__  # name of the function to use
        inputs = list(inputs)  # work with a list of objs not tuples

        # determine if the function needs compatible units
        sameunits = False
        if fname in ['lt', 'le', 'ge', 'gt', 'add', 'sub']:
            sameunits = True

        # take the first object out of the objs list
        obj = copy.deepcopy(inputs.pop(0))

        # Some flags to be set depending of the object
        isdataset = True
        # iscomplex = False
        isquaternion = False

        # and is masked ?
        ismasked = False

        # case our first object is a NDArray or a subclass of NDArray
        # (Coord or NDDataset are subclass of NDArray)
        if isinstance(obj, NDArray):

            d = obj._data  # The underlying data

            # do we have units?
            if obj.has_units:
                q = Quantity(1., obj.units)  # create a Quantity from the units
            else:
                q = 1.

            if obj.is_masked:
                ismasked = True

            # Check if our NDArray is actually a NDDataset
            if obj.implements("NDDataset"):

                # Our data may be complex or hypercomplex
                # iscomplex = obj.has_complex_dims and not obj.is_quaternion
                isquaternion = obj.is_quaternion

            else:

                # Ok it's an NDArray but not a NDDataset, then it's an Coord.
                isdataset = False

            # mask?
            if ismasked:
                d = obj._umasked(d, obj._mask)

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
        for o in inputs:
            other = copy.deepcopy(o)

            # is other a NDDataset or Coord?
            if isinstance(other, NDArray):

                # if the first arg (obj) is a nddataset
                if isdataset and other._coords != obj._coords:
                    # here it can be several situations:
                    # One acceptable situation could be that
                    # e.g., we suppress or add a row to the whole dataset
                    if other._squeeze_ndim == 1 and obj._data.shape[-1] != other._data.size:
                        raise ValueError(
                            "coordinate's sizes do not match")

                    if other._squeeze_ndim != 1 and \
                            obj.coords and other.coords and  \
                            not (obj._coords[0].is_empty and obj._coords[0].is_empty) and \
                                    not np.all(obj._coords[0]._data == other._coords[0]._data):
                        raise ValueError(
                            "coordinate's values do not match")

                # rescale according to units
                if not other.unitless:
                    if hasattr(obj, 'units'):  # obj is a Quantity
                        if sameunits:
                            other.to(obj._units,
                                     inplace=True)
                        argunits.append(Quantity(1., other._units))
                    else:
                        argunits.append(1.)
                else:
                    argunits.append(1.)

                arg = other._data

                # mask?
                arg = other._umasked(arg, other._mask)

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
        if isufunc:
            data = getattr(np, fname)(d, *args)

            # TODO: check the complex nature of the result to return it

        else:
            # make a simple operation
            try:
                if not isquaternion:
                    data = f(d, *args)
                else:
                    # TODO : handle hypercomplex quaternion
                    print(fname, d, args)
                    raise NotImplementedError('operation {} not yet implemented '
                                              'for quaternion'.format(fname))

            except Exception as e:
                raise ArithmeticError(e.args[0])

        # get possible mask
        if isinstance(data, np.ma.MaskedArray):
            mask = data._mask
            data = data._data
        else:
            mask = np.zeros_like(data, dtype=bool)

        # redo the calculation with the units to found the final one
        if fname in ['positive']:
            units = obj.units
        else:
            q = f(q, *argunits)
            if hasattr(q, 'units'):
                units = q.units
            else:
                units = None

        # determine the is_complex parameter:
        # data_iscomplex = [False] * data.ndim

        # if is_complex:
        # the resulting data are complex on the last dimension
        #    data_iscomplex[-1] = True

        # For the other dimension, this will depends on the history of the
        # objs:
        # TODO: The following will have to be carefully checked in many kind
        # of situation
        # for i in range(data.ndim)[:-1]:

        #    for item in objcomplex:
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
        #        if item:
        #            data_iscomplex[i] |= item[i]  # `or` operation

        return data, units, mask

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self):
            data, units, mask = self._op(f, [self])
            if hasattr(self, 'history'):
                history = 'unary operation %s applied' % f.__name__
            return self._op_result(data, units, mask, history)

        return func

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            if not reflexive:
                objs = [self, other]
            else:
                objs = [other, self]
            data, units, mask = self._op(f, objs)
            if hasattr(self, 'history'):
                history = 'binary operation ' + f.__name__ + \
                          ' with `%s` has been performed' % get_name(other)
            else:
                history = None
            return self._op_result(data, units, mask, history)

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            objs = [self, other]
            data, units, mask = self._op(f, objs)
            self._data = data
            self._units = units
            self._mask = mask

            self.history = 'inplace binary op : ' + f.__name__ + \
                           ' with %s ' % get_name(other)
            return self

        return func

    def _op_result(self, data, units=None, mask=None, history=None):
        # make a new NDArray resulting of some operation

        new = self.copy()

        new._data = copy.deepcopy(data)

        # update the attributes
        if units is not None:
            new._units = copy.copy(units)
        if mask is not None:
            new._mask = copy.copy(mask)
        if history is not None and hasattr(new, 'history'):
            new._history.append(history.strip())
        # if quaternion is not None:
        #    new._is_quaternion = quaternion#

        return new


if __name__ == '__main__':
    pass
