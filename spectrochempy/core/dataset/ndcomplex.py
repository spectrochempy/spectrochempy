# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
This module implements a subclass of |NDArray| with complex/quaternion related
attributes.

"""

__all__ = ['NDComplexArray', ]

__dataset_methods__ = []

# ======================================================================================================================
# Standard python imports
# ======================================================================================================================
import itertools
import textwrap
import warnings

# ======================================================================================================================
# third-party imports
# ======================================================================================================================
import numpy as np
from traitlets import validate
from quaternion import as_float_array, as_quat_array

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from .ndarray import NDArray
from ...utils import (SpectroChemPyWarning, NOMASK, TYPE_FLOAT, TYPE_COMPLEX, insert_masked_print, docstrings)
from ...core import info_, debug_, error_, warning_
from ...units.units import Quantity

# ======================================================================================================================
# quaternion dtype
# ======================================================================================================================

typequaternion = np.dtype(np.quaternion)


# ======================================================================================================================
# NDComplexArray
# ======================================================================================================================

class NDComplexArray(NDArray):
    """
    This class provides the complex/quaternion related functionalities
    to |NDArray|

    """

    # ..................................................................................................................
    def __init__(self, data=None, **kwargs):
        # TODO take the doc from NDArray

        super().__init__(data=data, **kwargs)

    # ..................................................................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `NDComplexArray`.
        
        Rather than isinstance(obj, NDComplexArrray) use object.implements('NDComplexArray').
        
        This is useful to check type without importing the module
        
        """
        if name is None:
            return 'NDComplexArray'
        else:
            return name == 'NDComplexArray'

    # ------------------------------------------------------------------------------------------------------------------
    # validators
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @validate('_data')
    def _data_validate(self, proposal):
        # validation of the _data attribute
        # overrides the NDArray method

        data = proposal['value']

        # cast to the desired type
        if self._dtype is not None:

            if self._dtype == data.dtype:
                pass  # nothing more to do

            elif self._dtype not in [typequaternion] + list(TYPE_COMPLEX):
                data = data.astype(np.dtype(self._dtype), copy=False)

            elif self._dtype in TYPE_COMPLEX:
                data = self._make_complex(data)

            elif self._dtype == typequaternion:
                data = self._make_quaternion(data)

            # reset dtype for another use
            self._dtype = None

        # return the validated data
        if self._copy:
            return data.copy()
        else:
            return data

    # ------------------------------------------------------------------------------------------------------------------
    # read-only properties / attributes
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def has_complex_dims(self):
        """
        bool - True if at least one of the `data` array dimension is complex
        (Readonly property).
        """
        if self._data is None:
            return False

        return (self._data.dtype in TYPE_COMPLEX) or (
                self._data.dtype == typequaternion)

    # ..................................................................................................................
    @property
    def is_complex(self):
        """
        bool - True if the 'data' are complex (Readonly property).
        """
        if self._data is None:
            return False
        return self._data.dtype in TYPE_COMPLEX

    # ..................................................................................................................
    @property
    def is_quaternion(self):
        """
        bool - True if the `data` array is hypercomplex (Readonly property).
        """
        if self._data is None:
            return False
        return (self._data.dtype == typequaternion)

    # ..................................................................................................................
    @property
    def is_masked(self):
        """
        bool - True if the `data` array has masked values (Readonly property).
        """
        try:
            return super().is_masked
        except:
            if self._data.dtype == typequaternion:
                return np.any(self._mask['I'])
            else:
                raise Exception()

    # ..................................................................................................................
    @property
    def real(self):
        """
        |ndarray|, dtype:float - The array with real part of the `data` (
        Readonly property).
        """
        new = self.copy()
        if not new.has_complex_dims:
            return new
        ma = new.masked_data

        if ma.dtype in TYPE_FLOAT:
            new._data = ma
        elif ma.dtype in TYPE_COMPLEX:
            new._data = ma.real
        elif ma.dtype == typequaternion:
            # get the scalar part
            # q = a + bi + cj + dk  ->   qr = a
            new._data = as_float_array(ma)[..., 0]
        else:
            raise TypeError('dtype %s not recognized' % str(ma.dtype))

        if isinstance(ma, np.ma.masked_array):
            new._mask = ma.mask
        return new

    # ..................................................................................................................
    @property
    def imag(self):
        """
        |ndarray|, dtype:float - The array with imaginary part of the `data`
        (Readonly property).
        """
        new = self.copy()
        if not new.has_complex_dims:
            return None

        ma = new._masked_data
        if ma.dtype in TYPE_FLOAT:
            new._data = np.zeros_like(ma.data)
        elif ma.dtype in TYPE_COMPLEX:
            new._data = ma.imag.data
        elif ma.dtype == typequaternion:
            # this is a more complex situation than for real part
            # get the imaginary part (vector part)
            # q = a + bi + cj + dk  ->   qi = bi+cj+dk
            as_float_array(ma)[..., 0] = 0  # keep only the imaginary part
            new._data = ma.data
        else:
            raise TypeError('dtype %s not recognized' % str(ma.dtype))

        if isinstance(ma, np.ma.masked_array):
            new._mask = ma.mask
        return new

    # ..................................................................................................................
    @property
    def RR(self):
        """
        |ndarray|, dtype:float - The array with real part in both dimension of
        hypercomplex 2D `data` (Readonly property).
        this is equivalent to the `real` property
        """
        if self.ndim != 2:
            raise TypeError('Not a two dimensional array')
        return self.real

    # ..................................................................................................................
    @property
    def RI(self):
        """
        |ndarray|, dtype:float - The array with real-imaginary part of
        hypercomplex 2D `data` (Readonly property).
        """
        if self.ndim != 2:
            raise TypeError('Not a two dimensional array')
        return self.part('RI')

    # ..................................................................................................................
    @property
    def IR(self):
        """
        |ndarray|, dtype:float - The array with imaginary-real part of
        hypercomplex 2D `data` (Readonly property).
        """
        if self.ndim != 2:
            raise TypeError('Not a two dimensional array')
        if not self.is_quaternion:
            raise TypeError('Not a quaternion\'s array')
        return self.part('IR')

    # ..................................................................................................................
    @property
    def II(self):
        """
        |ndarray|, dtype:float - The array with imaginary-imaginary part of
        hypercomplex 2D data (Readonly property).
        """
        if self.ndim != 2:
            raise TypeError('Not a two dimensional array')
        if not self.is_quaternion:
            raise TypeError('Not a quaternion\'s array')
        return self.part('II')

    # ------------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @docstrings.dedent
    def conjugate(self, dims='x', inplace=False):
        """
        Conjugate of the NDDataset in the specified dimension

        Parameters
        ----------
        %(generic_method.parameters.dims|inplace)s

        Returns
        -------
        %(generic_method.returns.object)s
        
        See Also
        --------
        conj, real, imag, RR, RI, IR, II, part, set_complex, is_complex
        """
        if not inplace:  # default is to return a new array
            new = self.copy()
        else:
            new = self  # work inplace
    
        dims = self._get_dims_from_args()
        axis = self._get_dims_index(dims)
    
        if new.is_quaternion:
            # TODO:
            new.swapaxes(axis, -1, inplace=True)
            new._data[..., 1::2] = - new._data[..., 1::2]
            new.swapaxes(axis, -1, inplace=True)
        else:
            new._data = new._data.conj()
        return new

    # ..................................................................................................................
    def part(self, select='REAL'):
        """
        Take selected components of an hypercomplex array (RRR, RIR, ...)

        Parameters
        ----------
        select : str, optional, default='REAL'
            if 'REAL', only real part in all dimensions will be selected.
            ELse a string must specify which real (R) or imaginary (I) component
            has to be selected along a specific dimension. For instance,
            a string such as 'RRI' for a 2D hypercomplex array indicated
            that we take the real component in each dimension except the last
            one, for which imaginary component is preferred.

        Returns
        -------
        %(generic_method.returns.object)s
        """
        if not select:
            # no selection - returns inchanged
            return self

        new = self.copy()

        ma = self._masked_data

        if select == 'REAL':
            select = 'R' * self.ndim

        w = x = y = z = None

        if self.is_quaternion:
            w, x, y, z = as_float_array(ma).T
            w, x, y, z = w.T, x.T, y.T, z.T
            if select == 'R':
                ma = (w + x * 1j)
            elif select == 'I':
                ma = y + z * 1j
            elif select == 'RR':
                ma = w
            elif select == 'RI':
                ma = x
            elif select == 'IR':
                ma = y
            elif select == 'II':
                ma = z
            else:
                raise ValueError(f'something wrong: cannot interpret `{select}` for hypercomplex (quaternion) data!')

        elif self.is_complex:
            w, x = ma.real, ma.imag
            if (select == 'R') or (select == 'RR'):
                ma = w
            elif (select == 'I') or (select == 'RI'):
                ma = x
            else:
                raise ValueError(f'something wrong: cannot interpret `{select}` for complex data!')
        else:
            warnings.warn(f'No selection was performed because datasets with complex data have no `{select}` part. ',
                          SpectroChemPyWarning)

        if isinstance(ma, np.ma.masked_array):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma

        if hasattr(ma, 'mask'):
            new._mask = ma.mask
        else:
            new._mask = NOMASK

        return new

    # ..................................................................................................................
    @docstrings.dedent
    def set_complex(self, inplace=False):
        """
        Set the object data as complex.
        
        When nD-dimensional array are set to complex, we assume that it is along the first dimension. Two succesives rows
        are merged to form a complex rows. This means that the number of row must be even.
        
        If the complexity is to be applied in other dimension, either transpose/swapaxes your data before applying this
        function in order that the complex dimension is the first in the array.
        
        Parameters
        ----------
        %(generic_method.parameters.inplace)s
        
        Returns
        -------
        %(generic_method.returns)s
        
        See Also
        --------
        set_quaternion, has_complex_dims, is_complex, is_quaternion

        """
        if not inplace:  # default is to return a new array
            new = self.copy()
        else:
            new = self  # work inplace

        if new.has_complex_dims:
            # not necessary in this case, it is already complex
            return new

        new._data = new._make_complex(new._data)

        return new

    # ..................................................................................................................
    @docstrings.dedent
    def set_quaternion(self, inplace=False):
        """
        Set the object data as quaternion

        Parameters
        ----------
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        """
        if not inplace:  # default is to return a new array
            new = self.copy()
        else:
            new = self  # work inplace

        if new.dtype != typequaternion:  # not already a quaternion
            new._data = self._make_quaternion(new.data)

        return new

    set_hypercomplex = set_quaternion

    # ..................................................................................................................
    def transpose(self, *dims, inplace=False):
        # TODO get doc from NDArray

        new = super().transpose(*dims, inplace=inplace)

        if new.is_quaternion:
            # here if it is hypercomplex quaternion
            # we should interchange the imaginary part
            w, x, y, z = as_float_array(new._data).T
            q = as_quat_array(list(zip(w.T.flatten(), z.T.flatten(), y.T.flatten(), x.T.flatten())))
            new._data = q.reshape(new.shape)

        return new

    def swapaxes(self, dim1, dim2, inplace=False):
        # TODO get doc from NDArray

        new = super().swapaxes(dim1, dim2, inplace=inplace)

        # we need also to swap the quaternion
        # WARNING: this work only for 2D - when swapaxes is equivalent to a 2D transpose
        # TODO: implement something for any n-D array (n>2)
        if self.is_quaternion:
            # here if it is is_quaternion
            # we should interchange the imaginary part
            w, x, y, z = as_float_array(new._data).T
            q = as_quat_array(list(zip(w.T.flatten(), z.T.flatten(), y.T.flatten(), x.T.flatten())))
            new._data = q.reshape(new.shape)

        return new

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _str_shape(self):

        if self.is_empty:
            return '         size: 0\n'

        out = ''
        cplx = [False] * self.ndim
        if self.is_quaternion:
            cplx = [True, True]
        elif self.is_complex:
            cplx[-1] = True

        shcplx = (x for x in itertools.chain.from_iterable(list(zip(self.dims, self.shape, cplx))))

        shape = (', '.join(['{}:{}{}'] * self.ndim)).format(*shcplx).replace('False', '').replace('True', '(complex)')

        size = self.size
        sizecplx = '' if not self.has_complex_dims else " (complex)"

        out += f'         size: {size}{sizecplx}\n' if self.ndim < 2 else f'        shape: ({shape})\n'

        return out

    # ..................................................................................................................
    def _str_value(self, sep='\n', ufmt=' {:~K}',
                   header="       values: ... \n"):
        prefix = ['']
        if self.is_empty:
            return header + '{}'.format(textwrap.indent('empty', ' ' * 9))

        if self.has_complex_dims:
            # we will display the different part separately
            if self.is_quaternion:
                prefix = ['RR', 'RI', 'IR', 'II']
            else:
                prefix = ['R', 'I']

        units = ufmt.format(self.units) if self.has_units else ''

        def mkbody(d, pref, units):
            # work around printing masked values with formatting
            ds = d.copy()
            if self.is_masked:
                dtype = self.dtype
                mask_string = f'--{dtype}'
                ds = insert_masked_print(ds, mask_string=mask_string)
            body = np.array2string(
                ds, separator=' ',
                prefix=pref)
            body = body.replace('\n', sep)
            text = ''.join([pref, body, units])
            text += sep
            return text

        text = ''
        if 'I' not in ''.join(
                prefix):  # case of pure real data (not hypercomplex)
            if self._data is not None:
                data = self.umasked_data
                if isinstance(data, Quantity):
                    data = data.magnitude
                text += mkbody(data, '', units)
        else:
            for pref in prefix:
                if self._data is not None:
                    data = self.part(pref).umasked_data
                    if isinstance(data, Quantity):
                        data = data.magnitude
                    text += mkbody(data, pref, units)

        out = '          DATA \n'
        out += f'        title: {self.title}\n' if self.title else ''
        out += header
        out += '\0{}\0'.format(textwrap.indent(text.strip(), ' ' * 9))
        out = out.rstrip()  # remove the trailings '\n'
        return out

    # ..................................................................................................................
    def _make_complex(self, data):

        if data.dtype in TYPE_COMPLEX:
            return data.astype(np.complex128)

        if data.shape[1] % 2 != 0:
            raise ValueError("An array of real data to be transformed to complex must have an even number of columns!.")

        data = data.astype(np.float64)

        # to work the data must be in C order
        fortran = np.isfortran(data)
        if fortran:
            data = np.ascontiguousarray(data)

        data.dtype = np.complex128

        # restore the previous order
        if fortran:
            data = np.asfortranarray(data)
        else:
            data = np.ascontiguousarray(data)

        return data

    # ..................................................................................................................
    def _make_quaternion(self, data):

        if data.ndim % 2 != 0:
            raise ValueError("An array of data to be transformed to quaternion must be 2D.")

        if data.dtype not in TYPE_COMPLEX:
            if data.shape[1] % 2 != 0:
                raise ValueError(
                    "An array of real data to be transformed to quaternion must have even number of columns!.")
            # convert to double precision complex
            data = self._make_complex(data)

        if data.shape[0] % 2 != 0:
            raise ValueError("An array data to be transformed to quaternion must have even number of rows!.")

        r = data[::2]
        i = data[1::2]
        _data = as_quat_array(list(zip(r.real.flatten(), r.imag.flatten(), i.real.flatten(), i.imag.flatten())))
        _data = _data.reshape(r.shape)

        return _data

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def __setitem__(self, items, value):

        try:
            super().__setitem__(items, value)
        except:
            if self.ndim > 1 and self.is_quaternion:  # TODO: why not?
                raise NotImplemented("Sorry but setting values for hypercomplex array is not yet possible")


# ======================================================================================================================
if __name__ == '__main__':
    pass

# end of module
