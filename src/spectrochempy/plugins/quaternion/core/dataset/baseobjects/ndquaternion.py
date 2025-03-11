# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module that implements a subclass of `NDArray` with complex/quaternion related attributes."""

__all__ = []


import numpy as np
from traitlets import Bool
from traitlets import validate

from spectrochempy.core.dataset.baseobjects.ndcomplex import NDComplexArray
from spectrochempy.utils.constants import NOMASK
from spectrochempy.utils.constants import TYPE_COMPLEX
from spectrochempy.utils.constants import TYPE_FLOAT

# Try to load quaternion support
# -----------------------------------------------------------------------------------
typequaternion = np.quaternion


# ======================================================================================
# NDQuaternionArray
# ======================================================================================
class NDQuaternionArray(NDComplexArray):
    """Class for hypercomplex/quaternion arrays."""

    _interleaved = Bool(False)

    def __init__(self, data=None, **kwargs):
        r"""
        Provide the hypercomplex/quaternion related functionalities to `NDComplexArray`.

        It is a subclass bringing  hypercomplex/quaternion related attributes.

        Parameters
        ----------
        data : array of complex number or quaternion.
            Data array contained in the object. The data can be a list, a tuple, a
            `~numpy.ndarray` , a ndarray-like,
            a  `NDArray` or any subclass of  `NDArray` . Any size or shape of data is
            accepted. If not given, an empty `NDArray` will be inited.
            At the initialisation the provided data will be eventually casted to a
            `numpy.ndarray`.
            If a subclass of  `NDArray` is passed which already contains some mask,
            labels, or units, these elements will
            be used to accordingly set those of the created object. If possible, the
            provided data will not be copied
            for `data` input, but will be passed by reference, so you should make a
            copy of the `data` before passing
            them if that's the desired behavior or set the `copy` argument to True.

        Other Parameters
        ----------------
        dims : list of chars, optional.
            if specified the list must have a length equal to the number od data
            dimensions (ndim) and the chars must be
            taken among among x,y,z,u,v,w or t. If not specified, the dimension names
            are automatically attributed in
            this order.
        name : str, optional
            A user friendly name for this object. If not given, the automatic `id`
            given at the object creation will be
            used as a name.
        labels : array of objects, optional
            Labels for the `data` . labels can be used only for 1D-datasets.
            The labels array may have an additional dimension, meaning several series
            of labels for the same data.
            The given array can be a list, a tuple, a `~numpy.ndarray` , a ndarray-like,
            a  `NDArray` or any subclass of `NDArray` .
        mask : array of bool or `NOMASK` , optional
            Mask for the data. The mask array must have the same shape as the data.
            The given array can be a list,
            a tuple, or a `~numpy.ndarray` . Each values in the array must be `False`
            where the data are *valid* and True when
            they are not (like in numpy masked arrays). If `data` is already a
            :class:`~numpy.ma.MaskedArray` , or any
            array object (such as a  `NDArray` or subclass of it), providing a `mask`
            here will causes the mask from the
            masked array to be ignored.
        units : `Unit` instance or str, optional
            Units of the data. If data is a `Quantity` then `units` is set to the unit
            of the `data`; if a unit is also
            explicitly provided an error is raised. Handling of units use the
            `pint <https://pint.readthedocs.org/>`_
            package.
        title : str, optional
            The title of the dimension. It will later be used for instance for
            labelling plots of the data.
            It is optional but recommended to give a title to each ndarray.
        dlabel :  str, optional.
            Alias of `title` .
        meta : dict-like object, optional.
            Additional metadata for this object. Must be dict-like but no
            further restriction is placed on meta.
        author : str, optional.
            name(s) of the author(s) of this dataset. BNy default, name of the computer
            note where this dataset is
            created.
        description : str, optional.
            A optional description of the nd-dataset. A shorter alias is `desc` .
        history : str, optional.
            A string to add to the object history.
        copy : bool, optional
            Perform a copy of the passed object. Default is False.

        See Also
        --------
        NDDataset : Object which subclass  `NDArray` with the addition of coordinates.

        Examples
        --------
        >>> from spectrochempy import NDComplexArray
        >>> myarray = NDComplexArray([1. + 0j, 2., 3.])
        >>> myarray
        NDComplexArray: [complex128] unitless (size: 3)

        """
        super().__init__(data=data, **kwargs)

    # ----------------------------------------------------------------------------------
    # validators
    # ----------------------------------------------------------------------------------
    @validate("_data")
    def _data_validate(self, proposal):
        # validation of the _data attribute
        # overrides the NDArray method

        data = proposal["value"]

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

        elif data.dtype not in [typequaternion] + list(TYPE_COMPLEX):
            data = data.astype(
                np.float64,
                copy=False,
            )  # by default dta are float64 if the dtype is not fixed

        # return the validated data
        if self._copy:
            return data.copy()
        return data

    # ----------------------------------------------------------------------------------
    # read-only properties / attributes
    # ----------------------------------------------------------------------------------
    @property
    def has_complex_dims(self):
        """
        True if at least one of the `data` array dimension is complex.

        (Readonly property).
        """
        if self._data is None:
            return False

        return (self._data.dtype in TYPE_COMPLEX) or (
            self._data.dtype == typequaternion
        )

    @property
    def is_quaternion(self):
        """
        True if the `data` array is hypercomplex .

        (Readonly property).
        """
        if self._data is None:
            return False
        return self._data.dtype == typequaternion

    @property
    def is_interleaved(self):
        """
        True if the `data` array is hypercomplex with interleaved data.

        (Readonly property).
        """
        if self._data is None:
            return False
        return self._interleaved  # (self._data.dtype == typequaternion)

    @property
    def is_masked(self):
        """
        True if the `data` array has masked values.

        (Readonly property).
        """
        try:
            return super().is_masked
        except Exception as e:
            if self._data.dtype == typequaternion:
                return np.any(self._mask["I"])
            raise e

    @property
    def real(self):
        """
        The array with real component of the `data` .

        (Readonly property).
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
            # get the scalar component
            # q = a + bi + cj + dk  ->   qr = a
            new._data = as_float_array(ma)[..., 0]
        else:
            raise TypeError(f"dtype {ma.dtype!s} not recognized")

        if isinstance(ma, np.ma.masked_array):
            new._mask = ma.mask
        return new

    R = real

    @property
    def qimag(self):  # TODO = revise this
        """
        The array with quaternion imaginary component of the `data` .

        (Readonly property).
        """
        new = self.copy()
        if not new.has_complex_dims:
            return None

        ma = new.masked_data
        if ma.dtype in TYPE_FLOAT:
            new._data = np.zeros_like(ma.data)
        elif ma.dtype in TYPE_COMPLEX:
            new._data = ma.imag.data
        elif ma.dtype == typequaternion:
            # this is a more complex situation than for real component
            # get the imaginary component (vector component)
            # q = a + bi + cj + dk  ->   qi = bi+cj+dk
            as_float_array(ma)[..., 0] = 0  # keep only the imaginary component
            new._data = ma  # .data
        else:
            raise TypeError(f"dtype {ma.dtype!s} not recognized")

        if isinstance(ma, np.ma.masked_array):
            new._mask = ma.mask
        return new

    I = qimag  # noqa: E741

    @property
    def RR(self):
        """
        The array with real component in both dimension of hypercomplex 2D `data` .

        This readonly property is equivalent to the `real` property.
        """
        if not self.is_quaternion:
            raise TypeError("Not an hypercomplex array")
        return self.real

    @property
    def RI(self):
        """
        The array with real-imaginary component of hypercomplex 2D `data` .

        (Readonly property).
        """
        if not self.is_quaternion:
            raise TypeError("Not an hypercomplex array")
        return self.component("RI")

    imag = RI  # TODO : docs for imag

    @property
    def IR(self):
        """
        The array with imaginary-real component of hypercomplex 2D `data` .

        (Readonly property).
        """
        if not self.is_quaternion:
            raise TypeError("Not an hypercomplex array")
        return self.component("IR")

    @property
    def II(self):
        """
        The array with imaginary-imaginary component of hypercomplex 2D data.

        (Readonly property).
        """
        if not self.is_quaternion:
            raise TypeError("Not an hypercomplex array")
        return self.component("II")

    @property
    def limits(self):
        """Range of the data."""
        if self.data is None:
            return None

        if self.is_complex:
            return [self.data.real.min(), self.data.imag.max()]
        if self.is_quaternion:
            data = as_float_array(self.data)[..., 0]
            return [data.min(), data.max()]

        return [self.data.min(), self.data.max()]

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def component(self, select="REAL"):
        """
        Take selected components of an hypercomplex array (RRR, RIR, ...).

        Parameters
        ----------
        select : str, optional, default='REAL'
            If 'REAL', only real component in all dimensions will be selected.
            ELse a string must specify which real (R) or imaginary (I) component
            has to be selected along a specific dimension. For instance,
            a string such as 'RRI' for a 2D hypercomplex array indicated
            that we take the real component in each dimension except the last
            one, for which imaginary component is preferred.

        Returns
        -------
        component
            Component of the complex or hypercomplex array.

        """
        if not select:
            # no selection - returns inchanged
            return self

        new = self.copy()

        ma = self.masked_data

        ma = get_component(ma, select)

        if isinstance(ma, np.ma.masked_array):
            new._data = ma.data
            new._mask = ma.mask
        else:
            new._data = ma

        if hasattr(ma, "mask"):
            new._mask = ma.mask
        else:
            new._mask = NOMASK

        return new

    def set_quaternion(self, inplace=False):
        """
        Set the object data as quaternion.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        out
            Same object or a copy depending on the `inplace` flag.

        """
        new = self.copy() if not inplace else self  # default is to return a new array

        if new.dtype != typequaternion:  # not already a quaternion
            new._data = new._make_quaternion(new.data)

        return new

    set_hypercomplex = set_quaternion
    set_hypercomplex.__doc__ = "Alias of set_quaternion."

    def transpose(self, *dims, inplace=False):
        """
        Transpose the complex array.

        Parameters
        ----------
        dims : int, str or tuple of int or str, optional, default=(0,)
            Dimension names or indexes along which the method should be applied.
        inplace : bool, optional, default=False
            Flag to say that the method return a new object (default)
            or not (inplace=True)

        Returns
        -------
        transposed
            Same object or a copy depending on the `inplace` flag.

        """
        new = super().transpose(*dims, inplace=inplace)

        if new.is_quaternion:
            # here if it is hypercomplex quaternion
            # we should interchange the imaginary component
            w, x, y, z = as_float_array(new._data).T
            q = as_quat_array(
                list(
                    zip(
                        w.T.flatten(),
                        y.T.flatten(),
                        x.T.flatten(),
                        z.T.flatten(),
                        strict=False,
                    ),
                ),
            )
            new._data = q.reshape(new.shape)

        return new

    def swapdims(self, dim1, dim2, inplace=False):
        """
        Swap dimension the complex array.

        swapdims and swapaxes are alias.

        Parameters
        ----------
        dims : int, str or tuple of int or str, optional, default=(0,)
            Dimension names or indexes along which the method should be applied.
        inplace : bool, optional, default=False
            Flag to say that the method return a new object (default)
            or not (inplace=True)

        Returns
        -------
        transposed
            Same object or a copy depending on the `inplace` flag.

        """
        new = super().swapdims(dim1, dim2, inplace=inplace)

        # we need also to swap the quaternion
        # WARNING: this work only for 2D - when swapdims is equivalent to a 2D transpose
        # TODO: implement something for any n-D array (n>2)
        if self.is_quaternion:
            # here if it is is_quaternion
            # we should interchange the imaginary component
            w, x, y, z = as_float_array(new._data).T
            q = as_quat_array(
                list(
                    zip(
                        w.T.flatten(),
                        y.T.flatten(),
                        x.T.flatten(),
                        z.T.flatten(),
                        strict=False,
                    ),
                ),
            )
            new._data = q.reshape(new.shape)

        return new

    # ----------------------------------------------------------------------------------
    # private methods
    # ----------------------------------------------------------------------------------
    def _str_cplx(self):
        cplx = [False] * self.ndim
        if self.is_quaternion:
            cplx = [True, True]
        elif self.is_complex:
            cplx[-1] = True
        return cplx

    def _str_prefix(self):
        if self.has_complex_dims:
            # we will display the different component separately
            return ["RR", "RI", "IR", "II"] if self.is_quaternion else ["R", "I"]
        return ""

    def _make_quaternion(self, data):
        # convert to quaternion
        if data.ndim % 2 != 0:
            raise ValueError(
                "An array of data to be transformed to quaternion must be 2D.",
            )

        if data.dtype not in TYPE_COMPLEX:
            if data.shape[1] % 2 != 0:
                raise ValueError(
                    "An array of real data to be transformed to quaternion must have "
                    "even number of columns!.",
                )
            # convert to double precision complex
            data = self._make_complex(data)

        if data.shape[0] % 2 != 0:
            raise ValueError(
                "An array data to be transformed to quaternion must have even number "
                "of rows!.",
            )

        r = data[::2]
        i = data[1::2]

        self._dtype = None  # reset dtype
        return as_quaternion(r, i)

    # ----------------------------------------------------------------------------------
    # special methods
    # ----------------------------------------------------------------------------------
    def __setitem__(self, items, value):
        keys = self._make_index(items)

        if self._data.dtype == np.dtype(np.quaternion) and np.isscalar(value):
            # sometimes do not work directly : here is a work around
            self._data[keys] = np.full_like(self._data[keys], value).astype(
                np.dtype(np.quaternion),
            )
            return

        super().__setitem__(items, value)
