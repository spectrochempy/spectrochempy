# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module that implements a subclass of `NDArray` with complex related attributes."""

import itertools
import textwrap

import numpy as np
from traitlets import validate

from spectrochempy.core.dataset.baseobjects.ndarray import NDArray
from spectrochempy.core.units import Quantity
from spectrochempy.utils.constants import TYPE_COMPLEX
from spectrochempy.utils.constants import TYPE_FLOAT
from spectrochempy.utils.print import insert_masked_print


# ======================================================================================
# NDComplexArray
# ======================================================================================
class NDComplexArray(NDArray):
    def __init__(self, data=None, **kwargs):
        r"""
        Provide the complex related functionalities to `NDArray`.

        It is a subclass bringing complex related attributes.

        Parameters
        ----------
        data : array of complex number.
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

            elif self._dtype not in TYPE_COMPLEX:
                data = data.astype(np.dtype(self._dtype), copy=False)

            elif self._dtype in TYPE_COMPLEX:
                data = self._make_complex(data)

        elif data.dtype not in TYPE_COMPLEX:
            data = data.astype(
                np.float64,
                copy=False,
            )  # by default data are float64 if the dtype is not fixed

        # return the validated data
        if self._copy:
            return data.copy()
        return data

    # ----------------------------------------------------------------------------------
    # read-only properties / attributes
    # ----------------------------------------------------------------------------------
    @property
    def is_complex(self):
        """True if the 'data' are complex (Readonly property)."""
        return self._data is not None and self._data.dtype in TYPE_COMPLEX

    @property
    def real(self):
        """
        The array with real component of the `data` .

        (Readonly property).
        """
        new = self.copy()
        if not new.is_complex:
            return new
        ma = new.masked_data

        if ma.dtype in TYPE_FLOAT:
            new._data = ma
        elif ma.dtype in TYPE_COMPLEX:
            new._data = ma.real
        else:
            raise TypeError(f"dtype {ma.dtype!s} not recognized")

        if isinstance(ma, np.ma.masked_array):
            new._mask = ma.mask
        return new

    R = real

    @property
    def imag(self):
        """
        The array with imaginary component of the `data` .

        (Readonly property).
        """
        new = self.copy()
        if not new.is_complex:
            return None

        ma = new.masked_data
        if ma.dtype in TYPE_FLOAT:
            new._data = np.zeros_like(ma.data)
        elif ma.dtype in TYPE_COMPLEX:
            new._data = ma.imag.data
        else:
            raise TypeError(f"dtype {ma.dtype!s} not recognized")

        if isinstance(ma, np.ma.masked_array):
            new._mask = ma.mask
        return new

    I = imag  # noqa: E741

    @property
    def limits(self):
        """Range of the data."""
        if self._data is None:
            return None

        if self.is_complex:
            return [np.min(self._data.real), np.max(self._data.imag)]

        return [np.min(self._data), np.max(self._data)]

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------

    def set_complex(self, inplace=False):
        """
        Set the object data as complex.

        When nD-dimensional array are set to complex, we assume that it is along the
        first dimension.
        Two succesives rows are merged to form a complex rows. This means that the
        number of row must be even
        If the complexity is to be applied in other dimension, either transpose/swapdims
        your data before applying this
        function in order that the complex dimension is the first in the array.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        NDComplexArray
            Same object or a copy depending on the `inplace` flag.

        Raises
        ------
        ValueError
            If data shape is incompatible with complex conversion

        See Also
        --------
        is_complex, is_complex

        """
        new = self.copy() if not inplace else self  # default is to return a new array

        if new.is_complex:
            # not necessary in this case, it is already complex
            return new

        # Validate data shape
        if new._data is None or new._data.size == 0:
            raise ValueError("Cannot convert empty array to complex")

        if new._data.shape[-1] % 2 != 0:
            raise ValueError(
                f"Last dimension must be even for complex conversion, got {new._data.shape[-1]}"
            )

        new._data = new._make_complex(new._data)

        return new

    # ----------------------------------------------------------------------------------
    # private methods
    # ----------------------------------------------------------------------------------
    def _str_cplx(self):
        cplx = [False] * self.ndim
        if self.is_complex:
            cplx[-1] = True
        return cplx

    def _str_shape(self):
        if self.is_empty:
            return "         size: 0\n"

        out = ""
        cplx = self._str_cplx()

        shcplx = list(
            itertools.chain.from_iterable(
                list(zip(self.dims, self.shape, cplx, strict=False)),
            )
        )

        shape = (
            (", ".join(["{}:{}{}"] * self.ndim))
            .format(*shcplx)
            .replace("False", "")
            .replace("True", "(complex)")
        )

        size = self.size
        sizecplx = "" if not any(cplx) else " (complex)"

        out += (
            f"         size: {size}{sizecplx}\n"
            if self.ndim < 2
            else f"        shape: ({shape})\n"
        )

        return out

    def _str_prefix(self):
        return "" if not self.is_complex else ["R", "I"]

    def _str_value(
        self,
        sep="\n",
        ufmt=" {:~P}",
        header="       values: ... \n",
    ):
        if self.is_empty:
            return header + "{}".format(textwrap.indent("empty", " " * 9))

        prefix = self._str_prefix()

        units = ufmt.format(self.units) if self.has_units else ""

        def mkbody(d, pref, units):
            # work around printing masked values with formatting
            ds = d.copy()
            if self.is_masked:
                dtype = self.dtype
                mask_string = f"--{dtype}"
                ds = insert_masked_print(ds, mask_string=mask_string)
            body = np.array2string(ds, separator=" ", prefix=pref)
            body = body.replace("\n", sep)
            text = "".join([pref, body, units])
            text += sep
            return text

        text = ""
        if "I" not in "".join(prefix):  # case of pure real data (not hypercomplex)
            if self._data is not None:
                data = self.umasked_data
                if isinstance(data, Quantity):
                    data = data.magnitude
                text += mkbody(data, "", units)
        else:
            for pref in prefix:
                if self._data is not None:
                    data = getattr(self, pref).umasked_data
                    if isinstance(data, Quantity):
                        data = data.magnitude
                    text += mkbody(data, pref, units)

        out = "          DATA \n"
        out += f"        title: {self.title}\n" if self.title else ""
        out += header
        out += "\0{}\0".format(textwrap.indent(text.strip(), " " * 9))
        return out.rstrip()  # remove the trailings '\n'

    def _make_complex(self, data):
        if data.dtype in TYPE_COMPLEX:
            return data.astype(np.complex128, copy=False)

        if data.shape[-1] % 2 != 0:
            raise ValueError(
                "An array of real data to be transformed to complex must have an even "
                "number of columns!.",
            )

        # Optimize memory usage by avoiding unnecessary copies
        fortran = np.isfortran(data)
        if not data.flags.c_contiguous and not fortran:
            data = np.ascontiguousarray(data, dtype=np.float64)
        else:
            data = data.astype(np.float64, copy=False)

        # View as complex
        data_complex = data.view(np.complex128)

        # Restore original memory layout if needed
        if fortran:
            data_complex = np.asfortranarray(data_complex)

        self._dtype = None
        return data_complex
