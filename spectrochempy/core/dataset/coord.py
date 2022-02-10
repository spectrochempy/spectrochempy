# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

"""
This module implements the class |Coord|.
"""

__all__ = ["Coord", "LinearCoord", "CoordSet", "trim_ranges"]

import copy as cpy
import warnings
import uuid
import textwrap

import numpy as np
import traitlets as tr

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndmath import NDMath, _set_operators
from spectrochempy.core.units import Quantity, ur, encode_quantity
from spectrochempy.core import error_, warning_
from spectrochempy.optional import import_optional_dependency

from spectrochempy.utils import (
    DEFAULT_DIM_NAME,
    colored_output,
    NOMASK,
    INPLACE,
    spacing_,
    is_sequence,
    convert_to_html,
    deprecated,
    SpectroChemPyWarning,
    InvalidDimensionNameError,
    InvalidCoordinatesTypeError,
    InvalidCoordinatesSizeError,
)
from spectrochempy.utils.traits import Range


# ======================================================================================
# Coord
# ======================================================================================


class Coord(NDMath, NDArray):
    """
    Explicit coordinates for a dataset along a given axis.

    The coordinates of a |NDDataset| can be created using the |Coord|
    object.
    This is a single dimension array with either numerical (float)
    values or labels (str, `Datetime` objects, or any other kind of objects) to
    represent the coordinates. Only a one numerical axis can be defined,
    but labels can be multiple.

    Parameters
    ----------
    data : ndarray, tuple or list
        The actual data array contained in the |Coord| object.
        The given array (with a single dimension) can be a list,
        a tuple, a |ndarray|, or a |ndarray|-like object.
        If an object is passed that contains labels, or units,
        these elements will be used to accordingly set those of the
        created object.
        If possible, the provided data will not be copied for `data` input,
        but will be passed by reference, so you should make a copy the
        `data` before passing it in the object constructor if that's the
        desired behavior or set the `copy` argument to True.
    **kwargs
        Optional keywords parameters. See other parameters.

    Other Parameters
    ----------------
    dtype : str or dtype, optional, default=np.float64
        If specified, the data will be cast to this dtype, else the
        type of the data will be used.
    dims : list of chars, optional.
        if specified the list must have a length equal to the number od
        data dimensions (ndim) and the chars must be
        taken among x,y,z,u,v,w or t. If not specified,
        the dimension names are automatically attributed in
        this order.
    name : str, optional
        A user-friendly name for this object. If not given,
        the automatic `id` given at the object creation will be
        used as a name.
    labels : array of objects, optional
        Labels for the `data`. labels can be used only for 1D-datasets.
        The labels array may have an additional dimension, meaning
        several series of labels for the same data.
        The given array can be a list, a tuple, a |ndarray|,
        a ndarray-like, a |NDArray| or any subclass of
        |NDArray|.
    units : |Unit| instance or str, optional
        Units of the data. If data is a |Quantity| then `units` is set
        to the unit of the `data`; if a unit is also
        explicitly provided an error is raised. Handling of units use
        the `pint <https://pint.readthedocs.org/>`_
        package.
    long_name : str, optional
        The long_name of the dimension. It will later be used for instance
        for labelling plots of the data.
        It is optional but recommended giving a long_name to each ndarray.
    dlabel :  str, optional
        Alias of `long_name`.
    meta : dict-like object, optional
        Additional metadata for this object. Must be dict-like but no
        further restriction is placed on meta.
    copy : bool, optional
        Perform a copy of the passed object. Default is False.
    linear : bool, optional
        If set to True, the coordinate is considered as a
        ``LinearCoord`` object.
    offset : float, optional
        Only used is linear is True.
        If omitted a value of 0.0 is taken for the coordinate offset.
    increment : float, optional
        Only used if linear is true.
        If omitted a value of 1.0 is taken for the coordinate increment.

    See Also
    --------
    NDDataset : Main SpectroChemPy object: an array with masks, units and
                coordinates.
    LinearCoord : linear coordinates.

    Examples
    --------

    We first import the object from the api :
    >>> from spectrochempy import Coord

    We then create a numpy |ndarray| and use it as the numerical `data`
    axis of our new |Coord| object.
    >>> c0 = Coord.arange(1., 12., 2., long_name='frequency', units='Hz')
    >>> c0
    Coord: [float64] Hz (size: 6)

    We can take a series of str to create a non-numerical but labelled
    axis :
    >>> tarr = list('abcdef')
    >>> tarr
    ['a', 'b', 'c', 'd', 'e', 'f']

    >>> c1 = Coord(labels=tarr, long_name='mylabels')
    >>> c1
    Coord: [labels] [  a   b   c   d   e   f] (size: 6)
    """

    _copy = tr.Bool()

    _html_output = False
    _parent_dim = tr.Unicode(allow_none=True)

    # For linear data generation
    _offset = tr.Union((tr.CFloat(), tr.CInt(), tr.Instance(Quantity)))
    _increment = tr.Union((tr.CFloat(), tr.CInt(), tr.Instance(Quantity)))
    _size = tr.Integer(0)
    _linear = tr.Bool(False)

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):

        super().__init__(data=data, **kwargs)

        if len(self.shape) > 1:
            raise ValueError("Only one 1D arrays can be used to define coordinates")

        self._linear = kwargs.pop("linear", False)
        self._increment = kwargs.pop("increment", 1.0)
        self._offset = kwargs.pop("offset", 0.0)
        self._size = kwargs.pop("size", 0)
        # self._accuracy = kwargs.pop('accuracy', None)

    # ------------------------------------------------------------------------
    # readonly property
    # ------------------------------------------------------------------------

    # ..........................................................................
    @property
    def reversed(self):
        """bool - Whether the axis is reversed (readonly
        property).
        """
        if self.units == "ppm":
            return True
        elif self.units == "1 / centimeter" and "raman" not in self.long_name.lower():
            return True
        return False

        # Return a correct result only if the data are sorted  # return  # bool(self.data[0] > self.data[-1])

    @property
    def data(self):
        """
        The `data` array (|ndarray|).

        If there is no data but labels, then the labels are returned instead of data.
        """
        if self.linear:
            data = np.arange(self.size) * self._increment + self._offset
            # if hasattr(data, "units"):
            #    data = data.m
        else:
            data = self._data

        # if self.is_dt64:
        #    data = data.astype('timedelta64[m]')

        return data

    @data.setter
    def data(self, data):

        self._set_data(data)

    @property
    def default(self):
        # this is in case default is called on a coord,
        # while it is a coordset property
        return self

    # ------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docstring)
    # some property of NDArray has to be hidden because they
    # are not useful for this Coord class
    # ------------------------------------------------------------------------

    # NDarray methods

    # ..........................................................................
    @property
    def is_complex(self):
        return False  # always real

    # ..........................................................................
    @property
    def is_empty(self):
        """
        True if the `data` array is empty or size=0, and if no label are present
        - Readonly property (bool).
        """
        if not self.linear:
            return super().is_empty

        return False

    # ..........................................................................
    @property
    def ndim(self):
        if self.linear:
            return 1
        ndim = super().ndim
        if ndim > 1:
            raise ValueError("Coordinate's array should be 1-dimensional!")
        return ndim

    # ..........................................................................
    @property
    def T(self):  # no transpose
        return self

    # ..........................................................................
    # @property
    # def values(self):
    #    return super().values

    # ..........................................................................
    def to(self, other, inplace=False, force=False):

        if self.is_dt64:
            warning_("method `to` cannot be used with datetime object. Ignored!")
            return self

        else:
            new = super().to(other, force=force)

        # else:
        #     units = get_units(other)
        #     try:
        #         to_dt64_units(units)
        #     except DimensionalityError:
        #         raise DimensionalityError(self.units, units,
        #                                   extra_msg=f". Valid units values are  {[v for v in DT64_UNITS.values()]}")
        #     new = self.copy()
        #     new._units = units

        if inplace:
            self._units = new._units
            self._long_name = new._long_name

            if self.linear:
                self._offset = new._offset
                self._increment = new._increment
                self._linear = new._linear
            else:
                self._data = new._data

            return self
        else:
            return new

    to.__doc__ = NDArray.to.__doc__

    # ..........................................................................
    @property
    def masked_data(self):
        return super().masked_data

    # ..........................................................................
    @property
    def is_masked(self):
        return False

    # ..........................................................................
    @property
    def linear(self):
        """
        Flag to specify if the data can be constructed using a linear variation (bool).
        """
        return self._linear

    @linear.setter
    def linear(self, val):

        self._linear = (
            val  # it val is true this lead to a linearisation (  # see observe)
        )

        # if val and self._data is not None:  #     # linearisation of the data, if possible  #     self._linearize()

    # ..........................................................................
    @property
    def offset(self):
        """
        Starting value for linear array
        """
        return self._offset

    # ..........................................................................
    @offset.setter
    def offset(self, val):
        if isinstance(val, Quantity):
            if self.has_units:
                val.ito(self.units)
                val = val.m
            else:
                self.units = val.units
                val = val.m
        self._offset = val

    # ..........................................................................
    @property
    def offset_value(self):
        offset = self.offset
        if self.units:
            return Quantity(offset, self._units)
        else:
            return offset

    # ..........................................................................
    @property
    def mask(self):
        return NOMASK

    # ..........................................................................
    @mask.setter
    def mask(self, val):
        # Coordinates cannot be masked. Set mask always to NOMASK
        self._mask = NOMASK

    # NDmath methods

    # ..........................................................................
    def cumsum(self, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def mean(self, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def pipe(self, func=None, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def remove_masks(self, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    @property
    def size(self):
        """
        Size of the underlying `data` array - Readonly property (int).
        """

        if self.linear:
            # the provided size is returned i or its default
            return self._size
        else:
            return super().size

    # ..........................................................................
    @property
    def shape(self):
        if self.linear:
            return (self._size,)
        return super().shape

    # ..........................................................................
    def std(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def sum(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def swapdims(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def swapaxes(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    @property
    def spacing(self):
        """
        Return coordinates spacing.

        It will be a scalar if the coordinates are uniformly spaced,
        else an array of the different spacings
        """
        if self.linear:
            return self.increment * self.units
        return spacing_(self.data) * self.units

    # ..........................................................................
    def squeeze(self, *args, **kwargs):
        return self

    # ..........................................................................
    def random(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def empty(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def empty_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def var(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def ones(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def ones_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def full(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def diag(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def diagonal(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def full_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def identity(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def eye(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def zeros(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def zeros_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def coordmin(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def coordmax(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def conjugate(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def conj(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def abs(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def absolute(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def all(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def any(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def argmax(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def argmin(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def asfortranarray(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def astype(self, dtype=None, **kwargs):
        """
        Cast the data to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        """
        if not self.linear:
            self._data = self._data.astype(dtype, **kwargs)
        else:
            self._increment = np.array(self._increment).astype(dtype, **kwargs)[()]
            self._offset = np.array(self._offset).astype(dtype, **kwargs)[()]
        return self

    # ..........................................................................
    def average(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def clip(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    def get_axis(self, *args, **kwargs):
        return super().get_axis(*args, **kwargs)

    # ..........................................................................
    @property
    def origin(self, *args, **kwargs):
        raise NotImplementedError

    # ..........................................................................
    @property
    def author(self):
        return None

    @property
    def descendant(self):
        return (self.data[-1] - self.data[0]) < 0

    # ..........................................................................
    @property
    def dims(self):
        return ["x"]

    # ..........................................................................
    @property
    def is_1d(self):
        return True

    # ..........................................................................
    def transpose(self, **kwargs):
        return self

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------
    def loc2index(self, loc):
        """
        Return the index corresponding to a given location.

        Parameters
        ----------
        loc : float.
            Value corresponding to a given location on the coordinate's axis.

        Returns
        -------
        index : int.
            The corresponding index.

        Examples
        --------

        >>> dataset = scp.NDDataset.read("irdata/nh4y-activation.spg")
        >>> dataset.x.loc2index(1644.0)
        4517
        """
        idx = self._loc2index(loc)
        if isinstance(idx, tuple):
            idx, _ = idx
            # warnings.warn(warning)
        return idx

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------
    # ..........................................................................
    def __copy__(self):
        res = self.copy(deep=False)  # we keep name of the coordinate by default
        res.name = self.name
        return res

    # ..........................................................................
    def __deepcopy__(self, memo=None):
        res = self.copy(deep=True, memo=memo)
        res.name = self.name
        return res

    # ..........................................................................
    def __dir__(self):
        # remove some methods with respect to the full NDArray
        # as they are not useful for Coord.
        return [
            "data",
            "labels",
            "units",
            "meta",
            "long_name",
            "name",
            "offset",
            "increment",
            "linear",
            "roi",
            "acquisition_date",
        ]

    def __getattr__(self, item):

        if (
            item
            in [
                "__numpy_ufunc__",
                "interface",
                "_pytestfixturefunction",
                "__dataclass_fields__",
                "_ipython_canary_method_should_not_exist_",
                "_baseclass",
                "_fill_value",
                "_ax_lines",
                "_axcb",
                "clevels",
                "__wrapped__",
                "coords",
                "__await__",
                "__aiter__",
                "dtype",
            ]
            or "_validate" in item
            or "_changed" in item
        ):
            # raise an error so that traits, ipython operation and more ... will be handled correctly
            raise AttributeError

        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(f"Can't find key/attribute `{item}`.")

    # ..........................................................................
    def __getitem__(self, items, **kwargs):

        if isinstance(items, list):
            # Special case of fancy indexing
            items = (items,)

        # choose, if we keep the same or create new object
        inplace = False
        if isinstance(items, tuple) and items[-1] == INPLACE:
            items = items[:-1]
            inplace = True

        # Eventually get a better representation of the indexes
        try:
            keys = self._make_index(items)
        except IndexError:
            # maybe it is a metadata
            if items in self.meta.keys():
                return self.meta[items]
            raise KeyError

        # init returned object
        new = self if inplace else self.copy()

        # slicing by index of all internal array
        if new.data is not None:
            udata = new.data[keys]

            if new.linear:
                # if self.increment > 0:
                #     new._offset = udata.min()
                # else:
                #     new._offset = udata.max()
                new._size = udata.size
                if new._size > 1:
                    inc = np.diff(udata)
                    variation = (inc.max() - inc.min()) / udata.ptp()
                    if variation < 1.0e-5:
                        new._increment = np.mean(inc)  # np.round(np.mean(
                        # inc), 5)
                        new._offset = udata[0]
                        new._data = None
                        new._linear = True
                    else:
                        new._linear = False
                else:
                    new._linear = False

            if not new.linear:
                new._data = np.asarray(udata)

        if self.is_labeled:
            # case only of 1D dataset such as Coord
            new._labels = np.array(self._labels[keys])

        if new.is_empty:
            error_(
                f"Empty array of shape {new._data.shape} resulted from slicing.\n"
                f"Check the indexes and make sure to use floats for location slicing"
            )
            new = None

        new._mask = NOMASK

        # we need to keep the names when copying coordinates to avoid later
        # problems
        new.name = self.name
        return new

    # ..........................................................................
    def __setitem__(self, items, value):

        if self.linear:
            error_("Linearly defined array are readonly")
            return

        super().__setitem__(items, value)

    # ..........................................................................
    def __str__(self):
        return repr(self)

    # ..........................................................................
    def _cstr(self, header="  coordinates: ... \n", print_size=True, **kwargs):

        indent = kwargs.get("indent", 0)

        out = ""
        if not self.is_empty and print_size:
            out += f"{self._str_shape().rstrip()}\n"
        out += f"    long_name: {self.long_name}\n" if self.long_name else ""
        if self.has_data:
            out += "{}\n".format(self._str_value(header=header))
        elif self.is_empty and not self.is_labeled:
            out += header.replace("...", "\0Undefined\0")

        if self.is_labeled:
            header = "       labels: ... \n"
            text = str(self.labels.T).strip()
            if "\n" not in text:  # single line!
                out += header.replace("...", "\0\0{}\0\0".format(text))
            else:
                out += header
                out += "\0\0{}\0\0".format(textwrap.indent(text.strip(), " " * 9))

        if out[-1] == "\n":
            out = out[:-1]

        if indent:
            out = "{}".format(textwrap.indent(out, " " * indent))

        first_indent = kwargs.get("first_indent", 0)
        if first_indent < indent:
            out = out[indent - first_indent :]

        if not self._html_output:
            return colored_output(out)
        else:
            return out

    # ..........................................................................
    def __repr__(self):
        out = self._repr_value().rstrip()
        return out

    # ------------------------------------------------------------------------
    # Private properties and methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    @tr.default("_increment")
    def _increment_default(self):
        return 1.0

    # ..........................................................................
    def _linearize(self):

        if not self.linear or self._data is None:
            return

        self._linear = False  # to avoid action of the observer

        if self._squeeze_ndim > 1:
            error_("Linearization is only implemented for 1D data")
            return

        data = self._data.squeeze()

        # try to find an increment
        if data.size > 1:
            inc = np.diff(data)
            variation = (inc.max() - inc.min()) / data.ptp()
            if variation < 1.0e-5:
                self._increment = (
                    data.ptp() / (data.size - 1) * np.sign(inc[0])
                )  # np.mean(inc)  # np.round(np.mean(inc), 5)
                self._offset = data[0]
                self._size = data.size
                self._data = None
                self._linear = True
            else:
                self._linear = False
        else:
            self._linear = False

    # ..........................................................................
    @tr.default("_offset")
    def _offset_default(self):
        return 0

    # ..........................................................................
    def _to_xarray(self):

        xr = import_optional_dependency("xarray")

        var = xr.Variable(dims=self.name, data=np.array(self.data))
        # dtype=np.float64))

        var.attrs["name"] = self.name
        var.attrs["pint_units"] = str(self.units)  # change name of attr
        # To not override a builtin units attribute.
        var.attrs["long_name"] = self.long_name
        var.attrs["offset"] = self.offset
        var.attrs["increment"] = self.increment
        var.attrs["linear"] = self.linear
        var.attrs["size"] = self.size
        var.attrs["roi"] = self.roi
        for k, v in self.meta.items():
            var.attrs[f"meta_{k}"] = v

        var.attrs = encode_quantity(var.attrs)

        coordinates = {self.name: var}

        # auxiliary coordinates
        if self.is_labeled:
            for level in range(self.labels.shape[-1]):
                label = self.labels[:, level]
                label = list(map(str, label.tolist()))
                label = xr.Variable(dims=self.name, data=label)
                coordinates[f"{self.name}_labels_{level}"] = label

        return coordinates
        # TODO: add multiple coordinates

    # ..........................................................................
    def _set_data(self, data):

        if data is None:
            return

        elif isinstance(data, Coord) and data.linear:
            # Case of LinearCoord
            for attr in self.__dir__():
                try:
                    if attr in ["linear", "offset", "increment"]:
                        continue
                    if attr == "data":
                        val = data.data
                    else:
                        val = getattr(data, f"_{attr}")
                    if self._copy:
                        val = cpy.deepcopy(val)
                    setattr(self, f"_{attr}", val)
                except AttributeError:
                    # some attribute of NDDataset are missing in NDArray
                    pass
            try:
                self.history = f"Copied from object:{data.name}."
            except AttributeError:
                pass

        elif isinstance(data, NDArray):
            # init data with data from another NDArray or NDArray's subclass
            # No need to check the validity of the data
            # because the data must have been already
            # successfully initialized for the passed NDArray.data
            for attr in self.__dir__():
                try:
                    val = getattr(data, f"_{attr}")
                    if self._copy:
                        val = cpy.deepcopy(val)
                    setattr(self, f"_{attr}", val)
                except AttributeError:
                    # some attribute of NDDataset are missing in NDArray
                    pass
            try:
                self.history = f"Copied from object:{data.name}."
            except AttributeError:
                pass

        elif isinstance(data, Quantity):
            self._data = np.array(data.magnitude, subok=True, copy=self._copy)
            self._units = data.units

        elif hasattr(data, "mask"):
            # an object with data and mask attributes
            self._data = np.array(data.data, subok=True, copy=self._copy)
            if isinstance(data.mask, np.ndarray) and data.mask.shape == data.data.shape:
                self.mask = np.array(data.mask, dtype=np.bool_, copy=False)

        elif (
            not hasattr(data, "shape")
            or not hasattr(data, "__getitem__")
            or not hasattr(data, "__array_struct__")
        ) and not isinstance(data, (list, tuple)):
            # Data doesn't look like a numpy array, try converting it to
            # one. Non-numerical input are converted to an array of objects.
            self._data = np.array(data, subok=True, copy=False)

        else:
            try:
                data = np.array(data, subok=True, copy=self._copy)

                if data.dtype.kind == "O":  # likely None value
                    self._data = data.astype(float)

                elif data.dtype.kind == "m":  # timedelta64
                    data, units = self._data_and_units_from_td64(data)
                    self._dtype = None
                    self._data, self._units = data, units
                    if "acquisition" in self._long_name.lower():
                        self._long_name = "time"
                elif data.dtype.kind == "M":  # datetime64:
                    self._acquisition_date = data.min()
                    self._data = data
            except ValueError:
                # happens if data is a list of quantities
                if isinstance(data[0], Quantity):
                    self._data = np.array([d.m for d in data], subok=True, copy=False)
                    self._units = data[0].units

        if self.linear:
            # we try to replace data by only an offset and an increment
            self._linearize()

    @staticmethod
    def _unittransform(new, units):
        oldunits = new.units

        if not new.linear:
            udata = (new.data * oldunits).to(units)
            new._data = udata.m
            new._units = udata.units
        else:
            offset = (new.offset * oldunits).to(units)
            increment = (new.increment * oldunits).to(units)
            new._offset = offset.m
            new._increment = increment.m
            new._units = increment.units

        if new._roi is not None:
            roi = (np.array(new._roi) * oldunits).to(units)
            new._roi = list(roi)

        return new

    # ------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------
    # ..........................................................................
    @tr.observe(tr.All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually
        #   'change'
        # }

        if change.name in ["_linear", "_increment", "_offset", "_size"]:
            if self._linear:
                self._linearize()
            return

    # ..........................................................................
    @property
    def increment(self):
        return self._increment

    @increment.setter
    def increment(self, val):
        if isinstance(val, Quantity):
            if self.has_units:
                val.ito(self.units)
                val = val.m
            else:
                self.units = val.units
                val = val.m
        self._increment = val

    @property
    def increment_value(self):
        increment = self.increment
        if self.units:
            return Quantity(increment, self._units)
        else:
            return increment

    def copy(self, deep=True, keepname=False, **kwargs):

        new = super().copy(deep=True, keepname=False, **kwargs)
        if new.linear and new.implements("Coord"):
            # we also need to set the size
            new._size = self.data.size
        return new


# ======================================================================================
# Coord
# ======================================================================================


class LinearCoord(Coord):
    """
    Linear coordinates.

    Such coordinates correspond to an ascending or descending linear
    sequence of values, fully determined by two parameters, i.e., an offset (off) and an increment (inc) :

    .. math::

        \\mathrm{data} = i*\\mathrm{inc} + \\mathrm{off}.

    Parameters
    ----------
    data : a 1D array-like object, optional
        WWen provided, the `size` parameters is adjusted to the size of
        the array, and a linearization of the
        array is performed (only if it is possible: regular spacing in
        the 1.e5 relative accuracy).
    offset : float, optional
        If omitted a value of 0.0 is taken for the coordinate offset.
    increment : float, optional
        If omitted a value of 1.0 is taken for the coordinate increment.
    **kwargs
        Optional keywords parameters. See other parameters.

    Other Parameters
    ----------------
    dtype : str or dtype, optional, default=np.float64
        If specified, the data will be cast to this dtype, else the
        type of the data will be used
    dims : list of chars, optional.
        if specified the list must have a length equal to the number od
        data dimensions (ndim) and the chars must be
        taken among x,y,z,u,v,w or t. If not specified,
        the dimension names are automatically attributed in
        this order.
    name : str, optional
        A user-friendly name for this object. If not given,
        the automatic `id` given at the object creation will be
        used as a name.
    labels : array of objects, optional
        Labels for the `data`. labels can be used only for 1D-datasets.
        The labels array may have an additional dimension, meaning
        several series of labels for the same data.
        The given array can be a list, a tuple, a |ndarray|,
        a ndarray-like, a |NDArray| or any subclass of
        |NDArray|.
    units : |Unit| instance or str, optional
        Units of the data. If data is a |Quantity| then `units` is set
        to the unit of the `data`; if a unit is also
        explicitly provided an error is raised. Handling of units use
        the `pint <https://pint.readthedocs.org/>`_
        package.
    long_name : str, optional
        The long_name of the dimension. It will later be used for instance
        for labelling plots of the data.
        It is optional but recommended giving a long_name to each ndarray.
    meta : dict-like object, optional.
        Additional metadata for this object. Must be dict-like but no
        further restriction is placed on meta.
    copy : bool, optional
        Perform a copy of the passed object. Default is False.
    fill_missing : bool
        Create a linear coordinate array where missing data are masked.

    See Also
    --------
    NDDataset : Main SpectroChemPy object: an array with masks, units and
    coordinates.
    Coord : Explicit coordinates.

    Examples
    --------
    >>> from spectrochempy import LinearCoord, Coord

    To create a linear coordinate, we need to specify an offset,
    an increment and
    the size of the data

    >>> c1 = LinearCoord(offset=2.0, increment=2.0, size=10)

    Alternatively, linear coordinates can be created using the
    ``linear`` keyword

    >>> c2 = Coord(linear=True, offset=2.0, increment=2.0, size=10)
    """

    _use_time = tr.Bool(False)
    _show_datapoints = tr.Bool(True)
    _zpd = tr.Integer

    def __init__(self, data=None, offset=0.0, increment=1.0, **kwargs):

        if data is not None and isinstance(data, Coord) and not data.linear:
            raise ValueError(
                "Only linear Coord (with attribute linear set to True, can be transformed into "
                "LinearCoord class"
            )

        super().__init__(data, **kwargs)

        # when data is present, we don't need an offset and increment, nor size,
        # we just do linear=True and these parameters are ignored

        if self._data is not None:
            self._linear = True

        elif not self.linear:
            # in case it was not already a linear array
            self.offset = offset
            self.increment = increment
            self._linear = True

    # ..........................................................................
    @property  # read only
    def linear(self):
        return self._linear

    # ..........................................................................
    def __dir__(self):
        # remove some methods with respect to the full NDArray
        # as they are not useful for Coord.

        return [
            "data",
            "labels",
            "units",
            "meta",
            "long_name",
            "name",
            "offset",
            "increment",
            "linear",
            "size",
            "roi",
            "show_datapoints",
        ]

    def set_laser_frequency(self, frequency=15798.26 * ur("cm^-1")):

        if not isinstance(frequency, Quantity):
            frequency = frequency * ur("cm^-1")

        frequency.ito("Hz")
        self.meta.laser_frequency = frequency

        if self._use_time:
            spacing = 1.0 / frequency
            spacing.ito("picoseconds")

            self.increment = spacing.m
            self.offset = 0
            self._units = ur.picoseconds
            self.long_name = "time"

        else:
            frequency.ito("cm^-1")
            spacing = 1.0 / frequency
            spacing.ito("mm")

            self.increment = spacing.m
            self.offset = -self.increment * self._zpd
            self._units = ur.mm
            self.long_name = "optical path difference"

    @property
    def _use_time_axis(self):
        # private property
        # True if timescale must be used for interferogram axis. Else it
        # will be set to optical path difference.
        return self._use_time

    @_use_time_axis.setter
    def _use_time_axis(self, val):

        self._use_time = val
        if "laser_frequency" in self.meta:
            self.set_laser_frequency(self.meta.laser_frequency)

    # ..........................................................................
    @property
    def dtype(self):
        """
        Data type (np.dtype).
        """
        if self.is_empty:
            return None
        return self.data.dtype

    @property
    def show_datapoints(self):
        """
        Bool : True if axis must discard values and show only datapoints.

        """
        if "laser_frequency" not in self.meta or self.units.dimensionality not in [
            "[time]",
            "[length]",
        ]:
            return False

        return self._show_datapoints

    @show_datapoints.setter
    def show_datapoints(self, val):

        self._show_datapoints = val

    @property
    def laser_frequency(self):
        """
        Laser frequency if needed (Quantity).
        """
        return self.meta.laser_frequency

    @laser_frequency.setter
    def laser_frequency(self, val):
        self.meta.aser_frequency = val


# ======================================================================================
# CoordSet
# ======================================================================================


class CoordSet(tr.HasTraits):
    """
    A collection of Coord objects for a NDArray object with validation.

    This object is an iterable containing a collection of Coord objects.

    Parameters
    ----------
    *coords : |NDarray|, |NDArray| subclass or |CoordSet| sequence of objects.
        If an instance of CoordSet is found, instead of an array, this means
        that all coordinates in this coords describe the same axis.
        It is assumed that the coordinates are passed in the order of the
        dimensions of a nD numpy array (
        `row-major <https://docs.scipy.org/doc/numpy-1.14.1/glossary.html#term-row-major>`_
        order), i.e., for a 3d object : 'z', 'y', 'x'.
    **kwargs
        Additional keyword parameters (see Other Parameters).

    Other Parameters
    ----------------
    x : |NDarray|, |NDArray| subclass or |CoordSet|
        A single coordinate associated to the 'x'-dimension.
        If a coord was already passed in the argument, this will overwrite
        the previous. It is thus not recommended to simultaneously use
        both way to initialize the coordinates to avoid such conflicts.
    y, z, u, ... : |NDarray|, |NDArray| subclass or |CoordSet|
        Same as `x` for the others dimensions.
    dims : list of string, optional
        Names of the dims to use corresponding to the coordinates. If not
        given, standard names are used: x, y, ...
    copy : bool, optional
        Perform a copy of the passed object. Default is True.

    See Also
    --------
    Coord : Explicit coordinates object.
    LinearCoord : Implicit coordinates object.
    NDDataset: The main object of SpectroChempy which makes use of CoordSet.

    Examples
    --------
    >>> from spectrochempy import Coord, CoordSet

    Define 4 coordinates, with two for the same dimension

    >>> coord0 = Coord.linspace(10., 100., 5, units='m', long_name='distance')
    >>> coord1 = Coord.linspace(20., 25., 4, units='K', long_name='temperature')
    >>> coord1b = Coord.linspace(1., 10., 4, units='millitesla', long_name='magnetic field')
    >>> coord2 = Coord.linspace(0., 1000., 6, units='hour', long_name='elapsed time')

    Now create a coordset

    >>> cs = CoordSet(t=coord0, u=coord2, v=[coord1, coord1b])

    Display some coordinates

    >>> cs.u
    Coord: [float64] hr (size: 6)

    >>> cs.v
    CoordSet: [_1:temperature, _2:magnetic field]

    >>> cs.v_1
    Coord: [float64] K (size: 4)
    """

    # Hidden attributes containing the collection of objects
    _coords = tr.List(allow_none=True)
    _references = tr.Dict()
    _updated = tr.Bool(False)

    # Hidden id and name of the object
    _id = tr.Unicode()
    _name = tr.Unicode()

    # Hidden attribute to specify if the collection is for a single dimension
    _is_same_dim = tr.Bool(False)

    # other settings
    _copy = tr.Bool(False)
    _sorted = tr.Bool(True)
    _html_output = tr.Bool(False)

    # default coord index
    _default = tr.Int(0)

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------
    # ..........................................................................
    def __init__(self, *coords, **kwargs):

        self._copy = kwargs.pop("copy", True)
        self._sorted = kwargs.pop("sorted", True)

        keepnames = kwargs.pop("keepnames", False)
        # if keepnames is false and the names of the dimensions are not passed in kwargs, then use dims if not none
        dims = kwargs.pop("dims", None)

        self.name = kwargs.pop("name", None)

        # initialise the coordinate list
        self._coords = []

        # First evaluate passed args
        # --------------------------

        # some cleaning
        if coords:

            if all(
                [
                    (
                        isinstance(coords[i], (np.ndarray, NDArray, list, CoordSet))
                        or coords[i] is None
                    )
                    for i in range(len(coords))
                ]
            ):
                # Any instance of a NDArray can be accepted as coordinates for a dimension.
                # If an instance of CoordSet is found, this means that all
                # coordinates in this set describe the same axis
                coords = tuple(coords)

            elif is_sequence(coords) and len(coords) == 1:
                # if isinstance(coords[0], list):
                #     coords = (CoordSet(*coords[0], sorted=False),)
                # else:
                coords = coords[0]

                if isinstance(coords, dict):
                    # we have passed a dict, postpone to the kwargs evaluation process
                    kwargs.update(coords)
                    coords = None

            else:
                raise InvalidCoordinatesTypeError(
                    "Probably invalid coordinate have been passed."
                )

        # now store the args coordinates in self._coords (validation is fired when this attribute is set)
        if coords:
            for coord in coords[::-1]:  # we fill from the end of the list
                # (in reverse order) because by convention when the
                # names are not specified, the order of the
                # coords follow the order of dims.
                if not isinstance(coord, CoordSet):
                    if isinstance(coord, list):
                        coord = CoordSet(*coord[::-1], sorted=False)
                    elif not isinstance(coord, LinearCoord):  # else
                        coord = Coord(coord, copy=True)
                else:
                    coord = cpy.deepcopy(coord)

                if not keepnames:
                    if dims is None:
                        # take the last available name of available names list
                        coord.name = self.available_names.pop(-1)
                    else:
                        # use the provided list of dims
                        dim = list(dims).pop(-1)
                        if dim in self.available_names:
                            coord.name = dims.pop(-1)
                        else:
                            raise InvalidDimensionNameError(dim, self.available_names)

                self._append(coord)  # append the coord
                # (but instead of append, use assignation -in _append - to fire the validation process )

        # now evaluate keywords argument
        # ------------------------------

        for key, coord in list(kwargs.items())[:]:
            # remove the already used kwargs (Fix: deprecation warning in Traitlets - all args, kwargs must be used)
            del kwargs[key]

            # prepare values to be either Coord, LinearCoord or CoordSet
            if isinstance(coord, (list, tuple)):
                coord = CoordSet(
                    *coord, sorted=False
                )  # make sure in this case it becomes a CoordSet instance

            elif isinstance(coord, np.ndarray) or coord is None:
                coord = Coord(
                    coord, copy=True
                )  # make sure it's a Coord  # (even if it is None -> Coord(None)

            elif isinstance(coord, str) and coord in DEFAULT_DIM_NAME:
                # may be a reference to another coordinates (e.g. same coordinates for various dimensions)
                self._references[key] = coord  # store this reference
                continue

            # Populate the coords with coord and coord's name.
            if isinstance(coord, (NDArray, Coord, LinearCoord, CoordSet)):  # NDArray,
                if key in self.available_names or (
                    len(key) == 2
                    and key.startswith("_")
                    and key[1] in list("123456789")
                ):
                    # ok we can find it as a canonical name:
                    # this will overwrite any already defined coord value
                    # which means also that kwargs have priority over args
                    coord.name = key
                    self._append(coord)

                elif not self.is_empty and key in self.names:
                    # append when a coordinate with this name is already set in passed arg.
                    # replace it
                    idx = self.names.index(key)
                    coord.name = key
                    self._coords[idx] = coord

                else:
                    raise InvalidDimensionNameError(key, DEFAULT_DIM_NAME)

            else:
                raise InvalidCoordinatesTypeError(
                    f"Probably an invalid type of coordinates has been passed: {key}:{coord} "
                )

        # store the item (validation will be performed)
        # self._coords = _coords

        # inform the parent about the update
        self._updated = True

        # set a notifier on the name traits name of each coordinates
        if self._coords is not None:
            for coord in self._coords:
                if coord is not None:
                    tr.HasTraits.observe(coord, self._coords_update, "_name")

        # initialize the base class with the eventual remaining arguments
        super().__init__(**kwargs)

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    def __call__(self, *args, **kwargs):
        # allow the following syntax: coords(), coords(0,2) or
        coords = []
        axis = kwargs.get("axis", None)
        if args:
            for idx in args:
                coords.append(self[idx])
        elif axis is not None:
            if not is_sequence(axis):
                axis = [axis]
            for i in axis:
                coords.append(self[i])
        else:
            coords = self._coords
        if len(coords) == 1:
            return coords[0]
        else:
            return CoordSet(*coords)

    # ..........................................................................
    def __copy__(self):
        coords = self.__class__(tuple(cpy.copy(ax) for ax in self), keepnames=True)
        # name must be changed
        coords.name = self.name
        # and is_same_dim and default for coordset
        coords._is_same_dim = self._is_same_dim
        coords._default = self._default
        return coords

    # ..........................................................................
    def __deepcopy__(self, memo):
        coords = self.__class__(
            tuple(cpy.deepcopy(ax, memo=memo) for ax in self), keepnames=True
        )
        coords.name = self.name
        coords._is_same_dim = self._is_same_dim
        coords._default = self._default
        return coords

    # ..........................................................................
    def __delattr__(self, item):
        if "notify_change" in item:
            pass

        else:
            try:
                return self.__delitem__(item)
            except (IndexError, KeyError):
                raise AttributeError

    # ..........................................................................
    def __delitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                del self._coords[idx]
                return

            # let's try in the long_name
            if index in self.long_names:
                # selection by coord long_names
                index = self.long_names.index(index)
                self._coords.__delitem__(index)
                return

            # maybe it is a long_name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.long_names:
                    # selection by subcoord long_name
                    return item.__delitem__(index)

            # let try with the canonical dimension names
            if index[0] in self.names:
                # ok we can find it a canonical name:
                c = self._coords.__getitem__(self.names.index(index[0]))
                if len(index) > 1 and index[1] == "_":
                    if isinstance(c, CoordSet):
                        return c.__delitem__(index[1:])

            raise KeyError(
                f"Could not find `{index}` in coordinates names or long_names"
            )

    # ..........................................................................
    def __dir__(self):
        return ["coords", "references", "is_same_dim", "name"]

    # ..........................................................................
    def __eq__(self, other):
        if other is None:
            return False
        try:
            return self._coords == other._coords
        except Exception:
            return False

    # ..........................................................................
    def __getattr__(self, item):
        # when the attribute was not found
        if "_validate" in item or "_changed" in item:
            raise AttributeError

        try:
            return self.__getitem__(item)
        except (IndexError, KeyError):
            raise AttributeError

    # ..........................................................................
    def __getitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                return self._coords.__getitem__(idx)

            # ok we did not find it!
            # let's try in references
            if index in self._references.keys():
                return self._references[index]

            # let's try in the long_name
            if index in self.long_names:
                # selection by coord long_names
                if self.long_names.count(index) > 1:
                    warnings.warn(
                        f"Getting a coordinate from its long_name. However `{index}` occurs several time. Only"
                        f" the first occurrence is returned!"
                    )
                return self._coords.__getitem__(self.long_names.index(index))

            # maybe it is a long_name or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.long_names:
                    # selection by subcoord long_name
                    return item.__getitem__(item.long_names.index(index))

            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.names:
                    # selection by subcoord name
                    return item.__getitem__(item.names.index(index))

            try:
                # let try with the canonical dimension names
                if index[0] in self.names:
                    # ok we can find it a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == "_":
                        if isinstance(c, CoordSet):
                            c = c.__getitem__(index[1:])
                        else:
                            c = c.__getitem__(index[2:])  # try on labels
                    return c
            except IndexError:
                pass

            raise AttributeError(
                f"Could not find `{index}` key/attribute in coordinates names or long_names"
            )

        try:
            res = self._coords.__getitem__(index)
        except TypeError:
            if self.is_same_dim:
                # in this case we try to slice a coord!
                # let's take the default
                return self.default.__getitem__(index)
            raise TypeError

        if isinstance(index, slice):
            if isinstance(res, CoordSet):
                res = (res,)
            return CoordSet(*res, keepnames=True)
        else:
            return res

    # ..........................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash(tuple(self._coords))

    # ..........................................................................
    def __len__(self):
        return len(self._coords)

    # ..........................................................................
    def __ne__(self, other):
        return not self.__eq__(other)

    # ..........................................................................
    def __repr__(self):
        out = "CoordSet: [" + ", ".join(["{}"] * len(self._coords)) + "]"
        s = []
        for item in self._coords:
            if isinstance(item, CoordSet):
                s.append(f"{item.name}:" + repr(item).replace("CoordSet: ", ""))
            else:
                s.append(f"{item.name}:{item.long_name}")
        out = out.format(*s)
        return out

    # ..........................................................................
    def __setattr__(self, key, value):
        keyb = key[1:] if key.startswith("_") else key
        if keyb in [
            "parent",
            "copy",
            "sorted",
            "coords",
            "updated",
            "name",
            "html_output",
            "is_same_dim",
            "parent_dim",
            "trait_values",
            "trait_notifiers",
            "trait_validators",
            "cross_validation_lock",
            "notify_change",
        ]:
            super().__setattr__(key, value)
            return

        try:
            self.__setitem__(key, value)
        except Exception:
            super().__setattr__(key, value)

    # ..........................................................................
    def __setitem__(self, index, coord):
        try:
            coord = coord.copy(keepname=True)  # to avoid modifying the original
        except TypeError as e:
            if isinstance(coord, list):
                coord = [c.copy(keepname=True) for c in coord[:]]
            else:
                raise e

        if isinstance(index, str):
            # find by name
            if index in self.names:
                idx = self.names.index(index)
                coord.name = index
                self._coords.__setitem__(idx, coord)
                return

            # ok we did not find it!
            # let's try in the long_name
            if index in self.long_names:
                # selection by coord long_names
                if self.long_names.count(index) > 1:
                    warnings.warn(
                        f"Getting a coordinate from its long_name. However `{index}` occurs several time. Only"
                        f" the first occurrence is returned!",
                        SpectroChemPyWarning,
                    )
                index = self.long_names.index(index)
                coord.name = self.names[index]
                self._coords.__setitem__(index, coord)
                return

            # maybe it is a long_name or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.long_names:
                    # selection by subcoord long_name
                    index = item.long_names.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.names:
                    # selection by subcoord long_name
                    index = item.names.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return

            try:
                # let try with the canonical dimension names
                if index[0] in self.names and (
                    len(index) == 1 or (len(index) == 3 and "_" in index)
                ):
                    # ok we can find it a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == "_":
                        c.__setitem__(index[1:], coord)
                    return

            except KeyError:
                pass

            # add the new coordinates
            if index in self.available_names or (
                len(index) == 2
                and index.startswith("_")
                and index[1] in list("123456789")
            ):
                coord.name = index
                self._coords.append(coord)
                return

            else:
                raise KeyError(
                    f"Could not find `{index}` in coordinates names or long_names"
                )

        self._coords[index] = coord

    # ..........................................................................
    def __str__(self):
        return repr(self)

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    @tr.observe(tr.All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The tr.HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }

        if change.name == "_updated" and change.new:
            self._updated = False  # reset

    # ..........................................................................
    def _append(self, coord):
        # utility function to append coordinate with full validation
        if not isinstance(coord, tuple):
            coord = (coord,)
        if self._coords:
            # some coordinates already present, prepend the new one
            self._coords = (*coord,) + tuple(
                self._coords
            )  # instead of append, fire the validation process
        else:
            # no coordinates yet, start a new tuple of coordinate
            self._coords = (*coord,)

    # ..........................................................................
    def _coords_update(self, change):
        # when notified that a coord name have been updated
        self._updated = True

    # ..........................................................................
    @tr.validate("_coords")
    def _coords_validate(self, proposal):
        coords = proposal["value"]
        if not coords:
            return []

        for idx, coord in enumerate(coords):
            if coord and not isinstance(coord, (Coord, LinearCoord, CoordSet)):
                raise InvalidCoordinatesTypeError(
                    "At this point all passed coordinates should be of type Coord or CoordSet!"
                )  #
            if self._copy:
                coords[idx] = coord.copy()
            else:
                coords[idx] = coord

        for coord in coords:
            if isinstance(coord, CoordSet):
                # it must be a single dimension axis
                # in this case we must have same length for all coordinates
                coord._is_same_dim = True

                # check this is valid in terms of size
                try:
                    _ = coord.sizes
                except InvalidCoordinatesSizeError:
                    raise

                # change the internal names
                n = len(coord)
                coord._set_names(
                    [f"_{i + 1}" for i in range(n)]
                )  # we must have  _1 for the first coordinates,
                # _2 the second, etc...
                coord._set_parent_dim(coord.name)

        # last check and sorting
        names = []
        for coord in coords:
            if coord.has_defined_name:
                names.append(coord.name)
            else:
                raise ValueError(
                    "At this point all passed coordinates should have a valid name!"
                )

        if coords:
            if self._sorted:
                _sortedtuples = sorted(
                    (coord.name, coord) for coord in coords
                )  # Final sort
                coords = list(zip(*_sortedtuples))[1]
            return list(coords)  # be sure it is a list not a tuple
        else:
            return None

    # ..........................................................................
    def _cstr(self, header="  coordinates: ... \n", print_size=True):

        txt = ""
        for idx, dim in enumerate(self.names):
            coord = getattr(self, dim)

            if coord:

                dimension = f"     DIMENSION `{dim}`"
                for k, v in self.references.items():
                    if dim == v:
                        # reference to this dimension
                        dimension += f"=`{k}`"
                txt += dimension + "\n"

                if isinstance(coord, CoordSet):
                    # txt += '        index: {}\n'.format(idx)
                    if not coord.is_empty:
                        if print_size:
                            txt += f"{coord[0]._str_shape().rstrip()}\n"

                        coord._html_output = self._html_output
                        for idx_s, dim_s in enumerate(coord.names):
                            c = getattr(coord, dim_s)
                            txt += f"          ({dim_s}) ...\n"
                            c._html_output = self._html_output
                            sub = c._cstr(
                                header="  coordinates: ... \n", print_size=False
                            )  # , indent=4, first_indent=-6)
                            txt += f"{sub}\n"

                elif not coord.is_empty:
                    # coordinates if available
                    # txt += '        index: {}\n'.format(idx)
                    coord._html_output = self._html_output
                    txt += "{}\n".format(
                        coord._cstr(header=header, print_size=print_size)
                    )

        txt = txt.rstrip()  # remove the trailing '\n'

        if not self._html_output:
            return colored_output(txt.rstrip())
        else:
            return txt.rstrip()

    # ..........................................................................
    @tr.default("_id")
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    # ..........................................................................
    def _loc2index(self, loc):
        # Return the index of a location
        # it search in the default coordinate for same_dim coordset.
        if not self.is_same_dim:
            raise TypeError("The current object is not a coordinate")
        else:
            return self.default.loc2index(loc)

    # ..........................................................................
    def _repr_html_(self):
        return convert_to_html(self)

    # ..........................................................................
    def _set_names(self, names):
        # utility function to change names of coordinates (in batch)
        # useful when a coordinate is a CoordSet itself
        for coord, name in zip(self._coords, names):
            coord.name = name

    # ..........................................................................
    def _set_parent_dim(self, name):
        # utility function to set the parent name for sub coordset
        for coord in self._coords:
            coord._parent_dim = name

    # ------------------------------------------------------------------------
    # Readonly Properties
    # ------------------------------------------------------------------------

    # ..........................................................................
    @property
    def available_names(self):
        """
        Chars that can be used for dimension name (list).

        It returns DEFAULT_DIM_NAMES less those already in use.
        """
        _available_names = DEFAULT_DIM_NAME.copy()
        for item in self.names:
            if item in _available_names:
                _available_names.remove(item)
        return _available_names

    # ..........................................................................
    @property
    def coords(self):
        """
        Coordinates in the coordset (list).
        """
        return self._coords

    # ..........................................................................
    @property
    def has_defined_name(self):
        """
        True if the name has been defined (bool).
        """
        return not (self.name == self.id)

    # ..........................................................................
    @property
    def id(self):
        """
        Object identifier (Readonly property).
        """
        return self._id

    # ..........................................................................
    @property
    def is_empty(self):
        """
        True if there is no coords defined (bool).
        """
        if self._coords:
            return len(self._coords) == 0
        else:
            return True

    # ..........................................................................
    @property
    def is_same_dim(self):
        """
        True if the coords define a single dimension (bool).
        """
        return self._is_same_dim

    # ..........................................................................
    @property
    def references(self):
        """
        return a dictionary returning the references to other dimensions.

        Examples
        --------

        >>> coord0 = scp.LinearCoord.arange(10)
        >>> c = scp.CoordSet(x=coord0, y="x")
        >>> assert c.references == dict(y="x")

        """
        return self._references

    # ..........................................................................
    @property
    def sizes(self):
        """
        Sizes of the coord object for each dimension (int or tuple of int).

        (readonly property). If the set is for a single dimension return a
        single size as all coordinates must have the same.
        """
        _sizes = []
        for i, item in enumerate(self._coords):
            _sizes.append(item.size)  # recurrence if item is a CoordSet

        if self.is_same_dim:
            _sizes = list(set(_sizes))
            if len(_sizes) > 1:
                raise InvalidCoordinatesSizeError(
                    "Coordinates must be of the same size for a dimension with multiple coordinates"
                )
            return _sizes[0]
        return _sizes

    # alias
    size = sizes

    # ..........................................................................
    @property
    def names(self):
        """
        Names of the coords in the current coords (list - read only property).
        """
        _names = []
        if self._coords:
            for item in self._coords:
                if item.has_defined_name:
                    _names.append(item.name)
        return _names

    @property
    def default(self):
        """
        Default coordinates (Coord).
        """
        if hasattr(self, "is_same_dim") and self.is_same_dim:
            # valid only for multicoordinate CoordSet
            default = self[self._default]
            return default
        # warning_('There is no multicoordinate in this CoordSet. Return None')

    # ------------------------------------------------------------------------
    # Mutable Properties
    # ------------------------------------------------------------------------

    @property
    def data(self):
        # in case data is called on a coordset for dimension with multiple coordinates
        # return the first coordinates
        if hasattr(self, "is_same_dim") and self.is_same_dim:
            return self.default.data
        warning_("There is no multicoordinate in this CoordSet. Return None")

    # ..........................................................................
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self._id

    @name.setter
    def name(self, value):
        if value is not None:
            self._name = value

    # ..........................................................................
    @property
    @deprecated(type="property", replace="long_names")
    def titles(self):
        """
        Titles of the coords in the current coords (list).
        """
        return self.long_names

    @property
    def long_names(self):
        """
        Titles of the coords in the current coords (list).
        """
        _long_names = []
        for item in self._coords:
            if isinstance(item, NDArray):
                _long_names.append(
                    item.long_name if item.long_name else item.name
                )  # TODO:name
            elif isinstance(item, CoordSet):
                _long_names.append(
                    [el.long_name if el.long_name else el.name for el in item]
                )  # TODO:name
            else:  # pragma: no cover
                raise ValueError("Something is wrong with the long_names!")
        return _long_names

    # ..........................................................................
    @property
    def labels(self):
        """
        Labels of the coordinates in the current coordset (list).
        """
        return [item.labels for item in self]

    # ..........................................................................
    @property
    def units(self):
        """
        Units of the coords in the current coords (list of string).
        """
        units = []
        for item in self:
            if isinstance(item, CoordSet):
                units.append(item.units)
            else:
                units.append(str(item.units))
        return units

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

    @staticmethod
    # ..........................................................................
    def implements(name=None):
        """
        Utility to check if the current object implement `CoordSet`.

        Rather than isinstance(obj, CoordSet) use object.implements('CoordSet').

        This is useful to check type without importing the module.
        """
        if name is None:
            return "CoordSet"
        else:
            return name == "CoordSet"

    # ..........................................................................
    def copy(self, keepname=False):
        """
        Make a disconnected copy of the current coords.

        Returns
        -------
        object
            an exact copy of the current object
        """
        return self.__copy__()

    # ..........................................................................
    def keys(self):
        """
        Similar to the names property but includes references to other coordinates.

        Returns
        -------
        out
            List of all coordinates names (including reference to other coordinates).
        """
        keys = []
        if self.names:
            keys.extend(self.names)
        if self._references:
            keys.extend(list(self.references.keys()))
        return keys

    def loc2index(self, loc):
        """
        Return the index corresponding to a given location .

        Parameters
        ----------
        loc : float.
            Value corresponding to a given location on the coordinate's axis.

        Returns
        -------
        index : int.
            The corresponding index.

        Examples
        --------

        >>> dataset = scp.NDDataset.read("irdata/nh4y-activation.spg")
        >>> dataset.x.loc2index(1644.0)
        4517
        """
        return self._loc2index(loc)

    # ..........................................................................
    def select(self, val):
        """
        Select the default coord index.
        """
        self._default = min(max(0, int(val) - 1), len(self.names))

    # ..........................................................................
    def set(self, *args, **kwargs):
        """
        Set one or more coordinates in the current CoordSet.

        Parameters
        ----------
        *args

        **kwargs

        Returns
        -------
        """
        if not args and not kwargs:
            return

        mode = kwargs.pop("mode", "a")

        if mode == "w":
            self._coords = []

        if len(args) == 1 and (
            (is_sequence(args[0]) and not isinstance(args[0], Coord))
            or isinstance(args[0], CoordSet)
        ):
            args = args[0]

        if isinstance(args, CoordSet):
            kwargs.update(args.to_dict())
            args = ()

        for i, item in enumerate(args[::-1]):
            item.name = self.available_names.pop()
            self._append(item)

        for k, item in kwargs.items():
            if isinstance(item, CoordSet):
                # try to keep this parameter to True!
                item._is_same_dim = True
            self[k] = item

    @deprecated(replace="set_long_names")
    def set_titles(self, *args, **kwargs):
        """
        DEPRECATED: use set_long_names instead
        """
        self.set_long_names(*args, **kwargs)

    # ..........................................................................
    def set_long_names(self, *args, **kwargs):
        """
        Set one or more coord long_name at once.

        Parameters
        ----------
        args : str(s)
            The list of long_names to apply to the set of coordinates. They must be given according to the coordinate's name
            alphabetical order.
        **kwargs
            Keyword attribution of the long_names. The keys must be valid names among the coordinate's name list. This
            is the recommended way to set long_names as this will be less prone to errors.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's  name alphabetical order :
        e.g, the first long_name will be for the `x` coordinates, the second for the `y`, etc.
        """
        if len(args) == 1 and (is_sequence(args[0]) or isinstance(args[0], CoordSet)):
            args = args[0]

        for i, item in enumerate(args):
            if not isinstance(self[i], CoordSet):
                self[i].long_name = item
            else:
                if is_sequence(item):
                    for j, v in enumerate(self[i]):
                        v.long_name = item[j]

        for k, item in kwargs.items():
            self[k].long_name = item

    # ..........................................................................
    def set_units(self, *args, **kwargs):
        """
        Set one or more coord units at once.

        Parameters
        ----------
        *args : str(s)
            The list of units to apply to the set of coordinates. They must be given according to the coordinate's name
            alphabetical order.
        **kwargs
            Keyword attribution of the units. The keys must be valid names among the coordinate's name list. This
            is the recommended way to set units as this will be less prone to errors.
            Also see Other Parameters.

        Other Parameters
        ----------------
        force : bool, optional, default=False
            Whether the new units must be compatible with the current units. See the `Coord`.`to` method.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's name alphabetical order :
        e.g, the first units will be for the `x` coordinates, the second for the `y`, etc.
        """
        force = kwargs.pop("force", False)

        if len(args) == 1 and is_sequence(args[0]):
            args = args[0]

        for i, item in enumerate(args):
            if not isinstance(self[i], CoordSet):
                self[i].to(item, force=force, inplace=True)
            else:
                # if is_sequence(item):
                #     for j, v in enumerate(self[i]):
                #         v.to(item[j], force=force, inplace=True)
                self[i].set_units(item, force=force)
        for k, item in kwargs.items():
            self[k].to(item, force=force, inplace=True)

    # ..........................................................................
    def to_dict(self):
        """
        Return a dict of the coordinates from the coordset.

        Returns
        -------
        out : dict
            A dictionary where keys are the names of the coordinates, and the values the coordinates themselves.
        """
        return dict(zip(self.names, self._coords))

    # ..........................................................................
    def update(self, **kwargs):
        """
        Update a specific coordinates in the CoordSet.

        Parameters
        ----------
        **kwarg
            Only keywords among the `CoordSet.names` are allowed - they denote the name of a dimension.
        """
        dims = kwargs.keys()
        for dim in list(dims)[:]:
            if dim in self.names:
                # we can replace the given coordinates
                idx = self.names.index(dim)
                self[idx] = Coord(kwargs.pop(dim), name=dim)
            elif len(dim) > 1 and dim[1] == "_":  # Probably in a subcoordset
                idx = self.names.index(dim[0])
                subdim = dim[1:]
                self[idx].update(**{subdim: kwargs.pop(dim)})
            else:
                raise InvalidDimensionNameError(dim)


# ======================================================================================
# _CoordRange
# ======================================================================================


class _CoordRange(tr.HasTraits):
    # TODO: May use also units ???
    ranges = tr.List(Range())
    reversed = tr.Bool()

    # ..........................................................................
    def __init__(self, *ranges, reversed=False, **kwargs):

        super().__init__(**kwargs)

        self.reversed = reversed
        if len(ranges) == 0:
            # first case: no argument passed, returns an empty range
            self.ranges = []
        elif len(ranges) == 2 and all(isinstance(elt, (int, float)) for elt in ranges):
            # second case: a pair of scalars has been passed
            # using the Interval class, we have autochecking of the interval
            # validity
            self.ranges = [list(map(float, ranges))]
        else:
            # third case: a set of pairs of scalars has been passed
            self._clean_ranges(ranges)
        if self.ranges:
            self._clean_ranges(self.ranges)

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------
    # ..........................................................................
    def _clean_ranges(self, ranges):
        """Sort and merge overlapping ranges
        It works as follows::
        1. orders each interval
        2. sorts intervals
        3. merge overlapping intervals
        4. reverse the orders if required
        """
        # transforms each pairs into valid interval
        # should generate an error if a pair is not valid
        ranges = [list(range) for range in ranges]
        # order the ranges
        ranges = sorted(ranges, key=lambda r: min(r[0], r[1]))
        cleaned_ranges = [ranges[0]]
        for range in ranges[1:]:
            if range[0] <= cleaned_ranges[-1][1]:
                if range[1] >= cleaned_ranges[-1][1]:
                    cleaned_ranges[-1][1] = range[1]
            else:
                cleaned_ranges.append(range)
        self.ranges = cleaned_ranges
        if self.reversed:
            for range in self.ranges:
                range.reverse()
            self.ranges.reverse()


def trim_ranges(*ranges, reversed=False):
    """
    Set of ordered, non-intersecting intervals.

    An ordered set of ranges is constructed from the inputs and returned.
    *e.g.,* [[a, b], [c, d]] with a < b < c < d or a > b > c > d.

    Parameters
    -----------
    *ranges :  iterable
        An interval or a set of intervals.
        set of  intervals. If none is given, the range will be a set of an empty interval [[]]. The interval
        limits do not need to be ordered, and the intervals do not need to be distincts.
    reversed : bool, optional
        The intervals are ranked by decreasing order if True or increasing order if False.

    Returns
    -------
    ordered
        list of ranges.

    Examples
    --------

    >>> scp.trim_ranges([1, 4], [7, 5], [6, 10])
    [[1, 4], [5, 10]]
    """
    return _CoordRange(*ranges, reversed=reversed).ranges


# ======================================================================================
# Set the operators
# ======================================================================================
_set_operators(Coord, priority=50)
_set_operators(LinearCoord, priority=50)

# ======================================================================================
if __name__ == "__main__":
    pass
