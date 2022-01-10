# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implements the class |Coord|.
"""

__all__ = ["Coord", "LinearCoord"]

import copy as cpy
import textwrap

import numpy as np
from traitlets import (
    Bool,
    observe,
    All,
    Unicode,
    Integer,
    Union,
    CFloat,
    CInt,
    Instance,
)
from traitlets import default as traitdefault

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndmath import NDMath, _set_operators
from spectrochempy.utils import (
    colored_output,
    NOMASK,
    INPLACE,
    spacing_,
)
from spectrochempy.units import Quantity, ur
from spectrochempy.core import error_


# ======================================================================================================================
# Coord
# ======================================================================================================================
class Coord(NDMath, NDArray):
    """
    Explicit coordinates for a dataset along a given axis.

    The coordinates of a |NDDataset| can be created using the |Coord|
    object.
    This is a single dimension array with either numerical (float)
    values or labels (str, `Datetime` objects, or any other kind of objects) to
    represent the coordinates. Only a one numerical axis can be defined,
    but labels can be multiple.
    """

    _copy = Bool()

    _html_output = False
    _parent_dim = Unicode(allow_none=True)

    # For linear data generation
    _offset = Union((CFloat(), CInt(), Instance(Quantity)))
    _increment = Union((CFloat(), CInt(), Instance(Quantity)))
    _size = Integer(0)
    _linear = Bool(False)

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):
        """
        Parameters
        -----------
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
            If specified, the data will be casted to this dtype, else the
            type of the data will be used.
        dims : list of chars, optional.
            if specified the list must have a length equal to the number od
            data dimensions (ndim) and the chars must be
            taken among among x,y,z,u,v,w or t. If not specified,
            the dimension names are automatically attributed in
            this order.
        name : str, optional
            A user friendly name for this object. If not given,
            the automatic `id` given at the object creation will be
            used as a name..
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
        title : str, optional
            The title of the dimension. It will later be used for instance
            for labelling plots of the data.
            It is optional but recommended to give a title to each ndarray.
        dlabel :  str, optional
            Alias of `title`.
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
            If omitted a value of 0.0 is taken for tje coordinate offset.
        increment : float, optional
            Only used if linear is true.
            If omitted a value of 1.0 is taken for the coordinate increment.

        See Also
        --------
        NDDataset : Main SpectroChemPy object: an array with masks, units and coordinates.
        LinearCoord : linear coordinates.

        Examples
        --------

        We first import the object from the api :
        >>> from spectrochempy import Coord

        We then create a numpy |ndarray| and use it as the numerical `data`
        axis of our new |Coord| object.
        >>> c0 = Coord.arange(1., 12., 2., title='frequency', units='Hz')
        >>> c0
        Coord: [float64] Hz (size: 6)

        We can take a series of str to create a non numerical but labelled
        axis :
        >>> tarr = list('abcdef')
        >>> tarr
        ['a', 'b', 'c', 'd', 'e', 'f']

        >>> c1 = Coord(labels=tarr, title='mylabels')
        >>> c1
        Coord: [labels] [  a   b   c   d   e   f] (size: 6)
        """

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
        if self.units in ["1 / centimeter", "ppm"]:
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

        return data

    @data.setter
    def data(self, data):

        self._set_data(data)

    @property
    def default(self):
        # this is in case default is called on a coord, while it is a coordset property
        return self

    # ------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docstring)
    # some of the property of NDArray has to be hidden because they
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

        new = super().to(other, force=force)

        if inplace:
            self._units = new._units
            self._title = new._title
            self._roi = new._roi
            if not self.linear:
                self._data = new._data
            else:
                self._offset = new._offset
                self._increment = new._increment
                self._linear = new._linear

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
            val  # it val is true this provoque the linearization (  # see observe)
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
        else an array of the differents spacings
        """
        if self.linear:
            return self.increment * self.units
        return spacing_(self.data) * self.units

    # ..........................................................................
    def squeeze(self, *args, **kwargs):
        raise NotImplementedError

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
        loc: float.
            Value corresponding to a given location on the coordinates axis.

        Returns
        -------
        index: int.
            The corresponding index.

        Examples
        --------

        >>> dataset = scp.NDDataset.read("irdata/nh4y-activation.spg")
        >>> dataset.x.loc2index(1644.0)
        4517
        """
        return self._loc2index(loc)

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
            "title",
            "name",
            "offset",
            "increment",
            "linear",
            "roi",
        ]

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
        keys = self._make_index(items)

        # init returned object
        if inplace:
            new = self
        else:
            new = self.copy()

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
        out += f"        title: {self.title}\n" if self.title else ""
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
    @traitdefault("_increment")
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
    @traitdefault("_offset")
    def _offset_default(self):
        return 0

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
                self.history = f"Copied from object:{data.name}"
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
                self.history = f"Copied from object:{data.name}"
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
        ):
            # Data doesn't look like a numpy array, try converting it to
            # one. Non-numerical input are converted to an array of objects.
            self._data = np.array(data, subok=True, copy=False)

        else:
            data = np.array(data, subok=True, copy=self._copy)
            if data.dtype == np.object_:  # likely None value
                data = data.astype(float)
            self._data = data

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

        # if new._linear:
        #     # try to make it linear as well
        #     new._linearize()
        #     if not new._linear and new.implements("LinearCoord"):
        #         # can't be linearized -> Coord
        #         if inplace:
        #             raise Exception(
        #                     "A LinearCoord object cannot be transformed to a non linear coordinate "
        #                     "`inplace`. "
        #                     "Use to() instead of ito() and leave the `inplace` attribute to False"
        #             )
        #         else:
        #             from spectrochempy import Coord
        #
        #             new = Coord(new)
        return new

    # ------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------
    # ..........................................................................
    @observe(All)
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


class LinearCoord(Coord):
    """
    Linear coordinates.

    Such coordinates correspond to a ascending or descending linear
    sequence of values, fully determined by two parameters, i.e., an offset (off) and an increment (inc) :

    .. math::

        \\mathrm{data} = i*\\mathrm{inc} + \\mathrm{off}.
    """

    _use_time = Bool(False)
    _show_datapoints = Bool(True)
    _zpd = Integer

    def __init__(self, data=None, offset=0.0, increment=1.0, **kwargs):

        """
        Parameters
        ----------
        data : a 1D array-like object, optional
            WWen provided, the `size` parameters is adjusted to the size of
            the array, and a linearization of the
            array is performed (only if it is possible: regular spacing in
            the 1.e5 relative accuracy).
        offset : float, optional
            If omitted a value of 0.0 is taken for tje coordinate offset.
        increment : float, optional
            If omitted a value of 1.0 is taken for the coordinate increment.
        **kwargs
            Optional keywords parameters. See other parameters.

        Other Parameters
        ----------------
        dtype : str or dtype, optional, default=np.float64
            If specified, the data will be casted to this dtype, else the
            type of the data will be used
        dims : list of chars, optional.
            if specified the list must have a length equal to the number od
            data dimensions (ndim) and the chars must be
            taken among among x,y,z,u,v,w or t. If not specified,
            the dimension names are automatically attributed in
            this order.
        name : str, optional
            A user friendly name for this object. If not given,
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
        title : str, optional
            The title of the dimension. It will later be used for instance
            for labelling plots of the data.
            It is optional but recommended to give a title to each ndarray.
        dlabel : str, optional.
            Alias of `title`.
        meta : dict-like object, optional.
            Additional metadata for this object. Must be dict-like but no
            further restriction is placed on meta.
        copy : bool, optional
            Perform a copy of the passed object. Default is False.
        fill_missing : bool
            Create a linear coordinate array where missing data are masked.

        See Also
        --------
        NDDataset : Main SpectroChemPy object: an array with masks, units and coordinates.
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

        if data is not None and isinstance(data, Coord) and not data.linear:
            raise ValueError(
                "Only linear Coord (with attribute linear set to True, can be transformed into "
                "LinearCoord class"
            )

        super().__init__(data, **kwargs)

        # when data is present, we don't need offset and increment, nor size,
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
        # as they are not usefull for Coord.

        return [
            "data",
            "labels",
            "units",
            "meta",
            "title",
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
            self.title = "time"

        else:
            frequency.ito("cm^-1")
            spacing = 1.0 / frequency
            spacing.ito("mm")

            self.increment = spacing.m
            self.offset = -self.increment * self._zpd
            self._units = ur.mm
            self.title = "optical path difference"

    @property
    def _use_time_axis(self):
        # private property
        # True if time scale must be used for interferogram axis. Else it
        # will be set to optical path difference.
        return self._use_time

    @_use_time_axis.setter
    def _use_time_axis(self, val):

        self._use_time = val
        if "laser_frequency" in self.meta:
            self.set_laser_frequency(self.meta.laser_frequency)

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


# ======================================================================================================================
# Set the operators
# ======================================================================================================================
_set_operators(Coord, priority=50)
_set_operators(LinearCoord, priority=50)

# ======================================================================================================================
if __name__ == "__main__":
    pass
