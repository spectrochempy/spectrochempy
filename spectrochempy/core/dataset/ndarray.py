# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
#  =
# ======================================================================================================================
"""
This module implements the base |NDArray| class.
"""

__all__ = ["NDArray", "DEFAULT_DIM_NAME"]

import copy as cpy
from datetime import datetime, timezone
import warnings
import re
import textwrap
import uuid
import itertools
import pathlib

from traitlets import (
    List,
    Unicode,
    Instance,
    Bool,
    Union,
    CFloat,
    Integer,
    CInt,
    HasTraits,
    default,
    validate,
    observe,
    All,
)
from pint.errors import DimensionalityError
import numpy as np
from traittypes import Array

from spectrochempy.units import Unit, ur, Quantity, set_nmr_context
from spectrochempy.core import info_, error_, print_
from spectrochempy.core.dataset.meta import Meta
from spectrochempy.utils import (
    TYPE_INTEGER,
    TYPE_FLOAT,
    MaskedConstant,
    MASKED,
    NOMASK,
    INPLACE,
    is_sequence,
    is_number,
    numpyprintoptions,
    spacing,
    insert_masked_print,
    SpectroChemPyWarning,
    make_new_object,
    convert_to_html,
    get_user_and_node,
    pathclean,
)

# ======================================================================================================================
# constants
# ======================================================================================================================

DEFAULT_DIM_NAME = list("xyzuvwpqrstijklmnoabcdefgh")[::-1]

# ======================================================================================================================
# Printing settings
# ======================================================================================================================

numpyprintoptions()


# ======================================================================================================================
# The basic NDArray class
# ======================================================================================================================


# noinspection PyPep8Naming
class NDArray(HasTraits):
    # hidden properties

    # array definition
    _id = Unicode()
    _name = Unicode()
    _title = Unicode(allow_none=True)
    _data = Array(allow_none=True)
    _dtype = Instance(np.dtype, allow_none=True)
    _dims = List(Unicode())
    _mask = Union((Bool(), Array(Bool()), Instance(MaskedConstant)))
    _labels = Array(allow_none=True)
    _units = Instance(Unit, allow_none=True)

    # region of interest
    _roi = List(allow_none=True)

    # dates
    _date = Instance(datetime)
    _modified = Instance(datetime)

    # metadata
    _author = Unicode()
    _description = Unicode()
    _origin = Unicode()
    _history = List(Unicode(), allow_none=True)
    _meta = Instance(Meta, allow_none=True)
    _transposed = Bool(False)

    # for linear data generation
    _offset = Union(
        (CFloat(), CInt(), Instance(Quantity)),
    )
    _increment = Union(
        (CFloat(), CInt(), Instance(Quantity)),
    )
    _size = Integer(0)
    _linear = Bool(False)

    # metadata

    # Basic NDArray setting
    _copy = Bool(False)  # by default we do not copy the data
    # which means that if the same numpy array
    # is used for too different NDArray, they
    # will share it.

    _labels_allowed = Bool(True)  # Labels are allowed for the data, if the

    # data are 1D only (they will essentially
    # serve as coordinates labelling.

    # other settings
    _text_width = Integer(120)
    _html_output = Bool(False)

    # Put this here as it is sometimes not found when only in NDIO
    _filename = Union((Instance(pathlib.Path), Unicode()), allow_none=True)

    # ..................................................................................................................
    def __init__(self, data=None, *args, **kwargs):
        """
        The basic |NDArray| object.

        The |NDArray| class is an array (numpy |ndarray|-like) container, usually not intended to be used directly,
        as its basic functionalities may be quite limited, but to be subclassed.

        Indeed, both the classes |NDDataset| and |Coord| which respectively implement a full dataset (with
        coordinates)  and the coordinates in a given dimension, are derived from |NDArray| in |scpy|.

        The key distinction from raw numpy |ndarray| is the presence of optional properties such as dimension names,
        labels, masks, units and/or extensible metadata dictionary.

        Parameters
        ----------
        data : array of floats
            Data array contained in the object. The data can be a list, a tuple, a |ndarray|, a ndarray-like,
            a |NDArray| or any subclass of |NDArray|. Any size or shape of data is accepted. If not given, an empty
            |NDArray| will be inited.
            At the initialisation the provided data will be eventually casted to a numpy-ndarray.
            If a subclass of |NDArray| is passed which already contains some mask, labels, or units, these elements
            will
            be used to accordingly set those of the created object. If possible, the provided data will not be copied
            for `data` input, but will be passed by reference, so you should make a copy of the `data` before passing
            them if that's the desired behavior or set the `copy` argument to True.

        Other Parameters
        ----------------
        dtype : str or dtype, optional, default=np.float64
            If specified, the data will be casted to this dtype, else the data will be casted to float64.
        dims : list of chars, optional.
            if specified the list must have a length equal to the number od data dimensions (ndim) and the chars
            must be
            taken among among x,y,z,u,v,w or t. If not specified, the dimension names are automatically attributed in
            this order.
        name : str, optional
            A user friendly name for this object. If not given, the automatic `id` given at the object creation will be
            used as a name.
        labels : array of objects, optional
            Labels for the `data`. labels can be used only for 1D-datasets.
            The labels array may have an additional dimension, meaning several series of labels for the same data.
            The given array can be a list, a tuple, a |ndarray|, a ndarray-like, a |NDArray| or any subclass of
            |NDArray|.
        mask : array of bool or `NOMASK`, optional
            Mask for the data. The mask array must have the same shape as the data. The given array can be a list,
            a tuple, or a |ndarray|. Each values in the array must be `False` where the data are *valid* and True when
            they are not (like in numpy masked arrays). If `data` is already a :class:`~numpy.ma.MaskedArray`, or any
            array object (such as a |NDArray| or subclass of it), providing a `mask` here will causes the mask from the
            masked array to be ignored.
        units : |Unit| instance or str, optional
            Units of the data. If data is a |Quantity| then `units` is set to the unit of the `data`; if a unit is also
            explicitly provided an error is raised. Handling of units use the `pint <https://pint.readthedocs.org/>`_
            package.
        title : str, optional
            The title of the dimension. It will later be used for instance for labelling plots of the data.
            It is optional but recommended to give a title to each ndarray.
        dlabel :  str, optional.
            Alias of `title`.
        meta : dict-like object, optional.
            Additional metadata for this object. Must be dict-like but no
            further restriction is placed on meta.
        author : str, optional.
            name(s) of the author(s) of this dataset. BNy default, name of the computer note where this dataset is
            created.
        description : str, optional.
            A optional description of the nd-dataset. A shorter alias is `desc`.
        history : str, optional.
            A string to add to the object history.
        copy : bool, optional
            Perform a copy of the passed object. Default is False.

        See Also
        --------
        NDDataset : Object which subclass |NDArray| with the addition of coordinates.
        Coord : Explicit coordinates object.
        LinearCoord : Implicit coordinates objet.

        Examples
        --------

        >>> myarray = scp.NDArray([1., 2., 3.])
        """
        super().__init__()

        # creation date

        self._date = datetime.now(timezone.utc)

        # by default, we try to keep a reference to the data, not copy them
        self._copy = kwargs.pop("copy", False)  #

        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            self._dtype = np.dtype(dtype)

        self._linear = kwargs.pop("linear", False)
        self._increment = kwargs.pop("increment", 1.0)
        self._offset = kwargs.pop("offset", 0.0)
        self._size = kwargs.pop("size", 0)

        # self._accuracy = kwargs.pop('accuracy', None)

        if data is not None:
            self.data = data

        if data is None or self.data is None:
            self._data = None
            self._dtype = None  # default

        if self._labels_allowed:
            self.labels = kwargs.pop("labels", None)

        self.title = kwargs.pop("title", self.title)

        mask = kwargs.pop("mask", NOMASK)
        if mask is not NOMASK:
            self.mask = mask

        if "dims" in kwargs.keys():
            self.dims = kwargs.pop("dims")

        self.units = kwargs.pop("units", None)

        self.meta = kwargs.pop("meta", None)

        self.name = kwargs.pop("name", None)

        self.description = kwargs.pop("description", kwargs.pop("desc", ""))

        author = kwargs.pop("author", get_user_and_node())
        if author:
            try:
                self.author = author
            except AttributeError:
                pass

        history = kwargs.pop("history", None)
        if history is not None:
            try:
                self.history = history
            except AttributeError:
                pass

        self._modified = self._date

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def __copy__(self):
        return self.copy(deep=False)

    # ..................................................................................................................
    def __deepcopy__(self, memo=None):
        return self.copy(deep=True, memo=memo)

    # ..................................................................................................................
    def __dir__(self):
        return [
            "data",
            "dims",
            "mask",
            "labels",
            "units",
            "meta",
            "title",
            "name",
            "origin",
            "roi",
            "author",
            "description",
            "history",
            "linear",
            "offset",
            "increment",
            "transposed",
        ]

    # ..................................................................................................................
    def __eq__(self, other, attrs=None):

        eq = True

        if not isinstance(other, NDArray):
            # try to make some assumption to make useful comparison.
            if isinstance(other, Quantity):
                otherdata = other.magnitude
                otherunits = other.units
            elif isinstance(other, (float, int, np.ndarray)):
                otherdata = other
                otherunits = False
            else:  # pragma: no cover
                return False

            if not self.has_units and not otherunits:
                eq = np.all(self._data == otherdata)
            elif self.has_units and otherunits:
                eq = np.all(self._data * self._units == otherdata * otherunits)
            else:  # pragma: no cover
                return False
            return eq

        if attrs is None:
            attrs = self.__dir__()

        for attr in ["name", "history"]:
            if attr in attrs:
                attrs.remove(attr)

        for attr in attrs:
            if attr != "units":
                sattr = getattr(self, f"_{attr}")
                if hasattr(other, f"_{attr}"):
                    oattr = getattr(other, f"_{attr}")
                    # to avoid deprecation warning issue for unequal array
                    if sattr is None and oattr is not None:
                        return False
                    if oattr is None and sattr is not None:
                        return False
                    if (
                        hasattr(oattr, "size")
                        and hasattr(sattr, "size")
                        and oattr.size != sattr.size
                    ):
                        # particular case of mask
                        if attr != "mask":
                            return False
                        else:
                            if other.mask != self.mask:
                                return False
                    eq &= np.all(sattr == oattr)
                    if not eq:
                        return False
                else:
                    return False
            else:
                # no unit and dimensionless are supposed equals
                sattr = self._units
                if sattr is None:
                    sattr = ur.dimensionless
                if hasattr(other, "_units"):
                    oattr = other._units
                    if oattr is None:
                        oattr = ur.dimensionless

                    eq &= np.all(sattr == oattr)
                    if not eq:
                        return False
                else:
                    return False

        return eq

    # ..................................................................................................................
    def __getitem__(self, items, return_index=False):

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
            udata = new.masked_data[keys]

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

        elif (self.data is not None) and hasattr(udata, "mask"):
            new._mask = udata.mask
        else:
            new._mask = NOMASK

        # for all other cases,
        # we do not needs to take care of dims, as the shape is not reduced by
        # this operation. Only a subsequent squeeze operation will do it
        if not return_index:
            return new
        else:
            return new, keys

    # ..................................................................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash((type(self), self.shape, self._units))

    # ..................................................................................................................
    def __iter__(self):
        # iterate on the first dimension
        if self.ndim == 0:
            error_("iteration over a 0-d array is not possible")
            return None

        for n in range(len(self)):
            yield self[n]

    # ..................................................................................................................
    def __len__(self):
        # len of the last dimension
        if not self.is_empty:
            return self.shape[0]
        else:
            return 0

    # ..................................................................................................................
    def __ne__(self, other, attrs=None):
        return not self.__eq__(other, attrs)

    # ..................................................................................................................
    def __repr__(self):
        out = f"{self._repr_value().strip()} ({self._repr_shape().strip()})"
        out = out.rstrip()
        return out

    # ..................................................................................................................
    def __setitem__(self, items, value):

        if self.linear:
            error_("Linearly define array are readonly")
            return

        keys = self._make_index(items)

        if isinstance(value, (bool, np.bool_, MaskedConstant)):
            # the mask is modified, not the data
            if value is MASKED:
                value = True
            if not np.any(self._mask):
                self._mask = np.zeros(self._data.shape).astype(np.bool_)
            self._mask[keys] = value
            return

        elif isinstance(value, Quantity):
            # first convert value to the current units
            value.ito(self.units)
            # self._data[keys] = np.array(value.magnitude, subok=True, copy=self._copy)
            value = np.array(value.magnitude, subok=True, copy=self._copy)

        if self._data.dtype == np.dtype(np.quaternion) and np.isscalar(value):
            # sometimes do not work directly : here is a work around
            self._data[keys] = np.full_like(self._data[keys], value).astype(
                np.dtype(np.quaternion)
            )
        else:
            self._data[keys] = value

    # ..................................................................................................................
    def __str__(self):
        return repr(self)

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @observe(All)
    def _anytrait_changed(self, change):

        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }

        if change["name"] in ["_date", "_modified", "trait_added"]:
            return

        if change.name in ["_linear", "_increment", "_offset", "_size"]:
            if self._linear:
                self._linearize()
            return

        # all the time -> update modified date
        self._modified = datetime.now(timezone.utc)
        return

    # ..................................................................................................................
    def _argsort(self, by="value", pos=None, descend=False):
        # found the indices sorted by values or labels
        if by == "value":
            args = np.argsort(self.data)
        elif "label" in by and not self.is_labeled:
            # by = 'value'
            # pos = None
            warnings.warn(
                "no label to sort, use `value` by default", SpectroChemPyWarning
            )
            args = np.argsort(self.data)
        elif "label" in by and self.is_labeled:
            labels = self._labels
            if len(self._labels.shape) > 1:
                # multidimensional labels
                if not pos:
                    pos = 0
                    # try to find a pos in the by string
                    pattern = re.compile(r"label\[(\d)]")
                    p = pattern.search(by)
                    if p is not None:
                        pos = int(p[1])
                labels = self._labels[..., pos]
            args = np.argsort(labels)
        else:
            # by = 'value'
            warnings.warn(
                "parameter `by` should be set to `value` or `label`, use `value` by default",
                SpectroChemPyWarning,
            )
            args = np.argsort(self.data)
        if descend:
            args = args[::-1]
        return args

    # ..................................................................................................................
    def _cstr(self):
        out = f"{self._str_value()}\n{self._str_shape()}"
        out = out.rstrip()
        return out

    # ..................................................................................................................
    @default("_data")
    def _data_default(self):
        return None

    # ..................................................................................................................
    @validate("_data")
    def _data_validate(self, proposal):
        # validation of the _data attribute
        data = proposal["value"]

        # cast to the desired type
        if self._dtype is not None:
            data = data.astype(np.dtype(self._dtype, copy=False))

        # return the validated data
        if self._copy:
            return data.copy()
        else:
            return data

    # ..................................................................................................................
    @default("_dims")
    def _dims_default(self):
        return DEFAULT_DIM_NAME[-self.ndim :]

    # ..................................................................................................................
    def _get_dims_from_args(self, *dims, **kwargs):
        # utility function to read dims args and kwargs
        # sequence of dims or axis, or `dim`, `dims` or `axis` keyword are accepted

        # check if we have arguments
        if not dims:
            dims = None

        # Check if keyword dims (or synonym axis) exists
        axis = kwargs.pop("axis", None)

        kdims = kwargs.pop("dims", kwargs.pop("dim", axis))  # dim or dims keyword
        if kdims is not None:
            if dims is not None:
                warnings.warn(
                    "the unamed arguments are interpreted as `dims`. But a named argument `dims` or `axis`"
                    "(DEPRECATED) has been specified. \nThe unamed arguments will thus be ignored.",
                    SpectroChemPyWarning,
                )
            dims = kdims

        if dims is not None and not isinstance(dims, list):
            if isinstance(dims, tuple):
                dims = list(dims)
            else:
                dims = [dims]

        if dims is not None:
            for i, item in enumerate(dims[:]):
                if item is not None and not isinstance(item, str):
                    item = self.dims[item]
                dims[i] = item

        if dims is not None and len(dims) == 1:
            dims = dims[0]

        return dims

    # ..................................................................................................................
    def _get_dims_index(self, dims):
        # get the index(es) corresponding to the given dim(s) which can be expressed as integer or string

        if dims is None:
            return

        if is_sequence(dims):
            if np.all([d is None for d in dims]):
                return
        else:
            dims = [dims]

        axis = []
        for dim in dims:
            if isinstance(dim, TYPE_INTEGER):
                axis.append(dim)  # nothing else to do

            elif isinstance(dim, str):
                if dim not in self.dims:
                    raise ValueError(
                        f"Error: Dimension `{dim}` is not recognized "
                        f"(not in the dimension's list: {self.dims})."
                    )
                id = self.dims.index(dim)
                axis.append(id)

            else:
                raise TypeError(
                    f"Dimensions must be specified as string or integer index, but a value of type "
                    f"{type(dim)} has been passed (value:{dim})!"
                )

        for i, item in enumerate(axis):
            # convert to positive index
            if item < 0:
                axis[i] = self.ndim + item

        axis = tuple(axis)

        return axis

    # ..................................................................................................................
    def _get_slice(self, key, dim):

        info = None
        if not isinstance(key, slice):
            # integer or float
            start = key
            if not isinstance(key, TYPE_INTEGER):
                start = self._loc2index(key, dim)
                if isinstance(start, tuple):
                    start, info = start
                if start is None:
                    return slice(None)
            else:
                if key < 0:  # reverse indexing
                    axis, dim = self.get_axis(dim)
                    start = self.shape[axis] + key
            stop = start + 1  # in order to keep an non squeezed slice
            return slice(start, stop, 1)
        else:
            start, stop, step = key.start, key.stop, key.step
            if start is not None and not isinstance(start, TYPE_INTEGER):
                start = self._loc2index(start, dim)
                if isinstance(start, tuple):
                    start, info = start
            if stop is not None and not isinstance(stop, TYPE_INTEGER):
                stop = self._loc2index(stop, dim)
                if isinstance(stop, tuple):
                    stop, info = stop
                if start is not None and stop < start:
                    start, stop = stop, start
                if stop != start:
                    stop = stop + 1  # to include last loc or label index
            if step is not None and not isinstance(step, (int, np.int_, np.int64)):
                raise NotImplementedError(
                    "step in location slicing is not yet possible."
                )  # TODO: we have may be a special case with
                # datetime  # step = 1
        if step is None:
            step = 1
        if start is not None and stop is not None and start == stop and info is None:
            stop = stop + 1  # to include last index

        newkey = slice(start, stop, step)
        return newkey

    # ..................................................................................................................
    @default("_id")
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    @default("_increment")
    def _increment_default(self):
        return 1.0

    # ..................................................................................................................
    @default("_labels")
    def _labels_default(self):
        return None

    # ..................................................................................................................
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

    # ..................................................................................................................
    def _loc2index(self, loc, dim=None):
        # Return the index of a location (label or values such as coordinates) along a 1D array.
        # Do not apply for multidimensional arrays (ndim>1)
        if self.ndim > 1:
            raise NotImplementedError(
                f"not implemented for {type(self).__name__} objects which are not 1-dimensional "
                f"(current ndim:{self.ndim})"
            )

        if self.is_empty and not self.is_labeled:

            raise IndexError(f"Could not find this location: {loc} on an empty array")

        else:

            data = self.data

            if is_number(loc):
                # get the index of a given values
                error = None
                if np.all(loc > data.max()) or np.all(loc < data.min()):
                    print_(
                        f"This coordinate ({loc}) is outside the axis limits ({data.min()}-{data.max()}).\n"
                        f"The closest limit index is returned"
                    )
                    error = "out_of_limits"
                index = (np.abs(data - loc)).argmin()
                # TODO: add some precision to this result
                if not error:
                    return index
                else:
                    return index, error

            elif is_sequence(loc):
                # TODO: is there a simpler way to do this with numpy functions
                index = []
                for lo in loc:
                    index.append(
                        (np.abs(data - lo)).argmin()
                    )  # TODO: add some precision to this result
                return index

            elif self.is_labeled:

                # TODO: look in all row of labels
                labels = self._labels
                indexes = np.argwhere(labels == loc).flatten()
                if indexes.size > 0:
                    return indexes[0]
                else:
                    raise IndexError(f"Could not find this label: {loc}")

            elif isinstance(loc, datetime):
                # not implemented yet
                raise NotImplementedError(
                    "datetime as location is not yet impemented"
                )  # TODO: date!

            else:
                raise IndexError(f"Could not find this location: {loc}")

    # ..................................................................................................................
    def _make_index(self, key):

        if isinstance(key, np.ndarray) and key.dtype == bool:
            # this is a boolean selection
            # we can proceed directly
            return key

        # we need to have a list of slice for each argument
        # or a single slice acting on the axis=0
        if isinstance(key, tuple):
            keys = list(key)
        else:
            keys = [
                key,
            ]

        def ellipsisinkeys(keys):
            try:
                # Ellipsis
                if isinstance(keys[0], np.ndarray):
                    return False
                test = Ellipsis in keys
            except ValueError as e:
                if e.args[0].startswith("The truth "):
                    # probably an array of index (Fancy indexing)  # should not happen any more with the test above
                    test = False
            return test

        while ellipsisinkeys(keys):
            i = keys.index(Ellipsis)
            keys.pop(i)
            for j in range(self.ndim - len(keys)):
                keys.insert(i, slice(None))

        if len(keys) > self.ndim:
            raise IndexError("invalid index")

        # pad the list with additional dimensions
        for i in range(len(keys), self.ndim):
            keys.append(slice(None))

        for axis, key in enumerate(keys):
            # the keys are in the order of the dimension in self.dims!
            # so we need to get the correct dim in the coordinates lists
            dim = self.dims[axis]
            if is_sequence(key):
                # fancy indexing
                # all items of the sequence must be integer index
                keys[axis] = key
            else:
                keys[axis] = self._get_slice(key, dim)
        return tuple(keys)

    # ..................................................................................................................
    @default("_mask")
    def _mask_default(self):
        return NOMASK if self._data is None else np.zeros(self._data.shape).astype(bool)

    # ..................................................................................................................
    @validate("_mask")
    def _mask_validate(self, proposal):
        pv = proposal["value"]
        mask = pv

        if mask is None or mask is NOMASK:
            return mask

        # mask will be stored in F_CONTIGUOUS mode, if data are in this mode
        if not mask.flags["F_CONTIGUOUS"] and self._data.flags["F_CONTIGUOUS"]:
            mask = np.asfortranarray(mask)
            # no more need for an eventual copy
            self._copy = False

        # no particular validation for now.
        if self._copy:
            return mask.copy()
        else:
            return mask

    # ..................................................................................................................
    @default("_meta")
    def _meta_default(self):
        return Meta()

    # ..................................................................................................................
    @default("_name")
    def _name_default(self):
        return ""

    # ..................................................................................................................
    @default("_offset")
    def _offset_default(self):
        return 0

    # ..................................................................................................................
    def _repr_html_(self):
        return convert_to_html(self)

    # ..................................................................................................................
    def _repr_shape(self):

        if self.is_empty:
            return "size: 0"

        out = ""

        shape_ = (
            x for x in itertools.chain.from_iterable(list(zip(self.dims, self.shape)))
        )

        shape = (", ".join(["{}:{}"] * self.ndim)).format(*shape_)

        size = self.size

        out += f"size: {size}" if self.ndim < 2 else f"shape: ({shape})"

        return out

    # ..................................................................................................................
    def _repr_value(self):

        numpyprintoptions(precision=4, edgeitems=0, spc=1, linewidth=120)

        prefix = type(self).__name__ + ": "
        units = ""

        size = ""
        if not self.is_empty:

            if self.data is not None:

                dtype = self.dtype
                data = ""
                if self.implements("Coord") or self.implements("LinearCoord"):
                    size = f" (size: {self.data.size})"
                units = " {:~K}".format(self.units) if self.has_units else " unitless"

            else:
                # no data but labels
                lab = self.get_labels()
                data = f" {lab}"
                size = f" (size: {len(lab)})"
                dtype = "labels"

            body = f"[{dtype}]{data}"

        else:
            size = ""
            body = "empty"

        numpyprintoptions()
        return "".join([prefix, body, units, size])

    # ..................................................................................................................
    def _set_data(self, data):

        if data is None:
            return

        elif isinstance(data, NDArray) and data.linear:
            # Case of Linearcoord
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

    # ..................................................................................................................
    def _sort(self, by="value", pos=None, descend=False, inplace=False):
        # sort an ndarray using data or label values

        args = self._argsort(by, pos, descend)

        if not inplace:
            new = self[args]
        else:
            new = self[args, INPLACE]

        return new

    # ..................................................................................................................
    @property
    def _squeeze_ndim(self):
        # The number of dimensions of the squeezed`data` array (Readonly property).

        if self.data is None and self.is_labeled:
            return 1
        if self.data is None:
            return 0
        else:
            return len([x for x in self.data.shape if x > 1])

    # ..................................................................................................................
    def _str_shape(self):

        if self.is_empty:
            return "         size: 0\n"

        out = ""

        shape_ = (
            x for x in itertools.chain.from_iterable(list(zip(self.dims, self.shape)))
        )

        shape = (", ".join(["{}:{}"] * self.ndim)).format(*shape_)

        size = self.size

        out += (
            f"         size: {size}\n"
            if self.ndim < 2
            else f"        shape: ({shape})\n"
        )

        return out

    # ..................................................................................................................
    def _str_value(self, sep="\n", ufmt=" {:~K}", header="         data: ... \n"):
        # prefix = ['']
        if self.is_empty and "data: ..." not in header:
            return header + "{}".format(textwrap.indent("empty", " " * 9))
        elif self.is_empty:
            return "{}".format(textwrap.indent("empty", " " * 9))

        print_unit = True
        units = ""

        def mkbody(d, pref, units):
            # work around printing masked values with formatting
            ds = d.copy()
            if self.is_masked:
                dtype = self.data.dtype
                mask_string = f"--{dtype}"
                ds = insert_masked_print(ds, mask_string=mask_string)
            body = np.array2string(ds, separator=" ", prefix=pref)
            body = body.replace("\n", sep)
            text = "".join([pref, body, units])
            text += sep
            return text

        text = ""
        if not self.is_empty:

            if self.data is not None:
                data = self.umasked_data
            else:
                # no data but labels
                data = self.get_labels()
                print_unit = False

            if isinstance(data, Quantity):
                data = data.magnitude

            if print_unit:
                units = ufmt.format(self.units) if self.has_units else ""

            text += mkbody(data, "", units)

        out = ""  # f'        title: {self.title}\n' if self.title else ''
        text = text.strip()
        if "\n" not in text:  # single line!
            out += header.replace("...", f"\0{text}\0")
        else:
            out += header
            out += "\0{}\0".format(textwrap.indent(text, " " * 9))
        out = out.rstrip()  # remove the trailings '\n'
        return out

    # ..................................................................................................................
    @default("_title")
    def _title_default(self):
        return None

    @default("_roi")
    def _roi_default(self):
        return None

    # ..................................................................................................................
    @staticmethod
    def _uarray(data, units=None):
        # return the array or scalar with units
        # if data.size==1:
        #    uar = data.squeeze()[()]
        # else:
        uar = data
        if units:
            return Quantity(uar, units)
        else:
            return uar

    # ..................................................................................................................
    @staticmethod
    def _umasked(data, mask):
        # This ensures that a masked array is returned.

        if not np.any(mask):
            mask = np.zeros(data.shape).astype(bool)
        data = np.ma.masked_where(mask, data)  # np.ma.masked_array(data, mask)

        return data

    # ------------------------------------------------------------------------------------------------------------------
    # Public Methods and Properties
    # ------------------------------------------------------------------------------------------------------------------
    def asfortranarray(self):
        """
        Make data and mask (ndim >= 1) laid out in Fortran order in memory.
        """
        # data and mask will be converted to F_CONTIGUOUS mode
        if not self._data.flags["F_CONTIGUOUS"]:
            self._data = np.asfortranarray(self._data)
            if self.is_masked:
                self._mask = np.asfortranarray(self._mask)

    def astype(self, dtype=None, **kwargs):
        """
        Cast the data to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            typecode or data-type to which the array is cast.
        """
        if not self.linear:
            self._data = self._data.astype(dtype, **kwargs)
        else:
            self._increment = np.array(self._increment).astype(dtype, **kwargs)[()]
            self._offset = np.array(self._offset).astype(dtype, **kwargs)[()]
        return self

    # .................................................................................................................
    @property
    def author(self):
        """
        str - creator of the array
        """
        return self._author

    # ..................................................................................................................
    @author.setter
    def author(self, value):
        self._author = value

    # ..................................................................................................................
    def copy(self, deep=True, memo=None, keepname=False):
        """
        Make a disconnected copy of the current object.

        Parameters
        ----------
        deep : bool, optional
            If True a deepcopy is performed which is the default behavior.
        memo : Not used
            This parameter ensure compatibility with deepcopy() from the copy package.
        keepname : bool
            If True keep the same name for the copied object.

        Returns
        -------
        object
            An exact copy of the current object.

        Examples
        --------

        >>> nd1 = scp.NDArray([1. + 2.j, 2. + 3.j])
        >>> nd1
        NDArray: [complex128] unitless (size: 2)
        >>> nd2 = nd1
        >>> nd2 is nd1
        True
        >>> nd3 = nd1.copy()
        >>> nd3 is not nd1
        True
        """

        if deep:
            do_copy = cpy.deepcopy
        else:
            do_copy = cpy.copy

        new = make_new_object(self)
        for (
            attr
        ) in (
            self.__dir__()
        ):  # do not use  dir(self) as the order in __dir__ list is important
            try:
                _attr = do_copy(getattr(self, f"_{attr}"))
                setattr(new, f"_{attr}", _attr)

            except ValueError:
                # ensure that if deepcopy do not work, a shadow copy can be done
                _attr = do_copy(getattr(self, f"_{attr}"))
                setattr(new, f"_{attr}", _attr)

        # name must be changed
        if not keepname:
            new.name = ""  # default

        return new

    # ..................................................................................................................
    @property
    def created(self):
        """
        `Datetime` - creation date object
        """
        return self._date

    # ..................................................................................................................
    @created.setter
    def created(self, date):

        self.date = date

    # ..................................................................................................................
    @property
    def data(self):
        """
        |ndarray| - The `data` array.

        If there is no data but labels, then the labels are returned instead of data.
        """
        if self._data is None and not self.linear:
            return None

        elif self.linear:
            data = np.arange(self.size) * self._increment + self._offset
            if hasattr(data, "units"):
                data = data.m
        else:
            data = self._data

        return data

    # ..................................................................................................................
    @data.setter
    def data(self, data):
        # property.setter for data
        # note that a subsequent validation is done in _data_validate
        # NOTE: as property setter doesn't work with super(),
        # see https://stackoverflow.com/questions/10810369/python-super-and-setting-parent-class-property
        # we use an intermediate function that can be called from a subclass

        self._set_data(data)

    # ..................................................................................................................
    magnitude = data
    m = data

    # ..................................................................................................................
    @property
    def date(self):
        """
        `Datetime` - creation date object - equivalent to the attribute `created`
        """
        return self._date

    # ..................................................................................................................
    @date.setter
    def date(self, date):

        if isinstance(date, datetime):
            self._date = date

        elif isinstance(date, str):
            try:
                self._date = datetime.strptime(date, "%Y/%m/%d")
            except ValueError:
                self._date = datetime.strptime(date, "%d/%m/%Y")

    # .................................................................................................................
    @property
    def description(self):
        """
        str - Provides a description of the underlying data
        """
        return self._description

    desc = description
    desc.__doc__ = """Alias to the description attribute"""

    # ..................................................................................................................
    @description.setter
    def description(self, value):
        self._description = value

    # ..................................................................................................................
    @property
    def dimensionless(self):
        """
        bool - True if the `data` array is dimensionless (Readonly property).

        Notes
        -----
        `Dimensionless` is different of `unitless` which means no unit.

        See Also
        --------
        unitless, has_units
        """
        if self.unitless:
            return False
        return self._units.dimensionless

    # ..................................................................................................................
    @property
    def dims(self):
        """
        list -  Names of the dimensions

        The name of the dimensions are 'x', 'y', 'z'.... depending on the number of dimension.
        """
        ndim = self.ndim
        if ndim > 0:
            # if len(self._dims)< ndim:
            #    self._dims = self._dims_default()
            dims = self._dims[:ndim]
            return dims
        else:
            return []

    # ..................................................................................................................
    @dims.setter
    def dims(self, values):

        if isinstance(values, str) and len(values) == 1:
            values = [values]

        if not is_sequence(values) or len(values) != self.ndim:
            raise ValueError(
                f"a sequence of chars with a length of {self.ndim} is expected, but `{values}` "
                f"has been provided"
            )

        for value in values:
            if value not in DEFAULT_DIM_NAME:
                raise ValueError(
                    f"{value} value is not admitted. Dimension's name must be among "
                    f"{DEFAULT_DIM_NAME[::-1]}."
                )

        self._dims = tuple(values)

    @property
    def filename(self):
        """
        `Pathlib` object - current filename for this dataset.
        """
        if self._filename:
            return self._filename.stem + self.suffix
        else:
            return None

    @filename.setter
    def filename(self, val):
        self._filename = pathclean(val)

    # ..................................................................................................................
    @property
    def dtype(self):
        """
        numpy dtype - data type
        """
        if self.is_empty:
            self._dtype = None
        else:
            self._dtype = self.data.dtype
        return self._dtype

    # ..................................................................................................................
    def get_axis(self, *args, **kwargs):
        """
        Helper function to determine an axis index whatever the syntax used (axis index or dimension names).

        Parameters
        ----------
        dim, axis, dims : str, int, or list of str or index
            The axis indexes or dimensions names - they can be specified as argument or using keyword 'axis', 'dim'
            or 'dims'.
        negative_axis : bool, optional, default=False
            If True a negative index is returned for the axis value (-1 for the last dimension, etc...).
        allows_none : bool, optional, default=False
            If True, if input is none then None is returned.
        only_first : bool, optional, default: True
            By default return only information on the first axis if dim is a list.
            Else, return an list for axis and dims information.

        Returns
        -------
        axis : int
            The axis indexes.
        dim : str
            The axis name.
        """
        # handle the various syntax to pass the axis
        dims = self._get_dims_from_args(*args, **kwargs)
        axis = self._get_dims_index(dims)
        allows_none = kwargs.get("allows_none", False)

        if axis is None and allows_none:
            return None, None

        if isinstance(axis, tuple):
            axis = list(axis)

        if not isinstance(axis, list):
            axis = [axis]

        dims = axis[:]
        for i, a in enumerate(axis[:]):
            # axis = axis[0] if axis else self.ndim - 1  # None
            if a is None:
                a = self.ndim - 1
            if kwargs.get("negative_axis", False):
                if a >= 0:
                    a = a - self.ndim
            axis[i] = a
            dims[i] = self.dims[a]

        only_first = kwargs.pop("only_first", True)

        if len(dims) == 1 and only_first:
            dims = dims[0]
            axis = axis[0]

        return axis, dims

    # ..................................................................................................................
    def get_labels(self, level=0):
        """Get the labels at a given level.

        Used to replace `data` when only labels are provided, and/or for
        labeling axis in plots.

        Parameters
        ----------
        level : int, optional, default:0

        Returns
        -------
        |ndarray|
            The labels at the desired level or None.
        """
        if not self.is_labeled:
            return None

        if level > self.labels.ndim - 1:
            warnings.warn(
                "There is no such level in the existing labels", SpectroChemPyWarning
            )
            return None

        if self.labels.ndim > 1:
            return self.labels[level]
        else:
            return self._labels

    # ..................................................................................................................
    @property
    def has_data(self):
        """
        bool - True if the `data` array is not empty and size > 0.
        (Readonly property).
        """
        if (self.data is None) or (self.data.size == 0):
            return False

        return True

    # ..................................................................................................................
    @property
    def has_defined_name(self):
        """
        bool - True is the name has been defined
        """
        return not (self.name == self.id)

    # ..................................................................................................................
    @property
    def has_units(self):
        """
        bool - True if the `data` array have units (Readonly property).

        See Also
        --------
        unitless, dimensionless
        """
        if self._units:
            if not str(self.units).strip():
                return False
            return True
        return False

    # ..................................................................................................................
    @property
    def history(self):
        """
        List of strings - Describes the history of actions made on this array
        """
        return self._history

    # ..................................................................................................................
    @history.setter
    def history(self, value):
        self._history.append(value)

    # ..................................................................................................................
    @property
    def id(self):
        """
        str - Object identifier (Readonly property).
        """
        return self._id

    # ..................................................................................................................
    @property
    def imag(self):
        return None

    # ..................................................................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `NDArray`.

        Rather than isinstance(obj, NDArrray) use object.implements('NDArray').

        This is useful to check type without importing the module
        """
        if name is None:
            return "NDArray"
        else:
            return name == "NDArray"

    # ..................................................................................................................
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

    # ..................................................................................................................
    @property
    def is_float(self):
        """
        bool - True if the `data` are real values (Readonly property).
        """
        if self.data is None:
            return False

        return self.data.dtype in TYPE_FLOAT

    # ..................................................................................................................
    @property
    def is_integer(self):
        """
        bool - True if the `data` are integer values (Readonly property).
        """
        if self.data is None:
            return False

        return self.data.dtype in TYPE_INTEGER

    # ..................................................................................................................
    @property
    def is_1d(self):
        """
        bool - True if the `data` array has only one dimension
        """
        return self.ndim == 1

    # ..................................................................................................................
    @property
    def is_empty(self):
        """
        bool - True if the `data` array is empty or size=0, and if no label are present
        (Readonly property).
        """
        if (
            not self.linear
            and ((self._data is None) or (self._data.size == 0))
            and not self.is_labeled
        ):
            return True

        return False

    # ..................................................................................................................
    @property
    def is_labeled(self):
        """
        bool - True if the `data` array have labels (Readonly property).
        """
        # label cannot exists (for now for nD dataset - only 1D dataset, such
        # as Coord can be labelled.
        if self._data is not None and self.ndim > 1:
            return False
        if self._labels is not None and np.any(self.labels != ""):
            return True
        else:
            return False

    # ..................................................................................................................
    @property
    def linear(self):
        """
        bool - flag to specify if the data can be constructed using a linear variation
        """
        return self._linear

    @linear.setter
    def linear(self, val):

        self._linear = (
            val  # it val is true this provoque the linearization (  # see observe)
        )

        # if val and self._data is not None:  #     # linearisation of the data, if possible  #     self._linearize()

    # ..................................................................................................................
    @property
    def is_masked(self):
        """
        bool - True if the `data` array has masked values (Readonly property).
        """
        if isinstance(self._mask, np.ndarray):
            return np.any(self._mask)
        elif self._mask == NOMASK or self._mask is None:
            return False
        elif isinstance(self._mask, (np.bool_, bool)):
            return self._mask

        return False

    # ..................................................................................................................
    def is_units_compatible(self, other):
        """
        Check the compatibility of units with another object.

        Parameters
        ----------
        other : |ndarray|
            The ndarray object for which we want to compare units compatibility.

        Returns
        -------
        result
            True if units are compatible.

        Examples
        --------
        >>> nd1 = scp.NDDataset([1. + 2.j, 2. + 3.j], units='meters')
        >>> nd1
        NDDataset: [complex128] m (size: 2)
        >>> nd2 = scp.NDDataset([1. + 2.j, 2. + 3.j], units='seconds')
        >>> nd1.is_units_compatible(nd2)
        False
        >>> nd1.ito('minutes', force=True)
        >>> nd1.is_units_compatible(nd2)
        True
        >>> nd2[0].values * 60. == nd1[0].values
        True
        """
        try:
            other.to(self.units, inplace=False)
        except DimensionalityError:
            return False
        return True

    # ..................................................................................................................
    @property
    def itemsize(self):
        """
        numpy itemsize - data type size
        """
        if self.data is None:
            return None

        return self.data.dtype.itemsize

    # ..................................................................................................................
    @property
    def iterdims(self):
        return list(range(self.ndim))

    # ..................................................................................................................
    def ito(self, other, force=False):
        """
        Inplace scaling of the current object data to different units.
        (same as `to` with inplace= True).

        Parameters
        ----------
        other : |Unit|, |Quantity| or str
            Destination units.
        force : bool, optional, default=`False`
            If True the change of units is forced, even for incompatible units

        See Also
        --------
        to : Rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of the current object data to different units.
        to_reduced_units : Rescaling to reduced units.
        ito_reduced_units : Rescaling to reduced units.
        """
        self.to(other, inplace=True, force=force)

    # ..................................................................................................................
    def ito_base_units(self):
        """
        Inplace rescaling to base units.

        See Also
        --------
        to : Rescaling of the current object data to different units.
        ito : Inplace rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        to_reduced_units : Rescaling to redunced units.
        ito_reduced_units : Inplace rescaling to reduced units.
        """
        self.to_base_units(inplace=True)

    # ..................................................................................................................
    def ito_reduced_units(self):
        """
        Quantity scaled in place to reduced units, inplace.

        Scaling to reduced units means, one unit per
        dimension. This will not reduce compound units (e.g., 'J/kg' will not
        be reduced to m**2/s**2).

        See Also
        --------
        to : Rescaling of the current object data to different units.
        ito : Inplace rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of the current object data to different units.
        to_reduced_units : Rescaling to reduced units.
        """
        self.to_reduced_units(inplace=True)

    # ..................................................................................................................
    @property
    def labels(self):
        """
        |ndarray| (str) - An array of labels for `data`.

        An array of objects of any type (but most generally string), with the last dimension size equal to that of the
        dimension of data. Note that's labelling is possible only for 1D data. One classical application is
        the labelling of coordinates to display informative strings instead of numerical values.
        """
        return self._labels

    # ..................................................................................................................
    @labels.setter
    def labels(self, labels):

        if labels is None:
            return

        if self.ndim > 1:
            warnings.warn(
                "We cannot set the labels for multidimentional data - Thus, these labels are ignored",
                SpectroChemPyWarning,
            )
        else:

            # make sure labels array is of type np.ndarray or Quantity arrays
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels, subok=True, copy=True).astype(
                    object, copy=False
                )

            if not np.any(labels):
                # no labels
                return

            else:
                if (self.data is not None) and (labels.shape[0] != self.shape[0]):
                    # allow the fact that the labels may have been passed in a transposed array
                    if labels.ndim > 1 and (labels.shape[-1] == self.shape[0]):
                        labels = labels.T
                    else:
                        raise ValueError(
                            f"labels {labels.shape} and data {self.shape} shape mismatch!"
                        )

                if np.any(self._labels):
                    info_(
                        f"{type(self).__name__} is already a labeled array.\nThe explicitly provided labels will "
                        f"be appended to the current labels"
                    )

                    labels = labels.squeeze()
                    self._labels = self._labels.squeeze()
                    if self._labels.ndim > 1:
                        self._labels = self._labels.T
                    self._labels = np.vstack((self._labels, labels)).T

                else:
                    if self._copy:
                        self._labels = labels.copy()
                    else:
                        self._labels = labels

    # ..................................................................................................................
    @property
    def limits(self):
        """list - range of the data"""
        if self.data is None:
            return None

        return [self.data.min(), self.data.max()]

    # ..................................................................................................................
    @property
    def mask(self):
        """
        |ndarray| (bool) - Mask for the data
        """
        if not self.is_masked:
            return NOMASK

        return self._mask

    # ..................................................................................................................
    @mask.setter
    def mask(self, mask):

        if mask is NOMASK or mask is MASKED:
            pass
        elif isinstance(mask, (np.bool_, bool)):
            if not mask:
                mask = NOMASK
            else:
                mask = MASKED
        else:
            # from now, make sure mask is of type np.ndarray if it provided
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.bool_)
            if not np.any(mask):
                # all element of the mask are false
                mask = NOMASK
            elif mask.shape != self.shape:
                raise ValueError(
                    f"mask {mask.shape} and data {self.shape} shape mismatch!"
                )

        # finally set the mask of the object
        if isinstance(mask, MaskedConstant):
            self._mask = (
                NOMASK if self.data is None else np.ones(self.shape).astype(bool)
            )
        else:
            if np.any(self._mask):
                # this should happen when a new mask is added to an existing one
                # mask to be combined to an existing one
                info_(
                    f"{type(self).__name__} is already a masked array.\n The new mask will be combined with the "
                    f"current array's mask."
                )
                self._mask |= mask  # combine (is a copy!)
            else:
                if self._copy:
                    self._mask = mask.copy()
                else:
                    self._mask = mask

    # ..................................................................................................................
    @property
    def masked_data(self):
        """
        |ma.ndarray| - The actual masked `data` array (Readonly property).
        """
        if self.is_masked and not self.is_empty:
            return self._umasked(self.data, self.mask)
        else:
            return self.data

    # ..................................................................................................................
    @property
    def meta(self):
        """
        |Meta| - Additional metadata.
        """
        return self._meta

    # ..................................................................................................................
    @meta.setter
    def meta(self, meta):

        if meta is not None:
            self._meta.update(meta)

    # ..................................................................................................................
    @property
    def modified(self):
        """
        `Datetime` object - Date of modification (readonly property).
        """
        return self._modified

    # ..................................................................................................................
    @property
    def name(self):
        """
        str - An user friendly name.

        When the name is not provided, the `id` of the object is retruned instead
        """
        if self._name:
            return self._name
        else:
            return self._id

    # ..................................................................................................................
    @name.setter
    def name(self, name):

        if name:
            if self._name:
                pass
            self._name = name

    # ..................................................................................................................
    @property
    def ndim(self):
        """
        int - The number of dimensions of the `data` array (Readonly property).
        """
        if self.linear:
            return 1

        if self.data is None and self.is_labeled:
            return 1

        if not self.size:
            return 0
        else:
            return self.data.ndim

    # ..................................................................................................................
    @property
    def offset(self):
        return self._offset

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

    @property
    def offset_value(self):
        offset = self.offset
        if self.units:
            return Quantity(offset, self._units)
        else:
            return offset

    # ..................................................................................................................
    @property
    def origin(self):
        """
        str - origin of the data
        """
        return self._origin

    # ..................................................................................................................
    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def real(self):
        return self

    # ..................................................................................................................
    def remove_masks(self):
        """
        Remove all masks previously set on this array.
        """
        self._mask = NOMASK

    # ..................................................................................................................
    @property
    def roi(self):
        """list - region of interest (ROI) limits"""
        if self._roi is None:
            self._roi = self.limits
        return self._roi

    @roi.setter
    def roi(self, val):
        self._roi = val

    @property
    def roi_values(self):
        if self.units is None:
            return list(np.array(self.roi))
        else:
            return list(self._uarray(self.roi, self.units))

    # ..................................................................................................................
    @property
    def shape(self):
        """
        tuple on int - A tuple with the size of each dimensions (Readonly property).
        The number of `data` element on each dimensions (possibly complex).
        For only labelled array, there is no data, so it is the 1D and the size is the size of the array of labels.
        """
        if self.linear:
            return (self._size,)

        if self.data is None and self.is_labeled:
            return (self.labels.shape[0],)

        elif self.data is None:
            return ()

        else:
            return self.data.shape

    # ..................................................................................................................
    @property
    def size(self):
        """
        int - Size of the underlying `data` array (Readonly property).
        The total number of data element (possibly complex or hypercomplex
        in the array).
        """

        if self._data is None and self.is_labeled:
            return self.labels.shape[-1]

        elif self._data is None:
            if self.linear:
                # the provided size is returned i or its default
                return self._size
            return None
        else:
            return self._data.size

    # ..................................................................................................................
    @property
    def spacing(self):
        # return a scalar for the spacing of the coordinates (if they are uniformly spaced,
        # else return an array of the differents spacings
        if self.linear:
            return self.increment * self.units
        return spacing(self.data) * self.units

    # ..................................................................................................................
    def squeeze(self, *dims, inplace=False, return_axis=False, **kwargs):
        """
        Remove single-dimensional entries from the shape of an array.

        Parameters
        ----------
        dims : None or int or tuple of ints, optional
            Selects a subset of the single-dimensional entries in the
            shape. If a dimension (dim) is selected with shape entry greater than
            one, an error is raised.

        Returns
        -------
        squeezed : same object type
            The input array, but with all or a subset of the
            dimensions of length 1 removed.

        Raises
        ------
        ValueError
            If `dims` is not `None`, and the dimension being squeezed is not
            of length 1
        """
        if inplace:
            new = self
        else:
            new = self.copy()

        dims = self._get_dims_from_args(*dims, **kwargs)

        if not dims:
            s = np.array(new.shape)
            dims = np.argwhere(s == 1).squeeze().tolist()
        axis = self._get_dims_index(dims)
        if axis is None:
            # nothing to squeeze
            return new, axis

        # recompute new dims
        for i in axis[::-1]:
            del new._dims[i]

        # performs all required squeezing
        new._data = new._data.squeeze(axis=axis)
        if self.is_masked:
            new._mask = new._mask.squeeze(axis=axis)

        if return_axis:  # in case we need to know which axis has been squeezed
            return new, axis

        return new

    # ..................................................................................................................
    def swapdims(self, dim1, dim2, inplace=False):
        """
        Interchange two dims of a NDArray.

        Parameters
        ----------
        dim1 : int or str
            First dimension index.
        dim2 : int
            Second dimension index.
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        swaped_array

        See Also
        --------
        transpose
        """
        if not inplace:
            new = self.copy()
        else:
            new = self
        if self.ndim < 2:  # cannot swap axe for 1D data
            return new

        i0, i1 = axis = self._get_dims_index([dim1, dim2])
        new._data = np.swapaxes(new._data, *axis)
        new._dims[i1], new._dims[i0] = self._dims[i0], self._dims[i1]

        # all other arrays, except labels have also to be swapped to reflect
        # changes of data ordering.
        # labels are presents only for 1D array, so we do not have to swap them
        if self.is_masked:
            new._mask = np.swapaxes(new._mask, *axis)

        new._meta = new._meta.swap(*axis, inplace=False)
        return new

    swapaxes = swapdims
    swapaxes.__doc__ = "Alias of `swapdims`."

    # ..................................................................................................................
    @property
    def T(self):
        """
        |NDArray| - Transposed array.

        The same object is returned if `ndim` is less than 2.
        """
        return self.transpose()

    # ..................................................................................................................
    @property
    def title(self):
        """
        str - An user friendly title.

        When the title is provided, it can be used for labeling the object,
        e.g., axe title in a matplotlib plot.
        """
        if self._title:
            return self._title
        else:
            return "<untitled>"

    @title.setter
    def title(self, title):

        if title:
            self._title = title

    # ..................................................................................................................
    def to(self, other, inplace=False, force=False):
        """
        Return the object with data rescaled to different units.

        Parameters
        ----------
        other : |Quantity| or str
            Destination units.
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True).
        force : bool, optional, default=False
            If True the change of units is forced, even for incompatible units.

        Returns
        -------
        rescaled

        See Also
        --------
        ito : Inplace rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of the current object data to different units.
        to_reduced_units : Rescaling to reduced_units.
        ito_reduced_units : Inplace rescaling to reduced units.

        Examples
        --------
        >>> np.random.seed(12345)
        >>> ndd = scp.NDArray(data=np.random.random((3, 3)),
        ...                   mask=[[True, False, False],
        ...                         [False, True, False],
        ...                         [False, False, True]],
        ...                   units='meters')
        >>> print(ndd)
        NDArray: [float64] m (shape: (y:3, x:3))

        We want to change the units to seconds for instance
        but there is no relation with meters,
        so an error is generated during the change

        >>> ndd.to('second')
        Traceback (most recent call last):
        ...
        pint.errors.DimensionalityError: Cannot convert from 'meter' ([length]) to 'second' ([time])

        However, we can force the change

        >>> ndd.to('second', force=True)
        NDArray: [float64] s (shape: (y:3, x:3))

        By default the conversion is not done inplace, so the original is not
        modified :

        >>> print(ndd)
        NDArray: [float64] m (shape: (y:3, x:3))
        """

        new = self.copy()

        if other is None:
            units = None
            if self.units is None:
                return new
            elif force:
                new._units = None
                if inplace:
                    self._units = None
                return new

        elif isinstance(other, str):
            units = ur.Unit(other)

        elif hasattr(other, "units"):
            units = other.units

        else:
            units = ur.Unit(other)

        if self.has_units:

            oldunits = self._units

            def _transform(new):
                # not a linear transform
                udata = (new.data * new.units).to(units)
                new._data = udata.m
                new._units = udata.units

                if new._roi is not None:
                    roi = (np.array(new._roi) * self.units).to(units)
                    new._roi = list(roi)
                if new._linear:
                    # try to make it linear as well
                    new._linearize()
                    if not new._linear and new.implements("LinearCoord"):
                        # can't be linearized -> Coord
                        if inplace:
                            raise Exception(
                                "A LinearCoord object cannot be transformed to a non linear coordinate `inplace`. "
                                "Use to() instead of ito() and leave the `inplace` attribute to False"
                            )
                        else:
                            from spectrochempy import Coord

                            new = Coord(new)
                return new

            try:
                if new.meta.larmor:  # _origin in ['topspin', 'nmr']
                    set_nmr_context(new.meta.larmor)
                    with ur.context("nmr"):
                        new = _transform(new)

                # particular case of dimensionless units: absorbance and transmittance
                if str(oldunits) in ["transmittance", "absolute_transmittance"]:
                    if str(units) == "absorbance":
                        udata = (new.data * new.units).to(units)
                        new._data = -np.log10(udata.m)
                        new._units = units
                        if new.title.lower() == "transmittance":
                            new._title = "absorbance"

                elif str(oldunits) == "absorbance":
                    if str(units) in ["transmittance", "absolute_transmittance"]:
                        scale = Quantity(1.0, self._units).to(units).magnitude
                        new._data = 10.0 ** -new.data * scale
                        new._units = units
                        if new.title.lower() == "absorbance":
                            new._title = "transmittance"
                else:
                    # change the title for spectrocopic units change
                    new = _transform(new)
                    if (
                        oldunits.dimensionality
                        in [
                            "1/[length]",
                            "[length]",
                            "[length] ** 2 * [mass] / [time] ** 2",
                        ]
                        and new._units.dimensionality == "1/[time]"
                    ):
                        new._title = "frequency"
                    elif (
                        oldunits.dimensionality
                        in ["1/[time]", "[length] ** 2 * [mass] / [time] ** 2"]
                        and new._units.dimensionality == "1/[length]"
                    ):
                        new._title = "wavenumber"
                    elif (
                        oldunits.dimensionality
                        in [
                            "1/[time]",
                            "1/[length]",
                            "[length] ** 2 * [mass] / [time] ** 2",
                        ]
                        and new._units.dimensionality == "[length]"
                    ):
                        new._title = "wavelength"
                    elif (
                        oldunits.dimensionality
                        in ["1/[time]", "1/[length]", "[length]"]
                        and new._units.dimensionality == "[length] ** 2 * "
                        "[mass] / [time] "
                        "** 2"
                    ):
                        new._title = "energy"

            except DimensionalityError as exc:
                if force:
                    new._units = units
                    info_("units forced to change")
                else:
                    raise exc
        else:
            if force:
                new._units = units
            else:
                warnings.warn(
                    "There is no units for this NDArray!", SpectroChemPyWarning
                )

        if inplace:
            self._data = new._data
            self._units = new._units
            self._offset = new._offset
            self._increment = new._increment
            self._title = new._title
            self._roi = new._roi
            self._linear = new._linear

        else:
            return new

    def to_base_units(self, inplace=False):
        """
        Return an array rescaled to base units.

        Parameters
        ----------
        inplace : bool
            If True the rescaling is done in place.

        Returns
        -------
        rescaled
            A rescaled array
        """
        q = Quantity(1.0, self.units)
        q.ito_base_units()

        if not inplace:
            new = self.copy()
        else:
            new = self

        new.ito(q.units)

        if not inplace:
            return new

    def to_reduced_units(self, inplace=False):
        """
        Return an array scaled in place to reduced units.

        Reduced units means one unit per
        dimension. This will not reduce compound units (e.g., 'J/kg' will not
        be reduced to m**2/s**2).

        Parameters
        ----------
        inplace : bool
            If True the rescaling is done in place.

        Returns
        -------
        rescaled
            A rescaled array.
        """
        q = Quantity(1.0, self.units)
        q.ito_reduced_units()

        if not inplace:
            new = self.copy()
        else:
            new = self

        new.ito(q.units)

        if not inplace:
            return new

    # ..................................................................................................................
    def transpose(self, *dims, inplace=False):
        """
        Permute the dimensions of an array.

        Parameters
        ----------
        *dims : list int or str
            Sequence of dimension indexes or names, optional.
            By default, reverse the dimensions, otherwise permute the dimensions
            according to the values given. If specified the list of dimension
            index or names must match the number of dimensions.
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True)

        Returns
        -------
        transpose
            A transposed array

        See Also
        --------
        swapdims : Interchange two dimensions of an array.
        """
        if not inplace:
            new = self.copy()
        else:
            new = self
        if self.ndim < 2:  # cannot transpose 1D data
            return new
        if not dims or list(set(dims)) == [None]:
            dims = self.dims[::-1]
        axis = self._get_dims_index(dims)

        new._data = np.transpose(new._data, axis)
        if new.is_masked:
            new._mask = np.transpose(new._mask, axis)
        new._meta = new._meta.permute(*axis, inplace=False)

        new._dims = list(np.take(self._dims, axis))

        new._transposed = not new._transposed  # change the transposed flag
        return new

    @property
    def transposed(self):
        return self._transposed

    # ..................................................................................................................
    @property
    def umasked_data(self):
        """
        |Quantity| - The actual array with mask and unit

        (Readonly property).
        """
        if self.data is None:
            return None
        return self._uarray(self.masked_data, self.units)

    # ..................................................................................................................
    @property
    def unitless(self):
        """
        `bool` - True if the `data` does not have `units` (Readonly property).
        """
        return not self.has_units

    # ..................................................................................................................
    @property
    def units(self):
        """
        |Unit| - The units of the data.
        """
        return self._units

    # ..................................................................................................................
    @units.setter
    def units(self, units):

        if units is None:
            return
        if isinstance(units, str):
            units = ur.Unit(units)
        elif isinstance(units, Quantity):
            raise TypeError(
                "Units or string representation of unit is expected, not Quantity"
            )
        if self.has_units and units != self._units:
            # first try to cast
            try:
                self.to(units)
            except Exception:
                raise TypeError(
                    f"Provided units {units} does not match data units: {self._units}.\nTo force a change,"
                    f" use the to() method, with force flag set to True"
                )
        self._units = units

    # ..................................................................................................................
    @property
    def values(self):
        """
        |Quantity| - The actual values (data, units) contained in this object (Readonly property).
        """

        if self.data is not None:
            if self.is_masked:
                data = self._umasked(self.masked_data, self.mask)
                if self.units:
                    return Quantity(data, self.units)
            else:
                data = self._uarray(self.data, self.units)
            if self.size > 1:
                return data
            else:
                return data.squeeze()[()]
        elif self.is_labeled:
            return self._labels[()]

    @property
    def value(self):
        """
        Alias of `values`.
        """
        return self.values
