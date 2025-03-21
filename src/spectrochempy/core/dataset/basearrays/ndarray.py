# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module that implements the `NDArray` base class."""

import copy as cpy
import itertools
import re
import textwrap
import uuid
import warnings
from datetime import datetime

import numpy as np
import traitlets as tr

from spectrochempy.application.application import error_
from spectrochempy.application.application import info_
from spectrochempy.core.units import DimensionalityError
from spectrochempy.core.units import Quantity
from spectrochempy.core.units import Unit
from spectrochempy.core.units import set_nmr_context
from spectrochempy.core.units import ur
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.constants import DEFAULT_DIM_NAME
from spectrochempy.utils.constants import INPLACE
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.constants import NOMASK
from spectrochempy.utils.constants import TYPE_FLOAT
from spectrochempy.utils.constants import TYPE_INTEGER
from spectrochempy.utils.constants import MaskedConstant
from spectrochempy.utils.docutils import docprocess
from spectrochempy.utils.meta import Meta
from spectrochempy.utils.objects import make_new_object
from spectrochempy.utils.print import convert_to_html
from spectrochempy.utils.print import insert_masked_print
from spectrochempy.utils.print import numpyprintoptions
from spectrochempy.utils.typeutils import is_number
from spectrochempy.utils.typeutils import is_sequence

# Printing settings
# --------------------------------------------------------------------------------------
numpyprintoptions()


# ======================================================================================
# The basic NDArray class
# ======================================================================================
class NDArray(tr.HasTraits):
    r"""
    The basic  `NDArray` object.

    The  `NDArray` class is an array (numpy :term:`array-like`) container, usually not
    intended to be used directly,
    as its basic functionalities may be quite limited, but to be subclassed.

    Indeed, both the classes `NDDataset` and  `Coord` which respectively implement a
    full dataset (with
    coordinates) and the coordinates in a given dimension, are derived from  `NDArray`
    in  `SpectroChemPy` .

    The key distinction from raw numpy `~numpy.ndarray` is the presence of optional
    properties such as dimension names,
    labels, masks, units and/or extensible metadata dictionary.

    Parameters
    ----------
    data : array of floats
        Data array contained in the object. The data can be a list, a tuple, a
        `~numpy.ndarray` , a ndarray-like,
        a  `NDArray` or any subclass of  `NDArray` . Any size or shape of data is
        accepted. If not given, an empty `NDArray` will be inited.
        At the initialisation the provided data will be eventually cast to a
        `numpy.ndarray`.
        If a subclass of  `NDArray` is passed which already contains some mask, labels,
        or units, these elements
        will
        be used to accordingly set those of the created object. If possible, the
        provided data will not be copied
        for `data` input, but will be passed by reference, so you should make a copy of
        the `data` before passing
        them if that's the desired behavior or set the `copy` argument to True.
    **kwargs
        Optional keywords parameters. See Other Parameters.

    Other Parameters
    ----------------
    dtype : str or dtype, optional, default=np.float64
        If specified, the data will be cast to this dtype, else the data will be cast
        to float64.
    dims : list of chars, optional
        If specified the list must have a length equal to the number of data dimensions
        (ndim).
        If not specified, dimension names are automatically attributed in the order
        given by
        `DEFAULT_DIM_NAME` .
    name : str, optional
        A user-friendly name for this object. If not given, the automatic `id` given at
        the object creation will be
        used as a name.
    labels : array of objects, optional
        Labels for the `data` . Note that the labels can be used only for 1D-datasets.
        The labels array may have an additional dimension, meaning several series of
        labels for the same data.
        The given array can be a list, a tuple, a `~numpy.ndarray` , a ndarray-like, a
        `NDArray` or any subclass of `NDArray` .
    mask : array of bool or `NOMASK` , optional
        Mask for the data. The mask array must have the same shape as the data. The
        given array can be a list,
        a tuple, or a `~numpy.ndarray` . Each values in the array must be `False` where
        the data are *valid* and True when
        they are not (like in numpy masked arrays). If `data` is already a
        :class:`~numpy.ma.MaskedArray` , or any
        array object (such as a  `NDArray` or subclass of it), providing a `mask` here
        will cause the mask from the
        masked array to be ignored.
    units : `Unit` instance or str, optional
        Units of the data. If data is a `Quantity` then `units` is set to the unit of
        the `data`; if a unit is also
        explicitly provided an error is raised. Handling of units use the
        `pint <https://pint.readthedocs.org/>`_
        package.
    title : str, optional
        The title of the dimension. It will later be used for instance for labelling
        plots of the data.
        It is optional but recommended giving a title to each ndarray.
    dlabel :  str, optional
        Alias of `title` .
    meta : dict-like object, optional
        Additional metadata for this object. Must be dict-like but no
        further restriction is placed on meta.
    copy : bool, optional, Default:False
        If True, a deep copy of the passed object is performed.

    See Also
    --------
    NDDataset : Object which subclass  `NDArray` with the addition of coordinates.
    Coord : Object which subclass  `NDArray` (coordinates object).

    Examples
    --------
    >>> myarray = scp.NDArray([1., 2., 3.], name='myarray')
    >>> assert myarray.name == 'myarray'

    """

    # Hidden properties

    # Main array properties
    _id = tr.Unicode()
    _name = tr.Unicode()
    _title = tr.Unicode(allow_none=True)
    _data = Array(None, allow_none=True)
    _dtype = tr.Instance(np.dtype, allow_none=True)
    _dims = tr.List(tr.Unicode())
    _mask = tr.Union((tr.Bool(), Array(tr.Bool()), tr.Instance(MaskedConstant)))
    _labels = Array(allow_none=True)
    _units = tr.Instance(Unit, allow_none=True)
    _meta = tr.Instance(Meta, allow_none=True)

    # Region of interest
    _roi = tr.List(allow_none=True)

    # Transposition flag
    _transposed = tr.Bool(False)

    # Basic NDArray setting.
    # by default, we do shallow copy of the data
    # which means that if the same numpy array is used for too different NDArray,
    # they will share it.
    _copy = tr.Bool(False)

    _labels_allowed = tr.Bool(True)
    # Labels are allowed for the data, if the data are 1D only
    # they will essentially serve as coordinates labelling.

    # Other settings
    _text_width = tr.Integer(120)
    _html_output = tr.Bool(False)

    def __init__(self, data=None, **kwargs):
        super().__init__()

        # By default, we try to keep a reference to the data, so we do not copy them.
        self._copy = kwargs.pop("copy", False)  #

        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            self._dtype = np.dtype(dtype)

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

        if "dims" in kwargs:
            self.dims = kwargs.pop("dims")

        self.units = kwargs.pop("units", None)

        self.meta = (
            kwargs.pop("meta", {}) if isinstance(kwargs.get("meta", {}), dict) else {}
        )

        self.name = kwargs.pop("name", None)

    def _attributes_(self):
        return [
            "data",
            "dims",
            "mask",
            "labels",
            "units",
            "meta",
            "title",
            "name",
            "roi",
            "transposed",
        ]

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------
    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None):
        return self.copy(deep=True, memo=memo)

    def __eq__(self, other, attrs=None):
        eq = True

        if not isinstance(other, NDArray):
            # try to make some assumption to make useful comparison.
            if isinstance(other, Quantity):
                otherdata = other.magnitude
                otherunits = other.units
            elif isinstance(other, float | int | np.ndarray):
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
            attrs = self._attributes_()

        for attr in ["name", "linear", "roi"]:
            if attr in attrs:
                attrs.remove(attr)

        for attr in attrs:
            if attr != "units":
                sattr = getattr(self, f"_{attr}")
                if hasattr(other, f"_{attr}"):
                    oattr = getattr(other, f"_{attr}")
                    # to avoid deprecation warning issue for unequal array
                    if sattr is None and oattr is not None:  # pragma: no cover
                        return False
                    if oattr is None and sattr is not None:  # pragma: no cover
                        return False
                    # noinspection PyUnresolvedReferences
                    if (
                        hasattr(oattr, "size")
                        and hasattr(sattr, "size")
                        and oattr.size != sattr.size
                    ):
                        # particular case of mask
                        if attr != "mask":
                            return False
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

    def __getitem__(self, items, return_index=False):
        if isinstance(items, list):
            # Special case of fancy indexing
            items = (items,)

        # # allow passing a quantity as indice or in slices
        # def remove_units(items):
        #     # recursive function
        #     if isinstance(items, (tuple,)):
        #         items = tuple(remove_units(item) for item in items)
        #
        #     elif isinstance(items, slice):
        #         items = slice(
        #             remove_units(items.start),
        #             remove_units(items.stop),
        #             remove_units(items.step),
        #         )
        #
        #     else:
        #         items = float(items.m) if isinstance(items, Quantity) else items
        #
        #     return items
        #
        # items = remove_units(items)

        # choose, if we keep the same or create new object
        inplace = False
        if isinstance(items, tuple) and np.any(str(items[-1]) == INPLACE):
            items = items[:-1]
            inplace = True

        # Eventually get a better representation of the indexes
        try:
            keys = self._make_index(items)
        except IndexError as e:
            if "Could not find this location" in str(e) and self._squeeze_ndim == 1:
                # Try to use the next dimension if items is not already a tuple
                if not isinstance(items, tuple):
                    new_items = (slice(None), items)
                    keys = self._make_index(new_items)
                else:
                    raise e
            else:
                raise e

        # init returned object
        new = self if inplace else self.copy()

        # slicing by index of all internal array
        if new._data is not None:
            udata = new.masked_data[keys]
            new._data = np.asarray(udata)

        if self.is_labeled:
            # case only of 1D dataset such as Coord
            new._labels = np.array(self._labels[keys])

        if new.is_empty:
            error_(
                IndexError,
                f"Empty array of shape {new._data.shape} resulted from slicing.\n"
                f"Check the indexes and make sure to use floats for location slicing",
            )
            new = None

        elif (self.data is not None) and hasattr(udata, "mask"):
            new._mask = udata.mask
        else:
            new._mask = NOMASK

        # for all other cases,
        # we do not need to take care of dims, as the shape is not reduced by
        # this operation. Only a subsequent squeeze operation will do it
        if not return_index:
            return new
        return new, keys

    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash((type(self), self.shape, self._units))

    def __iter__(self):
        # iterate on the first dimension
        if self.ndim == 0:
            error_(ValueError, "iteration over a 0-d array is not possible")
            return None

        for n in range(len(self)):
            yield self[n]

    def __len__(self):
        # len of the last dimension
        if not self.is_empty:
            return self.shape[0]
        return 0

    def __ne__(self, other, attrs=None):
        return not self.__eq__(other, attrs)

    def __repr__(self):
        out = f"{self._repr_value().strip()} ({self._repr_shape().strip()})"
        return out.rstrip()

    def __setitem__(self, items, value):
        keys = self._make_index(items)

        if isinstance(value, bool | np.bool_ | MaskedConstant):
            # the mask is modified, not the data
            if value is MASKED:
                value = True
            if not np.any(self._mask):
                self._mask = np.zeros(self._data.shape).astype(np.bool_)
            if isinstance(keys, np.ndarray):
                self._mask[keys] = value
            else:
                # Note that we can't mask single element, but row and/or column
                for i, key in enumerate(keys):
                    if key == slice(None) or key == slice(None, None, 1):
                        continue
                    m = self._mask.swapaxes(i, 0)
                    m[key] = value
                    self._mask = m.swapaxes(i, 0)
            return self._mask

        if isinstance(value, Quantity):
            # first convert value to the current units
            value.ito(self.units)
            value = np.asarray(value.magnitude)  # , copy=self.copy)

        if self._data.dtype == np.dtype(np.quaternion) and np.isscalar(value):
            # sometimes do not work directly : here is a work around
            self._data[keys] = np.full_like(self._data[keys], value).astype(
                np.dtype(np.quaternion),
            )
            return None
        self._data[keys] = value
        return None

    def __str__(self):
        return repr(self)

    # ----------------------------------------------------------------------------------
    # Private methods and properties
    # ----------------------------------------------------------------------------------
    def _argsort(self, by=None, pos=None, descend=False):
        # found the indices sorted by values or labels

        if by is None:
            warnings.warn(
                "parameter `by` should be set to `value` or `label` , use `value` "
                "by default",
                stacklevel=2,
            )
            by = "value"

        if by == "value":
            args = np.argsort(self.data)
        elif "label" in by and not self.is_labeled:
            # by = 'value'
            # pos = None
            warnings.warn("no label to sort, use `value` by default", stacklevel=2)
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
                if pos >= self._labels.shape[-1]:
                    pos = self._labels.shape[-1] - 1  # Adjust pos to be within bounds
                labels = self._labels[..., pos]
            args = np.argsort(labels)

        if descend:
            args = args[::-1]
        return args

    def _cstr(self):
        out = f"{self._str_value()}\n{self._str_shape()}"
        return out.rstrip()

    @tr.default("_data")
    def _data_default(self):
        return None

    @tr.validate("_data")
    def _data_validate(self, proposal):
        # validation of the _data attribute
        data = proposal["value"]

        # cast to the desired type
        if self._dtype is not None:
            data = data.astype(np.dtype(self._dtype, copy=False))

        # return the validated data
        if self._copy:
            return data.copy()
        return data

    @tr.default("_dims")
    def _dims_default(self):
        return DEFAULT_DIM_NAME[-self.ndim :]

    def _get_dims_from_args(self, *dims, **kwargs):
        # utility function to read dims args and kwargs
        # sequence of dims or axis, or `dim` , `dims` or `axis` keyword are accepted

        # check if we have arguments
        if not dims:
            dims = None

        # Check if keyword dims (or synonym axis) exists
        axis = kwargs.pop("axis", None)

        kdims = kwargs.pop("dims", kwargs.pop("dim", axis))  # dim or dims keyword
        if kdims is not None:
            if dims is not None:
                warnings.warn(
                    "the unnamed arguments are interpreted as `dims` . But a named "
                    "argument `dims` or `axis`"
                    "(DEPRECATED) has been specified. \nThe unnamed arguments will "
                    "thus be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            dims = kdims

        if dims is not None and not isinstance(dims, list):
            dims = list(dims) if isinstance(dims, tuple) else [dims]

        if dims is not None:
            for i, item in enumerate(dims[:]):
                if item is not None and not isinstance(item, str):
                    item = self.dims[item]
                dims[i] = item

        if dims is not None and len(dims) == 1:
            dims = dims[0]

        return dims

    def _get_dims_index(self, dims):
        # get the index(es) corresponding to the given dim(s) which can be expressed
        # as integer or string

        if dims is None:
            return None

        if is_sequence(dims):
            if np.all([d is None for d in dims]):
                return None
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
                        f"(not in the dimension's list: {self.dims}).",
                    )
                idx = self.dims.index(dim)
                axis.append(idx)

            else:
                raise TypeError(
                    f"Dimensions must be specified as string or integer index, but a "
                    f"value of type "
                    f"{type(dim)} has been passed (value: {dim})!",
                )

        for i, item in enumerate(axis):
            # convert to positive index
            if item < 0:
                axis[i] = self.ndim + item

        return tuple(axis)

    def _get_slice(self, key, dim):
        info = None

        # allow passing a quantity as indice or in slices
        def remove_units(items):
            # recursive function

            units = None

            if isinstance(items, tuple):
                items = tuple(remove_units(item) for item in items)

            elif isinstance(items, slice):
                start, units = remove_units(items.start)
                end, _ = remove_units(items.stop)
                step, _ = remove_units(items.step)
                items = slice(start, end, step)

            elif isinstance(items, Quantity):
                units = items.u
                items = float(items.m)

            return items, units

        key, units = remove_units(key)

        if not isinstance(key, slice):
            # integer or float
            start = key
            if not isinstance(key, TYPE_INTEGER):
                start = self._loc2index(key, dim, units=units)
                if isinstance(start, tuple):
                    start, info = start
                if start is None:
                    return slice(None)
            elif key < 0:  # reverse indexing
                axis, dim = self.get_axis(dim) if self.ndim > 1 else (0, dim)
                # get axis is not defined for Coord for example
                start = self.shape[axis] + key
            stop = start + 1  # in order to keep a non squeezed slice
            return slice(start, stop, 1)
        start, stop, step = key.start, key.stop, key.step
        if start is not None and not isinstance(start, TYPE_INTEGER):
            start = self._loc2index(start, dim, units=units)
            if isinstance(start, tuple):
                start, info = start
        if stop is not None and not isinstance(stop, TYPE_INTEGER):
            stop = self._loc2index(stop, dim, units=units)
            if isinstance(stop, tuple):
                stop, info = stop
            if start is not None and stop < start:
                start, stop = stop, start
            if stop != start:
                stop = stop + 1  # to include last loc or label index
        if step is not None and not isinstance(step, int | np.int_ | np.int64):
            raise NotImplementedError(
                "step in location slicing is not yet possible.",
            )  # TODO: we have may be a special case with
            # datetime  # step = 1
        if step is None and (start is not None or stop is not None):
            step = 1
        if start is not None and stop is not None and start == stop and info is None:
            stop = stop + 1  # to include last index

        return slice(start, stop, step)

    @tr.default("_id")
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    @tr.default("_labels")
    def _labels_default(self):
        return None

    def _loc2index(self, loc, dim=None, *, units=None):
        # Return the index of a location (label or values such as coordinates) along a
        # 1D array.
        # Do not apply for multidimensional arrays (ndim>1)
        if self.ndim > 1:
            raise NotImplementedError(
                f"not implemented for {type(self).__name__} objects which are "
                f"not 1-dimensional "
                f"(current ndim: {self.ndim})",
            )

        # check units compatibility
        if (
            units is not None
            and (is_number(loc) or is_sequence(loc))
            and units != self.units
        ):
            raise ValueError(
                f"Units of the location {loc} {units} are not compatible with those of "
                f"this array: "
                f"{self.units}",
            )

        if self.is_empty and not self.is_labeled:
            raise IndexError(f"Could not find this location {loc} on an empty array")

        data = self.data

        if is_number(loc):
            # get the index of a given values
            error = None
            if np.all(loc > data.max()) or np.all(loc < data.min()):
                info_(
                    f"This coordinate ({loc}) is outside the axis limits "
                    f"({data.min()}-{data.max()}).\n"
                    f"The closest limit index is returned",
                )
                error = "out_of_limits"
            index = (np.abs(data - loc)).argmin()
            # TODO: add some precision to this result
            if not error:
                return index
            return index, error

        if is_sequence(loc):
            # TODO: is there a simpler way to do this with numpy functions
            index = []
            for lo in loc:
                index.append(
                    (np.abs(data - lo)).argmin(),
                )  # TODO: add some precision to this result
            return index

        if self.is_labeled:
            # TODO: look in all row of labels
            labels = self._labels
            indexes = np.argwhere(labels == loc).flatten()
            if indexes.size > 0:
                return indexes[0]
            raise IndexError(f"Could not find this label: {loc}")

        if isinstance(loc, datetime):
            # not implemented yet
            raise NotImplementedError(
                "datetime as location is not yet implemented",
            )  # TODO: date!

        raise IndexError(f"Could not find this location: {loc}")

    def _make_index(self, key):
        if isinstance(key, np.ndarray) and key.dtype == bool:
            # this is a boolean selection
            # we can proceed directly
            return key

        # we need to have a list of slice for each argument
        # or a single slice acting on the axis=0
        keys = list(key) if isinstance(key, tuple) else [key]

        def ellipsisinkeys(_keys):
            try:
                # Ellipsis
                if isinstance(_keys[0], np.ndarray):
                    return False
                test = Ellipsis in _keys
            except ValueError as e:
                if e.args[0].startswith("The truth "):
                    # probably an array of index (Fancy indexing)
                    # should not happen anymore with the test above
                    test = False
            return test

        while ellipsisinkeys(keys):
            i = keys.index(Ellipsis)
            keys.pop(i)
            for _j in range(self.ndim - len(keys)):
                keys.insert(i, slice(None))

        if len(keys) > self.ndim:
            raise IndexError("invalid index")

        # pad the list with additional dimensions
        for _i in range(len(keys), self.ndim):
            keys.append(slice(None))

        for axis, key in enumerate(keys):
            # the keys are in the order of the dimension in self.dims!
            # so we need to get the correct dim in the coordinates lists
            dim = self.dims[axis]
            if is_sequence(key) and not isinstance(key, Quantity):
                # fancy indexing
                # all items of the sequence must be integer index
                if np.any([isinstance(k, TYPE_FLOAT) for k in key]):
                    key = [self._loc2index(k, dim) for k in key]
                keys[axis] = key
            else:
                keys[axis] = self._get_slice(key, dim)

        # when the array is one-dimensional (squeeze_ndim=1 and shape[0]==1)
        # then we can consider that slicing is to be performed on the significant
        # axis (axis=-1)
        if (
            self._squeeze_ndim == 1
            and self.ndim == 2
            and self.shape[0] == 1
            and keys[0] != slice(None)
            and keys[0] != slice(0, 1, 1)
            and keys[1] == slice(None)
        ):
            keys.reverse()  # WARNING; assume 2D
        return tuple(keys)

    @tr.default("_mask")
    def _mask_default(self):
        return NOMASK if self._data is None else np.zeros(self._data.shape).astype(bool)

    @tr.validate("_mask")
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
        return mask

    @tr.default("_meta")
    def _meta_default(self):
        return Meta()

    @tr.default("_name")
    def _name_default(self):
        return ""

    def _repr_html_(self):
        return convert_to_html(self)

    def _repr_shape(self):
        if self.is_empty:
            return "size: 0"

        out = ""

        shape_ = (
            x
            for x in itertools.chain.from_iterable(
                list(zip(self.dims, self.shape, strict=False)),
            )
        )

        shape = (", ".join(["{}:{}"] * self.ndim)).format(*shape_)

        size = self.size

        out += f"size: {size}" if self.ndim < 2 else f"shape: ({shape})"

        return out

    def _repr_units(self):
        if self.has_units:
            # if not self.units.dimensionless or self.units.scaling != 1:
            strunits = f"{self.units: ~P}"
            if strunits == "":
                strunits = "dimensionless"
        else:
            strunits = "unitless"
        return strunits

    def _repr_value(self):
        numpyprintoptions(precision=4, edgeitems=0, spc=1, linewidth=120)

        prefix = type(self).__name__ + ": "
        units = ""

        size = ""
        if not self.is_empty:
            if self.data is not None:
                dtype = self.dtype
                data = ""
                if self._implements("Coord"):
                    size = f" (size: {self.data.size})"
                units = " " + self._repr_units()
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

    @tr.default("_roi")
    def _roi_default(self):
        return None

    def _set_data(self, data):
        if data is None:
            return

        if isinstance(data, NDArray):
            # init data with data from another NDArray or NDArray's subclass
            # No need to check the validity of the data
            # because the data must have been already
            # successfully initialized for the passed NDArray.data
            for attr in self._attributes_():
                try:
                    val = getattr(data, f"_{attr}")
                    if self._copy:
                        val = cpy.deepcopy(val)
                    setattr(self, f"_{attr}", val)
                except AttributeError:
                    # some attribute of NDDataset are missing in NDArray
                    pass

        elif isinstance(data, Quantity):
            if self._copy:
                self._data = np.asarray(data.magnitude, subok=True, copy=True)
            else:
                self._data = np.asarray(data.magnitude)
            self._units = data.units

        elif hasattr(data, "mask"):
            # an object with data and mask attributes
            if self.copy:
                self._data = np.array(data.data, subok=True, copy=self._copy)
            else:
                self._data = np.asarray(data.data)
            if isinstance(data.mask, np.ndarray) and data.mask.shape == data.data.shape:
                self.mask = np.asarray(data.mask, dtype=np.bool_)

        elif (
            not hasattr(data, "shape")
            or not hasattr(data, "__getitem__")
            or not hasattr(data, "__array_struct__")
        ):
            # Data doesn't look like a numpy array, try converting it to
            # one.
            try:
                self._data = np.asarray(data)
            except ValueError:
                # happens if data is a list of quantities
                if isinstance(data[0], Quantity):
                    self._data = np.asarray([d.m for d in data])
                self._units = data[0].units

        else:
            data = np.array(data, subok=True, copy=self._copy)
            if data.dtype == np.object_:  # likely None value
                data = data.astype(float)
            self._data = data

    def _sort(self, by=None, pos=None, descend=False, inplace=False):
        # sort an ndarray using data or label values

        args = self._argsort(by, pos, descend)

        return self[args] if not inplace else self[args, INPLACE]

    @property
    def _squeeze_ndim(self):
        # The number of dimensions of the squeezed`data` array (Readonly property).

        if self.data is None and self.is_labeled:
            return 1
        if self.data is None:
            return 0
        return len([x for x in self.data.shape if x > 1])

    def _str_shape(self):
        if self.is_empty:
            return "         size: 0\n"

        out = ""

        shape_ = (
            x
            for x in itertools.chain.from_iterable(
                list(zip(self.dims, self.shape, strict=False)),
            )
        )

        shape = (", ".join(["{}:{}"] * self.ndim)).format(*shape_)

        size = self.size

        out += (
            f"         size: {size}\n"
            if self.ndim < 2
            else f"        shape: ({shape})\n"
        )

        return out

    def _str_value(self, sep="\n", ufmt=" {:~P}", header="         data: ... \n"):
        # prefix = ['']
        if self.is_empty and "data: ..." not in header:
            return header + "{}".format(textwrap.indent("empty", " " * 9))
        if self.is_empty:
            return "{}".format(textwrap.indent("empty", " " * 9))

        print_unit = True
        units = ""

        def mkbody(d, pref, _units):
            # work around printing masked values with formatting
            ds = d.copy()
            if self.is_masked:
                dtype = self.data.dtype
                mask_string = f"--{dtype}"
                ds = insert_masked_print(ds, mask_string=mask_string)
            _body = np.array2string(ds, separator=" ", prefix=pref)
            _body = _body.replace("\n", sep)
            _text = "".join([pref, _body, _units])
            _text += sep
            return _text

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
        return out.rstrip()  # remove the trailing '\n'

    @tr.default("_title")
    def _title_default(self):
        return None

    @staticmethod
    def _unittransform(new, units):
        oldunits = new.units
        udata = (new.data * oldunits).to(units)
        new._data = udata.m
        new._units = udata.units

        if new._roi is not None:
            roi = (np.array(new.roi) * oldunits).to(units)
            new._roi = list(roi)

        return new

    @staticmethod
    def _uarray(data, units=None):
        # return the array or scalar with units
        # if data.size==1:
        #    uar = data.squeeze()[()]
        # else:
        uar = data
        if units:
            return Quantity(uar, units)
        return uar

    @staticmethod
    def _umasked(data, mask):
        # This ensures that a masked array is returned.

        if not np.any(mask):
            mask = np.zeros(data.shape).astype(bool)
        return np.ma.masked_where(mask, data)  # np.ma.masked_array(data, mask)

    # ------------------------------------------------------------------------
    # Public Methods and Properties
    # ------------------------------------------------------------------------
    def asfortranarray(self):
        """Make data and mask (ndim >= 1) laid out in Fortran order in memory."""
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
            Typecode or data-type to which the array is cast.

        """
        self._data = self._data.astype(dtype, **kwargs)
        return self

    def copy(self, deep=True, keepname=False, **kwargs):
        """
        Make a disconnected copy of the current object.

        Parameters
        ----------
        deep : bool, optional
            If True a deepcopy is performed which is the default behavior.
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
        do_copy = cpy.deepcopy if deep else cpy.copy

        new = make_new_object(self)
        for attr in self._attributes_():
            try:
                _attr = do_copy(getattr(self, f"_{attr}"))
                setattr(new, f"_{attr}", _attr)

            except ValueError:
                # ensure that if deepcopy do not work, a shallow copy can be done
                _attr = do_copy(getattr(self, f"_{attr}"))
                setattr(new, f"_{attr}", _attr)

        # name must be changed
        if not keepname:
            new.name = ""  # default

        return new

    @property
    @docprocess.get_docstring(base="data")
    def data(self):
        """
        Data array (`~numpy.ndarray`).

        If there is no data but labels, then the labels are returned instead of data.
        """
        return self._data

    @data.setter
    def data(self, data):
        # property.setter for data
        # note that a subsequent validation is done in _data_validate
        # NOTE: as property setter doesn't work with super(),
        # see https://stackoverflow.com/questions/10810369/python-super-and-setting-parent-class-property
        # we use an intermediate function that can be called from a subclass

        self._set_data(data)

    magnitude = data
    m = data

    @property
    def dimensionless(self):
        """
        True if the `data` array is dimensionless - Readonly property (bool).

        See Also
        --------
        unitless : True if data has no units.
        has_units : True if data has units.

        Notes
        -----
        `Dimensionless` is different of `unitless` which means no unit.

        """
        if self.unitless:
            return False
        return self._units.dimensionless

    @property
    def dims(self):
        """
        Names of the dimensions (list).

        The name of the dimensions are 'x', 'y', 'z'....
        depending on the number of dimension.
        """
        ndim = self.ndim
        if ndim > 0:
            # if len(self._dims)< ndim:
            #    self._dims = self._dims_default()
            return self._dims[:ndim]
        return []

    @dims.setter
    def dims(self, values):
        if isinstance(values, str) and len(values) == 1:
            values = [values]

        if not is_sequence(values) or len(values) != self.ndim:
            raise ValueError(
                f"a sequence of chars with a length of {self.ndim} is expected, "
                f"but `{values}` "
                f"has been provided",
            )

        for value in values:
            if value not in DEFAULT_DIM_NAME:
                raise ValueError(
                    f"{value} value is not admitted. Dimension's name must be among "
                    f"{DEFAULT_DIM_NAME[::-1]}.",
                )

        self._dims = tuple(values)

    @property
    def dtype(self):
        """Return the data type."""
        if self.is_empty:
            return None
        return self._data.dtype

    def get_axis(self, *args, **kwargs):
        """
        Determine an axis index whatever the syntax used (axis index or dimension names).

        Parameters
        ----------
        dim, axis, dims : str, int, or list of str or index
            The axis indexes or dimensions names - they can be specified as argument or
            using keyword 'axis', 'dim'
            or 'dims'.
        negative_axis : bool, optional, default=False
            If True a negative index is returned for the axis value (-1 for the last
            dimension, etc...).
        allows_none : bool, optional, default=False
            If True, if input is none then None is returned.
        only_first : bool, optional, default: True
            By default return only information on the first axis if dim is a list.
            Else, return a list for axis and dims information.

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
            if kwargs.get("negative_axis", False) and a >= 0:
                a = a - self.ndim
            axis[i] = a
            dims[i] = self.dims[a]

        only_first = kwargs.pop("only_first", True)

        if len(dims) == 1 and only_first:
            dims = dims[0]
            axis = axis[0]

        return axis, dims

    def get_labels(self, level=0):
        """
        Get the labels at a given level.

        Used to replace `data` when only labels are provided, and/or for
        labeling axis in plots.

        Parameters
        ----------
        level : int, optional, default:0
            Label level.

        Returns
        -------
        `~numpy.ndarray`
            The labels at the desired level or None.

        """
        if not self.is_labeled:
            return None

        if level > self.labels.ndim - 1:
            warnings.warn("There is no such level in the existing labels", stacklevel=2)
            return None

        if self.labels.ndim > 1:
            return self.labels[level]
        return self._labels

    @property
    def has_data(self):
        """True if the `data` array is not empty."""
        return not (self._data is None or self._data.size == 0)

    @property
    def has_defined_name(self):
        """True is the name has been defined (bool)."""
        return self.name != self.id

    @property
    def has_units(self):
        """
        True if the `data` array have units - Readonly property (bool).

        See Also
        --------
        unitless : True if the data has no unit.
        dimensionless : True if the data have dimensionless units.

        """
        return self._units is not None

    @property
    def id(self):
        """Object identifier - Readonly property (str)."""
        return self._id

    @property
    def imag(self):
        return None

    def _implements(self, name=None):
        """
        Check if the current object implements a given class.

        Rather than isinstance(obj, <class>) use object._implements('<classname>').
        This is useful to check type without importing the module.

        Parameters
        ----------
        name : str, optional
            Name of the class implemented.

        Examples
        --------
        >>> ar = scp.NDArray([1., 2., 3.])
        >>> ar._implements('NDDataset')
        False
        >>> ar._implements('NDArray')
        True
        >>> ar._implements()
        'NDArray'

        """
        if name is None:
            return self.__class__.__name__
        return name == self.__class__.__name__

    @property
    def is_float(self):
        """True if the `data` are real values - Readonly property (bool)."""
        if self.data is None:
            return False

        return self.data.dtype in TYPE_FLOAT

    @property
    def is_integer(self):
        """True if the `data` are integer values - Readonly property (bool)."""
        if self.data is None:
            return False

        return self.data.dtype in TYPE_INTEGER

    @property
    def is_1d(self):
        """True if the `data` array has only one dimension (bool)."""
        return self.ndim == 1

    @property
    def is_empty(self):
        """
        True if the `data` array is empty or size=0, and if no label are present.

        Readonly property (bool).
        """
        return bool(
            (self._data is None or self._data.size == 0) and not self.is_labeled,
        )

    @property
    def is_labeled(self):
        """True if the `data` array have labels - Readonly property (bool)."""
        # label cannot exist for now for nD dataset - only 1D dataset, such
        # as Coord can be labelled.
        if self._data is not None and self.ndim > 1:
            return False
        return bool(self._labels is not None and np.any(self.labels != ""))

    @property
    def is_masked(self):
        """True if the `data` array has masked values - Readonly property (bool)."""
        if isinstance(self._mask, np.ndarray):
            return np.any(self._mask)
        if self._mask == NOMASK or self._mask is None:
            return False
        if isinstance(self._mask, np.bool_ | bool):
            return self._mask

        return False

    def is_units_compatible(self, other):
        """
        Check the compatibility of units with another object.

        Parameters
        ----------
        other : `~numpy.ndarray`
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

    def ito(self, other, force=False):
        """
        Inplace scaling to different units. (same as `to` with inplace= True).

        Parameters
        ----------
        other : `Unit` , `Quantity` or str
            Destination units.
        force : bool, optional, default=`False`
            If True the change of units is forced, even for incompatible units.

        See Also
        --------
        to : Rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of the current object data to different units.
        to_reduced_units : Rescaling to reduced units.
        ito_reduced_units : Rescaling to reduced units.

        """
        self.to(other, inplace=True, force=force)

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

    def ito_reduced_units(self):
        """
        Quantity scaled in place to reduced units, inplace.

        Scaling to reduced units means one unit per
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

    @property
    def labels(self):
        """
        An array of labels for `data` (`~numpy.ndarray` of str).

        An array of objects of any type (but most generally string), with the last
        dimension size equal to that of the
        dimension of data. Note that's labelling is possible only for 1D data. One
        classical application is
        the labelling of coordinates to display informative strings instead of
        numerical values.
        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        if labels is None:
            return

        if self.ndim > 1:
            warnings.warn(
                "We cannot set the labels for multidimentional data - Thus, these "
                "labels are ignored",
                stacklevel=2,
            )
        else:
            # make sure labels array is of type np.ndarray or Quantity arrays
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels, subok=True, copy=True).astype(
                    object,
                    copy=False,
                )

            if not np.any(labels):
                # no labels
                return

            if (self.data is not None) and (labels.shape[0] != self.shape[0]):
                # allow the fact that the labels may have been passed in a
                # transposed array
                if labels.ndim > 1 and (labels.shape[-1] == self.shape[0]):
                    labels = labels.T
                else:
                    raise ValueError(
                        f"labels {labels.shape} and data {self.shape} shape mismatch!",
                    )

            if np.any(self._labels):
                info_(
                    f"{type(self).__name__} is already a labeled array.\nThe "
                    f"explicitly provided labels will "
                    f"be appended to the current labels",
                )

                labels = labels.squeeze()
                self._labels = self._labels.squeeze()
                if self._labels.ndim > 1:
                    self._labels = self._labels.T
                self._labels = np.vstack((self._labels, labels)).T

            elif self._copy:
                self._labels = labels.copy()
            else:
                self._labels = labels

    @property
    def limits(self):
        """Range of the data (list)."""
        if self.data is None:
            return None

        return [self.data.min(), self.data.max()]

    @property
    def mask(self):
        """Mask for the data (`~numpy.ndarray` of bool)."""
        if not self.is_masked:
            return NOMASK

        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is NOMASK or mask is MASKED:
            pass
        elif isinstance(mask, np.bool_ | bool):
            mask = NOMASK if not mask else MASKED
        else:
            # from now, make sure mask is of type np.ndarray if it provided
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.bool_)
            if not np.any(mask):
                # all element of the mask are false
                mask = NOMASK
            elif mask.shape != self.shape:
                raise ValueError(
                    f"mask {mask.shape} and data {self.shape} shape mismatch!",
                )

        # finally, set the mask of the object
        if isinstance(mask, MaskedConstant):
            self._mask = (
                NOMASK if self.data is None else np.ones(self.shape).astype(bool)
            )
        elif np.any(self._mask):
            # this should happen when a new mask is added to an existing one
            # mask to be combined to an existing one
            info_(
                f"{type(self).__name__} is already a masked array.\n The new "
                f"mask will be combined with the "
                f"current array's mask.",
            )
            self._mask |= mask  # combine (is a copy!)
        elif self._copy:
            self._mask = mask.copy()
        else:
            self._mask = mask

    @property
    def masked_data(self):
        """The actual masked `data` array - Readonly property (`numpy.ma.ndarray`)."""
        if self.is_masked and not self.is_empty:
            return self._umasked(self.data, self.mask)
        return self.data

    @property
    def meta(self):
        """Additional metadata (`Meta`)."""
        return self._meta

    @meta.setter
    def meta(self, meta):
        if meta is not None:
            self._meta.update(meta)

    @property
    def name(self):
        """
        A user-friendly name (str).

        When no name is provided, the `id` of the object is returned instead.
        """
        if self._name:
            return self._name
        return self._id

    @name.setter
    def name(self, name):
        if name:
            if self._name:
                pass
            self._name = name

    @property
    def ndim(self):
        """The number of dimensions of the `data` array (Readonly property)."""
        if self.data is None and self.is_labeled:
            return 1

        if not self.size:
            return 0
        return self.data.ndim

    @property
    def real(self):
        return self

    def remove_masks(self):
        """Remove all masks previously set on this array."""
        self._mask = NOMASK

    @property
    def roi(self):
        """Region of interest (ROI) limits (list)."""
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
        return list(self._uarray(self.roi, self.units))

    @property
    def shape(self):
        """
        A tuple with the size of each dimension - Readonly property.

        The number of `data` element on each dimension (possibly complex).
        For only labelled array, there is no data, so it is the 1D and the size is the
        size of the array of labels.
        """
        if self.data is None and self.is_labeled:
            return (self.labels.shape[0],)

        if self.data is None:
            return ()

        return self.data.shape

    @property
    def size(self):
        """
        Size of the underlying `data` array - Readonly property (int).

        The total number of data elements (possibly complex or hypercomplex
        in the array).
        """
        if self._data is None and self.is_labeled:
            return self.labels.shape[-1]

        if self._data is None:
            return None
        return self._data.size

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
            If `dims` is not `None` , and the dimension being squeezed is not
            of length 1

        """
        new = self if inplace else self.copy()

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

    def swapdims(self, dim1, dim2, inplace=False):
        """
        Interchange a two dims of a  `NDArray` .

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
        transpose: Permute the dimensions of an array.

        """
        new = self.copy() if not inplace else self
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
    swapaxes.__doc__ = "Alias of `swapdims` ."

    @property
    def T(self):
        """
        Transposed array ( `NDArray`).

        The same object is returned if `ndim` is less than 2.

        See Also
        --------
        transpose: Permute the dimensions of an array.

        """
        return self.transpose()

    @property
    def title(self):
        """
        An user-friendly title (str).

        When the title is provided, it can be used for labeling the object,
        e.g., axe title in a matplotlib plot.
        """
        if self._title:
            return self._title
        return "<untitled>"

    @title.setter
    def title(self, title):
        if title:
            self._title = title

    def to(self, other, inplace=False, force=False):
        """
        Return the object with data rescaled to different units.

        Parameters
        ----------
        other : `Quantity` or str
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
        pint.errors.DimensionalityError: Cannot convert from 'meter' ([length]) to
        'second' ([time])

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
            if force:
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

            try:
                # noinspection PyUnresolvedReferences
                if new._implements("Coord") and new.larmor:
                    set_nmr_context(new.larmor)
                    with ur.context("nmr"):
                        new = self._unittransform(new, units)

                # particular case of dimensionless units: absorbance and transmittance
                elif f"{oldunits: P}" in ["transmittance", "absolute_transmittance"]:
                    if f"{units: P}" == "absorbance":
                        udata = (new.data * new.units).to(units)
                        new._data = -np.log10(udata.m)
                        new._units = units
                        new._title = "absorbance"

                    elif f"{units: P}" in [
                        "transmittance",
                        "absolute_transmittance",
                    ]:
                        new._data = (new.data * new.units).to(units)
                        new._units = units
                        new._title = "transmittance"

                elif f"{oldunits: P}" == "absorbance":
                    if f"{units: P}" in ["transmittance", "absolute_transmittance"]:
                        scale = Quantity(1.0, self._units).to(units).magnitude
                        new._data = 10.0**-new.data * scale
                        new._units = units
                        new._title = "transmittance"
                else:
                    new = self._unittransform(new, units)
                    # change the title for spectrocopic units change
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
                        and new._units.dimensionality
                        == "[length] ** 2 * [mass] / [time] ** 2"
                    ):
                        new._title = "energy"

                if force:
                    new._units = units

            except DimensionalityError as exc:
                if force:
                    new._units = units
                    info_("units forced to change")
                else:
                    raise exc
        elif force:
            new._units = units

        else:
            warnings.warn("There is no units for this NDArray!", stacklevel=2)

        if inplace:
            self._data = new._data
            self._units = new._units
            self._title = new._title
            self._roi = new._roi
            return None

        return new

    docprocess.get_docstring(to.__doc__, base="to")

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
            A rescaled array.

        """
        q = Quantity(1.0, self.units)
        q.ito_base_units()

        new = self.copy() if not inplace else self

        new.ito(q.units)

        if not inplace:
            return new
        return None

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

        new = self.copy() if not inplace else self

        new.ito(q.units)

        if not inplace:
            return new
        return None

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
        new = self.copy() if not inplace else self
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

    @property
    def umasked_data(self):
        """
        The actual array with mask and unit (`Quantity`).

        (Readonly property).
        """
        if self.data is None:
            return None
        return self._uarray(self.masked_data, self.units)

    @property
    def unitless(self):
        """`bool` - True if the `data` does not have `units` (Readonly property)."""
        return not self.has_units

    @property
    def units(self):
        """`Unit` - The units of the data."""
        return self._units

    @units.setter
    def units(self, units):
        if units is None:
            return
        if isinstance(units, str):
            units = ur.Unit(units)
        elif isinstance(units, Quantity):
            raise TypeError(
                "Units or string representation of unit is expected, not Quantity",
            )
        if self.has_units and units != self._units:
            # first try to cast
            try:
                self.to(units)
            except Exception as e:
                raise TypeError(
                    f"Provided units {units} does not match data units: {self._units}.\nTo force a change, "
                    f"use the to() method, with force flag set to True",
                ) from e
        self._units = units

    @property
    def values(self):
        """`Quantity` - The actual values (data, units) contained in this object (Readonly property)."""
        if self.data is not None:
            if self.is_masked:
                data = self._umasked(self.masked_data, self.mask)
                if self.units:
                    return Quantity(data, self.units)
            else:
                data = self._uarray(self.data, self.units)
            if self.size > 1:
                return data
            return data.squeeze()[()]
        if self.is_labeled:
            return self._labels[()]
        return None

    @property
    def value(self):
        """Alias of `values` ."""
        return self.values
