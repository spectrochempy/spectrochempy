# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the class  `Coord` .
"""

__all__ = ["Coord"]

import textwrap

import numpy as np
import traitlets as tr

from spectrochempy.application import error_
from spectrochempy.core.dataset.arraymixins.ndmath import NDMath, _set_operators
from spectrochempy.core.dataset.baseobjects.ndarray import NDArray
from spectrochempy.core.units import Quantity, ur
from spectrochempy.utils.compare import is_iterable, is_number
from spectrochempy.utils.constants import INPLACE, NOMASK
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.numutils import get_n_decimals, spacings
from spectrochempy.utils.print import colored_output


# ======================================================================================
# Coord
# ======================================================================================
@tr.signature_has_traits
class Coord(NDMath, NDArray):
    """
    Explicit coordinates for a dataset along a given axis.

    The coordinates of a `NDDataset` can be created using the  `Coord`
    object.
    This is a single dimension array with either numerical (float)
    values or labels (str, `Datetime` objects, or any other kind of objects) to
    represent the coordinates. Only a one numerical axis can be defined,
    but labels can be multiple.

    Parameters
    ----------
    data : ndarray, tuple or list
        The actual data array contained in the  `Coord` object.
        The given array (with a single dimension) can be a list,
        a tuple, a `~numpy.ndarray` , or a :term:`array-like` object.
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
        Labels for the `data` . labels can be used only for 1D-datasets.
        The labels array may have an additional dimension, meaning
        several series of labels for the same data.
        The given array can be a list, a tuple, a `~numpy.ndarray` ,
        a ndarray-like, a  `NDArray` or any subclass of `NDArray` .
    units : `Unit` instance or str, optional
        Units of the data. If data is a `Quantity` then `units` is set
        to the unit of the `data`; if a unit is also
        explicitly provided an error is raised. Handling of units use
        the `pint <https://pint.readthedocs.org/>`_
        package.
    title : str, optional
        The title of the dimension. It will later be used for instance
        for labelling plots of the data.
        It is optional but recommended to give a title to each ndarray.
    dlabel :  str, optional
        Alias of `title` .
    linearize_below : float, optional, default=0.1
        variation of spacing in % below which the coordinate is linearized. Set it to
    rounding : bool, optional, default=True
        If True, the data will be rounded to the number of significant
        digits given by `sigdigits`\ .
    sigdigits : int, optional, default=4
        Number of significant digits to be used for rounding and linearizing
        the data.
    larmor : `float` or `Quantity` instance, optional
        The Larmor frequency of the nucleus. This is used only for NMR
        data.
    offset : `float` instance, optional
        The offset of the axis. This is used to generate an evenly values spaced axis
        together with `ìncrement` and `size`\ .
    increment : `float` instance, optional
        The increment between two consecutive values of the axis. This is used to
        generate an evenly values spaced axis together with `offset` and `size`\ .
    size : `int` instance, optional
        The size of the axis. This is used to generate an evenly values spaced axis
        together with `offset` and `increment`\ .

    See Also
    --------
    NDDataset : Main SpectroChemPy object: an array with masks, units and coordinates.

    Examples
    --------

    We first import the object from the api :

    >>> from spectrochempy import Coord

    We then create a numpy `~numpy.ndarray` and use it as the numerical `data`
    axis of our new  `Coord` object :

    >>> c0 = Coord.arange(1., 12., 2., title='frequency', units='Hz')
    >>> c0
    Coord: [float64] Hz (size: 6)

    We can take a series of str to create a non-numerical but labelled
    axis :

    >>> tarr = list('abcdef')
    >>> tarr
    ['a', 'b', 'c', 'd', 'e', 'f']

    >>> c1 = Coord(labels=tarr, title='mylabels')
    >>> c1
    Coord: [labels] [  a   b   c   d   e   f] (size: 6)
    """

    _copy = tr.Bool()

    _html_output = tr.Bool(False)
    _parent_dim = tr.Unicode(allow_none=True)
    _parent = tr.Instance(
        "spectrochempy.core.dataset.nddataset.NDDataset", allow_none=True
    )
    _use_time = tr.Bool(False)
    _show_datapoints = tr.Bool(True)
    _zpd = tr.Integer()

    _linearize_below = tr.Float(0.1)
    _linear = tr.Bool(False)
    _sigdigits = tr.Int(4)
    _rounding = tr.Bool(True)

    # specific to NMR
    _larmor = tr.Instance(Quantity, allow_none=True)

    # ----------------------------------------------------------------------------------
    # initialization
    # ----------------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):
        # check if data is iterable
        if data is not None and not is_iterable(data):
            raise ValueError("Data for coordinates must be an iterable or None")

        # in case Coord replace old LinearCoord object
        # without changing the arguments
        _offset = kwargs.pop("offset", 0)
        _increment = kwargs.pop("increment", None)
        _size = kwargs.pop("size", None)

        if data is None and _size is not None and _increment is not None:
            data = np.arange(_size) * _increment + _offset

        # specific case of NMR (initialize unit context NMR)
        larmor = kwargs.pop("larmor", None)

        self._linearize_below = kwargs.pop("linearize_below", 1.0)

        # extract parameters for linearization and data rounding
        self._sigdigits = kwargs.pop("sigdigits", 4)

        # if data is a Coord, rounding may have been set already
        if isinstance(data, Coord):
            self._rounding = data._rounding
        else:
            self._rounding = kwargs.pop("rounding", True)  # rounding of data by default

        # initialize the object
        super().__init__(data=data, **kwargs)

        # set the larmor frequency if any
        if larmor is not None:
            self.larmor = larmor

    # ----------------------------------------------------------------------------------
    # default values
    # ----------------------------------------------------------------------------------
    @tr.default("_larmor")
    def _default_larmor(self):
        return None

    # ----------------------------------------------------------------------------------
    # readonly property
    # ----------------------------------------------------------------------------------
    @property
    def reversed(self):
        """Whether the axis is reversed."""
        if self.units == "ppm":
            return True
        elif self.units == "1 / centimeter" and "raman" not in self.title.lower():
            return True
        return False

        # Return a correct result only if the data are sorted  # return  # bool(self.data[0] > self.data[-1])

    @property
    @_docstring.dedent
    def data(self):
        """%(data)s

        Notes
        -----
        The data are always returned as a 1D array of float rounded to the number
        of significant digits given by the `sigdigits` parameters.
        If the spacing between the data is constant with the accuracy given by the
        significant digits, the data are thus linearized
        and the `linear` attribute is set to True.
        """
        data = super().data
        # now eventually round the data to the number of significant digits
        # for displaying (internally _data as its full precision)
        if data is not None and len(data) > 0 and self._rounding:
            maxval = np.max(np.abs(data))
            rounding = 3
            nd = get_n_decimals(maxval, self.sigdigits) if maxval > 0 else rounding
            data = np.around(data, max(nd, rounding))
        return data

    @data.setter
    def data(self, data):
        # set the data
        self._set_data(data)

        # check if data is 1D
        if self.has_data and len(self.shape) > 1:
            raise ValueError("Only one 1D arrays can be used to define coordinates")

        # linearize the data if possible or at least round it
        # to the number of significant digits

        if self.has_data and self.dtype.kind not in "M":
            # First try to linearize the data if it is not a datetime
            self._linear = False
            self.linearize(self._sigdigits)
            if self._linear:
                return

    @property
    def default(self):
        # this is in case default is called on a coord, while it is a coordset property
        return self

    # ----------------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docstring)
    # some of the property of NDArray has to be hidden because they
    # are not useful for this Coord class
    # ----------------------------------------------------------------------------------
    # NDarray methods

    @property
    def ndim(self):
        if self.linear:
            return 1
        ndim = super().ndim
        if ndim > 1:  # pragma: no cover
            raise ValueError("Coordinate's array should be 1-dimensional!")
        return ndim

    @property
    def T(self):  # no transpose
        return self

    # @property
    # def values(self):
    #    return super().values

    @_docstring.dedent
    def to(self, other, inplace=False, force=False):
        """%(to)s"""
        new = super().to(other, force=force)

        if inplace:
            # update the current object
            self.data = new._data  # here we assign to the data attribute to fire
            # the linearisation (eventually) and the rounding
            # the _linear attribute is set to True if the data are linearized
            self._units = new._units
            self._title = new._title
            self._roi = new._roi
        else:
            new.data = new._data  # here we assign to the data attribute to fire
            # the linearisation (eventually) and the rounding
            return new

    @property
    def masked_data(self):
        return super().masked_data

    @property
    def is_masked(self):
        return False

    @property
    def linear(self):
        """
        Whether the coordinates axis is linear (i.e. regularly spaced)
        """
        if self.has_data and self.dtype.kind not in "M":
            return self._linear
        return False

    @property
    def mask(self):
        return NOMASK

    @mask.setter
    def mask(self, val):
        # Coordinates cannot be masked. Set mask always to NOMASK
        self._mask = NOMASK

    # NDmath methods

    def cumsum(self, **kwargs):
        raise NotImplementedError

    def mean(self, **kwargs):
        raise NotImplementedError

    def pipe(self, func=None, *args, **kwargs):
        raise NotImplementedError

    def remove_masks(self, **kwargs):
        raise NotImplementedError

    def std(self, *args, **kwargs):
        raise NotImplementedError

    def sum(self, *args, **kwargs):
        raise NotImplementedError

    def swapdims(self, *args, **kwargs):
        raise NotImplementedError

    def swapaxes(self, *args, **kwargs):
        raise NotImplementedError

    def squeeze(self, *args, **kwargs):
        raise NotImplementedError

    def random(self, *args, **kwargs):
        raise NotImplementedError

    def empty(self, *args, **kwargs):
        raise NotImplementedError

    def empty_like(self, *args, **kwargs):
        raise NotImplementedError

    def var(self, *args, **kwargs):
        raise NotImplementedError

    def ones(self, *args, **kwargs):
        raise NotImplementedError

    def ones_like(self, *args, **kwargs):
        raise NotImplementedError

    def full(self, *args, **kwargs):
        raise NotImplementedError

    def diag(self, *args, **kwargs):
        raise NotImplementedError

    def diagonal(self, *args, **kwargs):
        raise NotImplementedError

    def full_like(self, *args, **kwargs):
        raise NotImplementedError

    def identity(self, *args, **kwargs):
        raise NotImplementedError

    def eye(self, *args, **kwargs):
        raise NotImplementedError

    def zeros(self, *args, **kwargs):
        raise NotImplementedError

    def zeros_like(self, *args, **kwargs):
        raise NotImplementedError

    def coordmin(self, *args, **kwargs):
        raise NotImplementedError

    def coordmax(self, *args, **kwargs):
        raise NotImplementedError

    def conjugate(self, *args, **kwargs):
        raise NotImplementedError

    def conj(self, *args, **kwargs):
        raise NotImplementedError

    def abs(self, *args, **kwargs):
        raise NotImplementedError

    def absolute(self, *args, **kwargs):
        raise NotImplementedError

    def all(self, *args, **kwargs):
        raise NotImplementedError

    def any(self, *args, **kwargs):
        raise NotImplementedError

    def argmax(self, *args, **kwargs):
        raise NotImplementedError

    def argmin(self, *args, **kwargs):
        raise NotImplementedError

    def asfortranarray(self, *args, **kwargs):
        raise NotImplementedError

    # TODO: make it work
    # def astype(self, dtype=None, **kwargs):
    #     """
    #     Cast the data to a specified type.
    #
    #     Parameters
    #     ----------
    #     dtype : str or dtype
    #         Typecode or data-type to which the array is cast.
    #     """
    #     if dtype is None:
    #         return self # no copy
    #
    #     if isinstance(dtype, str):
    #         dtype = np.dtype(dtype) # convert to dtype
    #
    #     if kwargs.pop("copy", False) or not kwargs.pop("inplace", False):
    #         new = self.copy()
    #     else:
    #         new = self  # no copy
    #         kwargs["copy"] = False
    #
    #     data = self._data.astype(dtype, **kwargs)
    #     new._data = data
    #
    #     return new

    def average(self, *args, **kwargs):
        raise NotImplementedError

    def clip(self, *args, **kwargs):
        raise NotImplementedError

    def get_axis(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def is_complex(self):
        return False  # always real

    @property
    def is_descendant(self):
        return (self._data[-1] - self._data[0]) < 0

    @property
    def dims(self):
        return ["x"]

    @property
    def is_1d(self):
        return True

    # ----------------------------------------------------------------------------------
    # public methods
    # ----------------------------------------------------------------------------------
    def loc2index(self, loc, return_error=False):
        """
        Return the index corresponding to a given location.

        Parameters
        ----------
        loc : float.
            Value corresponding to a given location on the coordinates axis.

        Returns
        -------
        index : int.
            The corresponding index.

        Examples
        --------

        >>> dataset = scp.read("irdata/nh4y-activation.spg")
        >>> dataset.x.loc2index(1644.0)
        4517
        """
        res = self._loc2index(loc)
        if isinstance(res, tuple):
            if return_error:
                return res
            else:
                return res[0]
        return res

    # TODO: new method to replace the old loc2index
    # def loc2index(self, *loc):
    #     """
    #     Return the index(es) corresponding to given location(s).
    #
    #     Parameters
    #     ----------
    #     *loc : int, float, label or str
    #         Value(s) corresponding to given location(s) on the coordinate's axis.
    #
    #     Returns
    #     -------
    #     int
    #         The corresponding index.
    #     """
    #     if self.is_empty:
    #         raise IndexError("Can not search location on an empty array")
    #
    #     # in case several location has been passed
    #     if len(loc) > 1:
    #         return [self.loc2index(loc_) for loc_ in loc]
    #
    #     res = self._interpret_key(*loc)
    #     return res if not isinstance(res, tuple) else res[0]

    def transpose(self, **kwargs):
        return self

    # ----------------------------------------------------------------------------------
    # special methods
    # ----------------------------------------------------------------------------------
    def __copy__(self):
        res = self.copy(deep=False)  # we keep name of the coordinate by default
        res.name = self.name
        return res

    def __deepcopy__(self, memo=None):
        res = self.copy(deep=True, memo=memo)
        res.name = self.name
        return res

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
            "roi",
            "linear",
            "sigdigits",
            "larmor",
        ]

    def __getattr__(self, attr):
        if attr.startswith("_"):
            # raise an error so that traits, ipython operation and more ...
            # will be handled correctly
            raise AttributeError
        if attr in ("default", "coords"):
            # this is in case these attributes are called while it is not a coordset.
            return self
        raise AttributeError

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
            new._data = new.data[keys]

        if self.is_labeled:
            # case only of 1D dataset such as Coord
            new._labels = np.array(self._labels[keys])

        if new.is_empty:
            error_(
                IndexError,
                f"Empty array of shape {new._data.shape} resulted from slicing.\n"
                f"Check the indexes and make sure to use floats for location slicing",
            )
            return None

        new._mask = NOMASK

        # we need to keep the names when copying coordinates to avoid later
        # problems
        new.name = self.name
        return new

    def __str__(self):
        return repr(self)

    # ----------------------------------------------------------------------------------
    # private methods and properties
    # ----------------------------------------------------------------------------------
    # @property
    # def _axis_reversed(self):
    #     # Whether the axis is usually _axis_reversed for plotting.
    #     # This is usually the case of ppm and IR wavenumber.
    #
    #     if self.units == "ppm":
    #         return True
    #     if self.units == "1 / centimeter" and "raman" not in self.title.lower():
    #         return True
    #     return False

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

    def __repr__(self):
        out = self._repr_value().rstrip()
        return out

    @staticmethod
    def _unittransform(new, units):
        oldunits = new.units
        udata = (new.data * oldunits).to(units)
        new._data = udata.m
        new._units = udata.units
        if new._roi is not None:
            roi = (np.array(new._roi) * oldunits).to(units)
            new._roi = list(roi)
        return new

    # ----------------------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------------------
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
        pass

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------

    def set_laser_frequency(self, frequency=15798.26 * ur("cm^-1")):
        """
        Set the laser frequency.

        This method is used to set the laser frequency of the dataset.
        The laser frequency is used to convert the x-axis from optical path
        difference to time. The laser frequency is also used to calculate
        the wavenumber axis.

        Parameters
        ----------
        frequency : `float` or `Quantity`\ , optional, default=15798.26 * ur("cm^-1")
            The laser frequency in cm^-1 or Hz. If the value is in cm^-1, the
            frequency is converted to Hz using the current speed of light value.
        """
        if not isinstance(frequency, Quantity):
            frequency = frequency * ur("cm^-1")

        frequency.ito("Hz")
        self.meta.laser_frequency = frequency

        if self._use_time:
            spacing = 1.0 / frequency
            spacing.ito("picoseconds")
            self._data = np.arange(self.shape[-1]) * spacing.m
            self._units = ur.picoseconds
            self.title = "time"

        else:
            frequency.ito("cm^-1")
            spacing = 1.0 / frequency
            spacing.ito("mm")
            offset = -spacing.m * self._zpd
            self._data = np.arange(self.shape[-1]) * spacing.m + offset
            self._units = ur.mm
            self.title = "optical path difference"

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
    def larmor(self):
        """
        Return larmor frequency in NMR spectroscopy context.
        """
        return self._larmor

    @larmor.setter
    def larmor(self, val):
        self._larmor = val

    @property
    def laser_frequency(self):
        """
        Laser frequency if needed (Quantity).
        """
        return self.meta.laser_frequency

    @laser_frequency.setter
    def laser_frequency(self, val):
        self.meta.laser_frequency = val

    def linearize(self, sigdigits=4):
        """
        Linearize the coordinate's data.

        Make coordinates with an equally distributed spacing, when possible, i.e.,
        if the spacings are not too different when rounded to the number of
        significant digits passed in parameters.
        If the spacings are too different, the coordinates are not linearized.
        In this case, the `linear` attribute is set to False.

        Parameters
        ----------
        sigdigits :  Int, optional, default=4
            The number of significant digit for coordinates values.
        """
        if not self.has_data or self.data.size < 3:
            return

        data = self._data.squeeze()

        self._sigdigits = sigdigits

        spacing = spacings(self._data, sigdigits)

        makeitlinear = is_number(spacing)

        if not makeitlinear and is_iterable(spacing):
            # may be the variation in % are small enough (0.1%)
            variation = (
                (np.max(spacing) - np.min(spacing))
                * 100.0
                / np.abs(np.max(spacing))
                / 2.0
            )
            if variation <= self._linearize_below:
                makeitlinear = True

        if makeitlinear:
            # single spacing with this precision
            # we set the number with their full precision
            # rounding will be made if necessary when reading the data property
            nd = get_n_decimals(np.diff(self._data).max(), self._sigdigits)
            data = np.around(data, nd)
            self._data = np.linspace(data[0], data[-1], data.size)
            self._linear = True
        else:
            # from spectrochempy.application import debug_
            # debug_(
            #      "The coordinates spacing is not enough uniform to allow linearization."
            #  )
            self._linear = False

    @property
    def sigdigits(self):
        """
        Number of significant digits for rounding coordinate values.

        Note
        ----
        The number of significant digits is used when linearizing the coordinates. It is
        also used when setting the coordinates values at the Coord initialization
        or everytime the data array is changed.
        """
        return self._sigdigits

    @sigdigits.setter
    def sigdigits(self, val):
        self._sigdigits = val

    @property
    def spacing(self):
        """
        Coordinate spacing.

        It will be a scalar if the coordinates are uniformly spaced, else
        an array of the different spacings.

        Note
        ----
        The spacing is returned in the units of the coordinate.
        """
        units = self.units if self.units is not None else 1
        if self.has_data:
            return spacings(self._data) * units
        return None


# ======================================================================================
# LinearCoord (Deprecated)
# TODO : should be removed in version 0.8
# ======================================================================================
@tr.signature_has_traits
class LinearCoord(Coord):
    @deprecated(
        kind="object",
        replace="Coord",
        removed="0.8",
    )
    def __init__(self, **kwargs):
        # TODO : remove in version 0.8
        super().__init__(**kwargs)


# ======================================================================================
# Set the operators
# ======================================================================================
_set_operators(Coord, priority=50)
_set_operators(LinearCoord, priority=50)  # Suppress 0.8
