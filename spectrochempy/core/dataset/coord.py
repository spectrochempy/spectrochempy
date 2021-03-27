# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie,
#  Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in
#  the root directory                         =
# ======================================================================================================================
"""
This module implements the class |Coord|.
"""

__all__ = ['Coord', 'LinearCoord']

import textwrap

from traitlets import Bool, observe, All, Unicode, Integer

from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndmath import NDMath, _set_operators
from spectrochempy.utils import colored_output, NOMASK
from spectrochempy.units import Quantity, ur


# ======================================================================================================================
# Coord
# ======================================================================================================================
class Coord(NDMath, NDArray):
    _copy = Bool()

    _html_output = False
    _parent_dim = Unicode(allow_none=True)

    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):
        """
        Explicit coordinates for a dataset along a given axis.

        The coordinates of a |NDDataset| can be created using the |Coord|
        object.
        This is a single dimension array with either numerical (float)
        values or
        labels (str, `Datetime` objects, or any other kind of objects) to
        represent the coordinates. Only a one numerical axis can be defined,
        but labels can be multiple.

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
            See other parameters

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
        dlabel :  str, optional.
            Alias of `title`.
        meta : dict-like object, optional.
            Additional metadata for this object. Must be dict-like but no
            further restriction is placed on meta.
        copy : bool, optional
            Perform a copy of the passed object. Default is False.
        linear : bool, optional
            If set to True, the coordinate is considered as a
            ``LinearCoord`` object.

        See Also
        --------
        NDDataset : Main SpectroChemPy object: an array with masks,
        units and coordinates.
        LinearCoord : Implicit linear coordinates.

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
            raise ValueError('Only one 1D arrays can be used to define coordinates')

    # ..................................................................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `Coord`.

        Rather than isinstance(obj, Coord) use object.implements('Coord').

        This is useful to check type without importing the module
        """

        if name is None:
            return 'Coord'
        else:
            return name == 'Coord'

    # ------------------------------------------------------------------------------------------------------------------
    # readonly property
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    @property
    def reversed(self):
        """bool - Whether the axis is reversed (readonly
        property).
        """
        if self.units in ['1 / centimeter', 'ppm']:
            return True
        return False

        # Return a correct result only if the data are sorted  # return  # bool(self.data[0] > self.data[-1])

    @property
    def default(self):
        # this is in case default is called on a coord, while it is a coordset property
        return self

    # ------------------------------------------------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docstring)
    # some of the property of NDArray has to be hidden because they
    # are not useful for this Coord class
    # ------------------------------------------------------------------------------------------------------------------

    # NDarray methods

    # ..................................................................................................................
    @property
    def is_complex(self):
        return False  # always real

    # ..................................................................................................................
    @property
    def ndim(self):
        ndim = super().ndim
        if ndim > 1:
            raise ValueError("Coordinate's array should be 1-dimensional!")
        return ndim

    # ..................................................................................................................
    @property
    def T(self):  # no transpose
        return self

    # ..................................................................................................................
    # @property
    # def values(self):
    #    return super().values

    # ..................................................................................................................
    @property
    def masked_data(self):
        return super().masked_data

    # ..................................................................................................................
    @property
    def is_masked(self):
        return False

    # ..................................................................................................................
    @property
    def mask(self):
        return super().mask

    # ..................................................................................................................
    @mask.setter
    def mask(self, val):
        # Coordinates cannot be masked. Set mask always to NOMASK
        self._mask = NOMASK

    # NDmath methods

    # ..................................................................................................................
    def cumsum(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def mean(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def pipe(self, func=None, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def remove_masks(self, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def std(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def sum(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def swapdims(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def swapaxes(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def squeeze(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def random(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def empty(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def empty_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def var(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def ones(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def ones_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def full(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def diag(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def diagonal(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def full_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def identity(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def eye(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def zeros(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def zeros_like(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def coordmin(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def coordmax(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def conjugate(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def conj(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def abs(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def absolute(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def all(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def any(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def argmax(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def argmin(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def asfortranarray(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def average(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def clip(self, *args, **kwargs):
        raise NotImplementedError

    # ..................................................................................................................
    def get_axis(self, *args, **kwargs):
        return super().get_axis(*args, **kwargs)

    # ..................................................................................................................
    @property
    def origin(self, *args, **kwargs):
        return None

    # ..................................................................................................................
    @property
    def author(self):
        return None

    # ..................................................................................................................
    @property
    def dims(self):
        return ['x']

    # ..................................................................................................................
    @property
    def is_1d(self):
        return True

    # ..................................................................................................................
    def transpose(self):
        return self

    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------
    def loc2index(self, loc):
        return self._loc2index(loc)

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def __copy__(self):
        res = self.copy(deep=False)  # we keep name of the coordinate by default
        res.name = self.name
        return res

    # ..................................................................................................................
    def __deepcopy__(self, memo=None):
        res = self.copy(deep=True, memo=memo)
        res.name = self.name
        return res

    # ..................................................................................................................
    def __dir__(self):
        # remove some methods with respect to the full NDArray
        # as they are not usefull for Coord.
        return ['data', 'labels', 'units', 'meta', 'title', 'name', 'offset', 'increment', 'linear', 'roi']

    # ..................................................................................................................
    def __getitem__(self, items, return_index=False):
        # we need to keep the names when copying coordinates to avoid later
        # problems
        res = super().__getitem__(items, return_index=return_index)
        res.name = self.name
        return res

    # ..................................................................................................................
    def __str__(self):
        return repr(self)

    # ..................................................................................................................
    def _cstr(self, header='  coordinates: ... \n', print_size=True, **kwargs):

        indent = kwargs.get('indent', 0)

        out = ''
        if not self.is_empty and print_size:
            out += f'{self._str_shape().rstrip()}\n'
        out += f'        title: {self.title}\n' if self.title else ''
        if self.has_data:
            out += '{}\n'.format(self._str_value(header=header))
        elif self.is_empty and not self.is_labeled:
            out += header.replace('...', '\0Undefined\0')

        if self.is_labeled:
            header = '       labels: ... \n'
            text = str(self.labels.T).strip()
            if '\n' not in text:  # single line!
                out += header.replace('...', '\0\0{}\0\0'.format(text))
            else:
                out += header
                out += '\0\0{}\0\0'.format(textwrap.indent(text.strip(), ' ' * 9))

        if out[-1] == '\n':
            out = out[:-1]

        if indent:
            out = "{}".format(textwrap.indent(out, ' ' * indent))

        first_indent = kwargs.get("first_indent", 0)
        if first_indent < indent:
            out = out[indent - first_indent:]

        if not self._html_output:
            return colored_output(out)
        else:
            return out

    # ..................................................................................................................
    def __repr__(self):
        out = self._repr_value().rstrip()
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
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

        if change.name in ['_linear', '_increment', '_offset', '_size']:
            super()._anytrait_changed(change)


class LinearCoord(Coord):
    _use_time = Bool(False)
    _show_datapoints = Bool(True)
    _zpd = Integer

    def __init__(self, *args, offset=0.0, increment=1.0, **kwargs):
        """
        Linear coordinates.

        Such coordinates correspond to a ascending or descending linear
        sequence of values, fully determined
        by two
        parameters, i.e., an offset (off) and an increment (inc) :

        .. math::

            \\mathrm{data} = i*\\mathrm{inc} + \\mathrm{off}

        Parameters
        ----------
        data : a 1D array-like object, optional
            wWen provided, the `size` parameters is adjusted to the size of
            the array, and a linearization of the
            array is performed (only if it is possible: regular spacing in
            the 1.e5 relative accuracy)
        offset : float, optional
            If omitted a value of 0.0 is taken for tje coordinate offset.
        increment : float, optional
            If omitted a value of 1.0 is taken for the coordinate increment.

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
        dlabel :  str, optional.
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
        NDDataset : Main SpectroChemPy object: an array with masks,
        units and coordinates.
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
        if args and isinstance(args[0], Coord) and not args[0].linear:
            raise ValueError('Only linear Coord (with attribute linear set to True, can be transformed into '
                             'LinearCoord class')

        super().__init__(*args, **kwargs)

        # when data is present, we don't need offset and increment, nor size,
        # we just do linear=True and these parameters are ignored

        if self._data is not None:
            self._linear = True

        elif not self.linear:
            # in case it was not already a linear array
            self.offset = offset
            self.increment = increment
            self._linear = True

    # ..................................................................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `LinearCoord`.

        Rather than isinstance(obj, Coord) use object.implements(
        'LinearCoord').

        This is useful to check type without importing the module
        """

        if name is None:
            return 'LinearCoord'
        else:
            return name == 'LinearCoord'

    # ..................................................................................................................
    @property  # read only
    def linear(self):
        return self._linear

    # ..................................................................................................................
    def geomspace(self):
        raise NotImplementedError

    # ..................................................................................................................
    def logspace(self):
        raise NotImplementedError

    # ..................................................................................................................
    def __dir__(self):
        # remove some methods with respect to the full NDArray
        # as they are not usefull for Coord.
        return ['data', 'labels', 'units', 'meta', 'title', 'name', 'offset', 'increment', 'linear', 'size', 'roi',
                'show_datapoints']

    def set_laser_frequency(self, frequency=15798.26 * ur('cm^-1')):

        if not isinstance(frequency, Quantity):
            frequency = frequency * ur('cm^-1')

        frequency.ito('Hz')
        self.meta.laser_frequency = frequency

        if self._use_time:
            spacing = 1. / frequency
            spacing.ito('picoseconds')

            self.increment = spacing.m
            self.offset = 0
            self._units = ur.picoseconds
            self.title = 'time'

        else:
            frequency.ito('cm^-1')
            spacing = 1. / frequency
            spacing.ito('mm')

            self.increment = spacing.m
            self.offset = -self.increment * self._zpd
            self._units = ur.mm
            self.title = 'optical path difference'

    @property
    def _use_time_axis(self):
        # private property
        # True if time scale must be used for interferogram axis. Else it
        # will be set to optical path difference.
        return self._use_time

    @_use_time_axis.setter
    def _use_time_axis(self, val):

        self._use_time = val
        if 'laser_frequency' in self.meta:
            self.set_laser_frequency(self.meta.laser_frequency)

    @property
    def show_datapoints(self):
        """
        Bool : True if axis must discard values and show only datapoints.

        """
        if 'laser_frequency' not in self.meta or self.units.dimensionality not in ['[time]', '[length]']:
            return False

        return self._show_datapoints

    @show_datapoints.setter
    def show_datapoints(self, val):

        self._show_datapoints = val

    @property
    def laser_frequency(self):
        """
        Quantity: Laser frequency (if needed)
        """
        return self.meta.laser_frequency

    @laser_frequency.setter
    def laser_frequency(self, val):
        self.meta.aser_frequency = val


# ======================================================================================================================
# Set the operators
# ======================================================================================================================
_set_operators(Coord, priority=50)

# ======================================================================================================================
if __name__ == '__main__':
    pass
