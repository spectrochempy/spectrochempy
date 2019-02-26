# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module implements the |NDDataset| class.

"""

__all__ = ['NDDataset', 'HAS_XARRAY', 'HAS_PANDAS']

__dataset_methods__ = []

# ======================================================================================================================
# Standard python imports
# ======================================================================================================================

import itertools
import textwrap
from datetime import datetime
import warnings
from collections import OrderedDict
import re

# ======================================================================================================================
# third-party imports
# ======================================================================================================================
import numpy as np
from traitlets import (HasTraits, List, Unicode, Instance, Bool, All, Float,
                       validate, observe, default)
import matplotlib.pyplot as plt

try:
    import xarray as xr

    HAS_XARRAY = True
except:
    HAS_XARRAY = False

try:
    import pandas as pd
    from pandas.core.generic import NDFrame

    HAS_PANDAS = True
except:
    HAS_PANDAS = False

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from spectrochempy.utils import (SpectroChemPyWarning, SpectroChemPyDeprecationWarning, SpectroChemPyException,
                                 get_user_and_node, set_operators, docstrings,
                                 make_func_from, deprecated, info_, debug_, error_, warning_)
from spectrochempy.extern.traittypes import Array
from spectrochempy.core.project.baseproject import AbstractProject
from .ndarray import NDArray, DEFAULT_DIM_NAME
from .ndcomplex import NDComplexArray
from .ndcoords import Coord, CoordSet
from .ndmath import NDMath
from .ndio import NDIO
from .ndplot import NDPlot
from spectrochempy.utils import (INPLACE, TYPE_INTEGER,
                                 TYPE_COMPLEX, TYPE_FLOAT,
                                 colored_output)


# ======================================================================================================================
# numpy print options
# ======================================================================================================================

# numpyprintoptions()   probably we don't need this


# ======================================================================================================================
# NDDataset class definition
# ======================================================================================================================

class NDDataset(
    NDIO,
    NDPlot,
    NDMath,
    NDComplexArray,
):
    """
    The main N-dimensional dataset class used by |scpy|.

    """
    author = Unicode(get_user_and_node(),
                     desc='Name of the author of this dataset',
                     config=True)

    # private metadata in addition to those of the base NDArray class
    _modified = Instance(datetime)
    _description = Unicode
    _history = List(Unicode())

    # coordinates
    _coords = Instance(CoordSet, allow_none=True)

    # model data (e.g., for fit)
    _modeldata = Array(Float(), allow_none=True)

    # some setting for NDDataset
    _copy = Bool(False)
    _labels_allowed = Bool(False)  # no labels for NDDataset

    # dataset can be members of a project or panels.
    # we use the abstract class to avoid circular imports.
    _parent = Instance(AbstractProject, allow_none=True)

    # _ax is a hidden variable containing the matplotlib axis defined
    # for a NDArray object.
    # most generally it is accessed using the public read-only property ax
    _ax = Instance(plt.Axes, allow_none=True)
    _fig = Instance(plt.Figure, allow_none=True)

    # ------------------------------------------------------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------------------------------------------------------

    docstrings.delete_params('NDArray.parameters', 'labels')

    # ..................................................................................................................
    @docstrings.dedent
    def __init__(self,
                 data=None,
                 coords=None,
                 coordunits=None,
                 coordtitles=None,
                 description='',
                 **kwargs):
        """
        Parameters
        ----------
        %(NDArray.parameters.no_labels)s

        coords : An instance of |CoordSet|, optional
            `coords` contains the coordinates for the different
            dimensions of the `data`. if `coords` is provided, it must
            specified
            the `coord` and `labels` for all dimensions of the `data`.
            Multiple `coord`'s can be specified in an |CoordSet| instance
            for each dimension.
        description : str, optional
            A optional description of the nd-dataset.

        Notes
        -----
        The underlying array in a |NDDataset| object can be accessed
        through the `data` attribute, which will return a conventional
        |ndarray|.

        Examples
        --------
        Usage by an end-user:

        >>> from spectrochempy import *

        >>> x = NDDataset([1,2,3])
        >>> print(x.data) # doctest : +NORMALIZE_WHITESPACE
        [       1        2        3]


        """
        super().__init__(data, **kwargs)

        self.parent = None
        self._modified = self._date
        self._history = []

        if coords is not None:
            self.coords = coords

        if coordtitles is not None:
            self.coordtitles = coordtitles

        if coordunits is not None:
            self.coordunits = coordunits

        if description:
            self.description = description

    #
    # Default values
    #
    # ..................................................................................................................
    @default('_coords')
    def _coords_default(self):
        return None  # CoordSet([Coord() for dim in self.shape])

    # ..................................................................................................................
    @default('_copy')
    def _copy_default(self):
        return False

    # ..................................................................................................................
    @default('_modeldata')
    def _modeldata_default(self):
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # additional properties (not in the NDArray base class)
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def data(self):
        """
        |ndarray|, The `data` array.

        If there is no data but labels, then the labels are returned instead of data.

        .. note::
            See the |userguide|_ for more information

        """
        return super().data

    # ..................................................................................................................
    @data.setter
    def data(self, data):
        # as we can't write super().data = data, we call _set_data
        # see comment in the data.setter of NDArray
        super()._set_data(data)

        if HAS_PANDAS and isinstance(data, NDFrame):  # pandas object
            if hasattr(self, "coords"):  # case of the NDDataset subclass
                self.coords = data.axes

    # .................................................................................................................
    @property
    def description(self):
        """
        str - Provides a description of the underlying data

        """
        return self._description

    # ..................................................................................................................
    @description.setter
    def description(self, value):
        self._description = value

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
    @validate('_coords')
    def _coords_validate(self, proposal):
        coords = proposal['value']
        return self._valid_coords(coords)

    # ..................................................................................................................
    def _valid_coords(self, coords):
        # uses in coords_validate and setattr
        if coords is None:
            return

        for k, coord in enumerate(coords):

            if coord is not None and coord.data is None:
                continue

            # For coord to be acceptable, we require at least a NDArray, a NDArray subclass or a CoordSet
            if not isinstance(coord, (Coord, CoordSet)):
                if isinstance(coord, NDArray):
                    coord = coords[k] = Coord(coord)
                else:
                    raise TypeError('Coordinates must be an instance or a subclass of Coord class or NDArray, or of '
                                    f' CoordSet class, but an instance of {type(coord)} has been passed')

            if self.dims and coord.name not in self.dims:
                raise AttributeError(f'The name of a coordinate must have name among the current dims: {self.dims}'
                                     f' but the name is `{coord.name}`')

            # check the validity of the given coordinates in terms of size
            size = coord.size

            idx = self._get_dims_index(coord.name)[0]  # idx in self.dims
            if size != self._data.shape[idx]:
                raise ValueError(
                    f'the size of each coordinates array must None or be equal to that of the respective `{coord.name}`'
                    f' data dimension but coordinate size={size} != data shape[{idx}]={self._data.shape[idx]}')

        coords._parent = self
        return coords

    # ..................................................................................................................
    @property
    def coords(self):
        """
        |CoordSet| instance - Contains the coordinates of the various
        dimensions of the dataset.

        """
        return self._coords

    # ..................................................................................................................
    @coords.setter
    def coords(self, value):

        if value is not None:

            if self._coords is not None:
                debug_("Overwriting NDDataset's current coords with one specified")

            if not isinstance(value, CoordSet):
                coords = CoordSet(value)
            else:
                coords = value

            # make some adjustement to the coordset
            for i, coord in enumerate(coords.copy()):
                # iterate on a copy as we may change the coordset during this  process
                if i >= self.ndim:
                    # there are too many coords
                    warnings.warn(f"There is too many coordinates defined for this dataset."
                                  f" The extra {i}'s coordinates ({coord.name}) will be eliminated. ",
                                  SpectroChemPyWarning)
                    # remove this extra coordinates
                    del coords[self.ndim]

                    continue  # bypass the remaining code

            # eventually set the names
            coords.names = DEFAULT_DIM_NAME[-self.ndim:]
            # now we can go for a final validation
            self._coords = coords
        else:
            # reset coordinates
            self._coords = None

        if self._coords:
            # set a notifier to the updated traits of the CoordSet instance
            HasTraits.observe(self._coords, self._dims_update, '_updated')
            # force it one time after this initialization
            self._coords._updated = True

    # ..................................................................................................................
    @property
    def coordset(self):
        """
        |CoordSet| instance - Contains the coordinates of the various
        dimensions of the dataset.[DEPRECATED - use coords instead]

        """
        warnings.warn('DEPRECATED: use coords instead', SpectroChemPyDeprecationWarning)
        return self._coords

    @coordset.setter
    def coordset(self, value):
        warnings.warn('DEPRECATED: use coords instead', SpectroChemPyDeprecationWarning)
        self.coords = value

    # ..................................................................................................................
    @property
    def coordtitles(self):
        """
        list - list of the |Coord| titles.

        """
        if self._coords is not None:
            return self._coords.titles

    # ..................................................................................................................
    @coordtitles.setter
    def coordtitles(self, value):
        if self._coords is not None:
            self._coords.titles = value

    # ..................................................................................................................
    @property
    def coordunits(self):
        """
        list - list of the |Coord| units

        """
        if self._coords is not None:
            return self._coords.units

    # ..................................................................................................................
    @coordunits.setter
    def coordunits(self, value):

        if self._coords is not None:
            self._coords.units = value

    # ..................................................................................................................
    @property
    def modeldata(self):
        """
        |ndarray| - models data eventually generated by modelling of the data

        """
        return self._modeldata

    # ..................................................................................................................
    @modeldata.setter
    def modeldata(self, data):
        self._modeldata = data

    # ..................................................................................................................
    @property
    def T(self):
        """
        |NDDataset| - Transposed array.

        The same object is returned if `ndim` is less than 2.

        """
        return self.transpose()

    # ..................................................................................................................
    @property
    def modified(self):
        """
        `Datetime` object - Date of modification (readonly property).

        """
        return self._modified

    # ..................................................................................................................
    @property
    def parent(self):
        """
        |Project| instance - The parent project of the dataset.

        """
        return self._parent

    # ..................................................................................................................
    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            # A parent project already exists for this dataset but the
            # entered values gives a different parent. This is not allowed,
            # as it can produce impredictable results. We will first remove it
            # from the current project.
            self._parent.remove_dataset(self.name)
        self._parent = value

    # ------------------------------------------------------------------------------------------------------------------
    # hidden properties (for the documentation, only - we remove the docs)
    # some of the property of NDArray has to be hidden because they are not
    # useful for this Coord class
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def labels(self):
        # not valid for NDDataset
        "There is no label for nd-dataset"
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def implements(self, name=None):
        # For compatibility with pyqtgraph
        # Rather than isinstance(obj, NDDataset) use object.implements(
        # 'NDDataset')
        # This is useful to check type without importing the module
        if name is None:
            return ['NDDataset']
        else:
            return name == 'NDDataset'

    # ..................................................................................................................
    def coord(self, dim='x'):
        """
        Returns the coordinates along the given dimension.

        Parameters
        ----------
        dim : int or str.
            A dimension index or name, default index = `x` for the last axis

        Returns
        -------
        |Coord|
            Coordinates along the given axis.

        """
        idx = self._get_dims_index(dim)[0]  # should generate an error if the
        # dimension name is not recognized
        if idx is None:
            return None

        if self._coords is None:
            return None

        # idx is not necessarily the position of the coordinates in the CoordSet
        # indeed, transposition may have taken place. So we need to retrieve the coordinates by its name
        name = self.dims[idx]
        if name in self._coords.names:
            idx = self._coords.names.index(name)
            return self._coords[idx]
        else:
            error_(f'could not fin this dimenson name: `{name}`')
            return None

    # ..................................................................................................................
    @docstrings.dedent
    def transpose(self, *dims, inplace=False):
        """
        Permute the dimensions of a NDDataset.

        Parameters
        ----------
        dims : sequence of dimension indexes or names, optional.
            By default, reverse the dimensions, otherwise permute the dimensions
            according to the values given.
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        See Also
        --------
        swapaxes

        """
        new = super().transpose(*dims, inplace=inplace)
        new.history = 'Data transposed' + f' between dims: {dims}' if dims else ''
        return new

    # ..................................................................................................................
    @docstrings.dedent
    def swapaxes(self, dim1, dim2, inplace=False):
        """
        Interchange two dimensions of a NDDataset.

        Parameters
        ----------
        dim1 : int
            First axis.
        dim2 : int
            Second axis.
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        See Also
        --------
        transpose

        """

        new = super().swapaxes(dim1, dim2, inplace=inplace)
        new.history = f'Data swapped between dims {dim1} and {dim2}'
        return new

    # ..................................................................................................................
    def take(self, indices, **kwargs):
        """

        Parameters
        ----------
        indices
        kwargs

        Returns
        -------

        """

        # handle the various syntax to pass the axis
        dims = self._get_dims_from_args(**kwargs)
        axis = self._get_dims_index(dims)
        axis = axis[0] if axis else None

        # indices = indices.tolist()
        if axis is None:
            # just do a fancy indexing
            return self[indices]

        if axis < 0:
            axis = self.ndim + axis

        index = tuple([...] + [indices] + [slice(None) for i in range(self.ndim - 1 - axis)])
        new = self[index]
        return new

    # ..................................................................................................................
    @docstrings.dedent
    def sort(self, **kwargs):
        """
        Returns the dataset sorted along a given dimension
        (by default, the first dimension [axis=0]) using the numeric or label values

        Parameters
        ----------
        dim : str or int, optional, default = 0
            dimension index or name along which to sort.
        pos: int , optional
            If labels are multidimensional  - allow to sort on a define
            row of labels: labels[pos]. Experimental: Not yet checked
        by : str among ['value', 'label'], optional, default = ``value``.
            Indicate if the sorting is following the order of labels or
            numeric coord values.
        descend : `bool`, optional, default = `False`.
            If true the dataset is sorted in a descending direction. Default is False  except if coordinates
            are reversed.
        %(generic_method.parameters.inplace)s

        Returns
        -------
        %(generic_method.returns)s

        """

        inplace = kwargs.get('inplace', False)
        if not inplace:
            new = self.copy()
        else:
            new = self


        pos = kwargs.pop('pos', None)
        by = kwargs.pop('by', 'value')

        axis = self.get_axis(**kwargs)
        if axis is None:
            axis = 0

        descend = kwargs.pop('descend', None)
        if descend is None:
            # when non specified, default is False (except for reversed coordinates
            descend = self._coords[axis].reversed

        indexes = []
        for i in range(self.ndim):
            if i == axis:
                if not self._coords[axis].has_data:
                    # sometimes we have only label for Coord objects.
                    # in this case, we sort labels if they exist!
                    if self._coords[axis].is_labeled:
                        by = 'label'
                    else:
                        # nothing to do for sorting
                        # return self itself
                        return self

                args = self._coords[axis]._argsort(by=by, pos=pos, descend=descend)
                new._coords[axis] = self._coords[axis][args]
                indexes.append(args)
            else:
                indexes.append(slice(None))

        new._data = new._data[indexes]
        if new.is_masked:
            new._mask = new._mask[indexes]

        return new

    # def set_complex(self):
    #    raise NotImplementedError

    # def set_quaternion(self):
    #    raise NotImplementedError

    # ..................................................................................................................
    def squeeze(self, *dims, inplace=False):
        """
        Remove single-dimensional entries from the shape of a NDDataset.

        Parameters
        ----------
        dim : None or int or tuple of ints, optional
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
            If `dim` is not `None`, and the dimension being squeezed is not
            of length 1
        """
        old = self.dims[:]

        new, axis = super(NDDataset, self).squeeze(*dims, inplace=inplace, return_axis=True)

        if new._coords is not None:
            # if there are coordinates they have to be squeezed as well (remove
            # coordinate for the squeezed axis)

            for i in axis:
                dim = old[i] 
                idx = new._coords.names.index(dim)
                del new._coords[idx]

        return new

    # ..................................................................................................................
    def set_real(self, axis):
        raise NotImplementedError

    # Create the returned values of functions should be same class as input.
    # The units should have been handled by __array_wrap__ already

    # ... Converters ..........................................................
    def to_dataframe(self, **kwargs):
        """
        Convert to a tidy structured Pandas DataFrame.
        (needs Pandas library installed)

        tidy data :  http://www.jstatsoft.org/v59/i10/

        Each column holds a different variable (For a NDDataset there is only one column)
        Each rows holds a different observation.

        """
        if not HAS_PANDAS:
            warnings.warn('Pandas is not available! This function can not be used',
                          SpectroChemPyWarning)
            return None

        columns = [self.title]  # title
        data = [self.data.reshape(-1)]
        index = self._coords.to_index()

        return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)

    def to_xarray(self, **kwargs):
        """
        Convert a NDDataset instance to an `~xarray.DataArray` object
        ( the xarray library must be available )

        Parameters

        Returns
        -------
        xarray : a xarray.DataArray object


        """
        # Information about DataArray from the DataArray docstring
        #
        # Attributes
        # ----------
        # dims : tuple
        #     Dimension names associated with this array.
        # values : np.ndarray
        #     Access or modify DataArray values as a numpy array.
        # coords : dict-like
        #     Dictionary of DataArray objects that label values along each dimension.
        # name : str or None
        #     Name of this array.
        # attrs : OrderedDict
        #     Dictionary for holding arbitrary metadata.
        # Init docstring:
        # Parameters
        # ----------
        # data : array_like
        #     Values for this array. Must be an ``numpy.ndarray``, ndarray like,
        #     or castable to an ``ndarray``. If a self-described xarray or pandas
        #     object, attempts are made to use this array's metadata to fill in
        #     other unspecified arguments. A view of the array's data is used
        #     instead of a copy if possible.
        # coords : sequence or dict of array_like objects, optional
        #     Coordinates (tick labels) to use for indexing along each dimension.
        #     If dict-like, should be a mapping from dimension names to the
        #     corresponding coordinates. If sequence-like, should be a sequence
        #     of tuples where the first element is the dimension name and the
        #     second element is the corresponding coordinate array_like object.
        # dims : str or sequence of str, optional
        #     Name(s) of the data dimension(s). Must be either a string (only
        #     for 1D data) or a sequence of strings with length equal to the
        #     number of dimensions. If this argument is omitted, dimension names
        #     are taken from ``coords`` (if possible) and otherwise default to
        #     ``['dim_0', ... 'dim_n']``.
        # name : str or None, optional
        #     Name of this array.
        # attrs : dict_like or None, optional
        #     Attributes to assign to the new instance. By default, an empty
        #     attribute dictionary is initialized.
        # encoding : dict_like or None, optional
        #     Dictionary specifying how to encode this array's data into a
        #     serialized format like netCDF4. Currently used keys (for netCDF)
        #     include '_FillValue', 'scale_factor', 'add_offset', 'dtype',
        #     'units' and 'calendar' (the later two only for datetime arrays).
        #     Unrecognized keys are ignored.
        if not HAS_XARRAY:
            warnings.warn('Xarray is not available! This function can not be used',
                          SpectroChemPyWarning)
            return None
        x, y = self.x, self.y
        tx = x.title
        if y:
            ty = y.title
            da = xr.DataArray(np.array(self.data, dtype=np.float64),
                              coords=[(ty, y.data), (tx, x.data)],
                              attrs=self.meta,
                              )
            da.attrs['units'] = (y.units, x.units, self.units)
        else:
            da = xr.DataArray(np.array(self.data, dtype=np.float64),
                              coords=[(tx, x.data)],
                              attrs=self.meta,
                              )
            da.attrs['units'] = (x.units, self.units)

        return da

    def to_panel(self, **kwargs):
        # TODO: for now I just return a list, waiting that the NDPanel structure is finished
        return [self]

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def __dir__(self):
        return ['data', 'dims', 'mask', 'labels', 'units',
                'meta', 'plotmeta', 'name', 'title', 'coords', 'description',
                'history', 'date', 'modified', 'modeldata'] + NDIO().__dir__()

    # ..................................................................................................................
    def __getitem__(self, items):

        saveditems = items

        new, items = super().__getitem__(items, return_index=True)

        if new is None:
            return None

        if self._coords is not None:
            names = self.coords.names  # all names of the current coordinates
            new_coords = [None] * len(names)
            for i, item in enumerate(items):
                # get the corresponding dimension name in the dims list
                name = self.dims[i]
                # get the corresponding index in the coordinate's names list
                idx = names.index(name)
                if self._coords[idx].is_empty:
                    new_coords[idx] = Coord(None, name=self._coords[idx].name)
                elif isinstance(item, slice):
                    # add the slice on the corresponding coordinates on the dim to the new list of coordinates
                    new_coords[idx] = self._coords[idx][item]
                elif isinstance(item, (np.ndarray, list)):
                    new_coords[idx] = Coord(item, name=self._coords[idx].name)
            new._coords = CoordSet(new_coords, keepnames=True)

        new.history = f'slice extracted: ({saveditems})'
        return new

    # ..................................................................................................................
    def __getattr__(self, item):
        # when the attribute was not found
        if item in ["__numpy_ufunc__", "interface"] or '_validate' in item or \
                '_changed' in item:
            # raise an error so that masked array will be handled correctly
            # with arithmetic operators and more
            raise AttributeError

        elif item in DEFAULT_DIM_NAME:  # syntax such as ds.x, ds.y, etc...
            return self.coord(item)

    #        return super().__getattr__(item)

    def __setattr__(self, item, value):
        # debug_(item, value)
        if item in DEFAULT_DIM_NAME:  #:# syntax such as ds.x, ds.y, etc...
            # Note the above test is important to avoid errors with traitlets
            # even if it looks redundant with the folllowing
            if item in self.dims:
                if self.coords is None:
                    # we need to create a coordset first
                    self.coords = [None] * self.ndim
                idx = self.coords.names.index(item)
                _coords = self.coords
                _coords[idx] = Coord(value, name=item)
                _coords = self._valid_coords(_coords)
                self.coords = _coords
            else:
                raise AttributeError(f'Coordinate `{item}` is not used.')

        else:
            super().__setattr__(item, value)

    # ..................................................................................................................
    def __eq__(self, other, attrs=None):
        attrs = self.__dir__()
        for attr in ('filename', 'plotmeta', 'name', 'description',
                     'history', 'date', 'modified'):
            attrs.remove(attr)
        # some attrib are not important for equality
        return super(NDDataset, self).__eq__(other, attrs)

    # ..................................................................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return super().__hash__ + hash(self._coords)

    # ..................................................................................................................
    def _cstr(self):
        # Display the metadata of the object and partially the data
        out = ''
        out += '           id: {}\n'.format(self.id)
        out += '         name: {}\n'.format(self.name) if self._name else ''
        out += '       author: {}\n'.format(self.author)
        out += '      created: {}\n'.format(self._date)
        out += '     modified: {}\n'.format(self._modified) \
            if self._modified != self._date else ''

        wrapper1 = textwrap.TextWrapper(initial_indent='',
                                        subsequent_indent=' ' * 15,
                                        replace_whitespace=True,
                                        width=self._text_width)

        pars = self.description.strip().splitlines()
        if pars:
            out += '  description: '
            desc = ''
            if pars:
                desc += '{}\n'.format(wrapper1.fill(pars[0]))
            for par in pars[1:]:
                desc += '{}\n'.format(textwrap.indent(par, ' ' * 15))
            # the three escaped null characters are here to facilitate
            # the generation of html outputs
            desc = '\0\0\0{}\0\0\0\n'.format(desc.rstrip())
            out += desc

        if self._history:
            pars = self.history
            out += '      history: '
            hist = ''
            if pars:
                hist += '{}\n'.format(wrapper1.fill(pars[0]))
            for par in pars[1:]:
                hist += '{}\n'.format(textwrap.indent(par, ' ' * 15))
            hist = '\0\0\0{}\0\0\0\n'.format(hist.rstrip())
            out += hist

        out += '{}\n'.format(self._str_value().rstrip())
        out += '{}\n'.format(self._str_shape().rstrip())
        out += '{}\n'.format(self._str_dims().rstrip())

        if not out.endswith('\n'):
            out += '\n'
        out += '\n'

        if not self._html_output:
            return colored_output(out.rstrip())
        else:
            return out.rstrip()

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _loc2index(self, loc, dim):
        # Return the index of a location (label or coordinates) along the dim
        # This can work only if `coords` exists.

        if self._coords is None:
            raise SpectroChemPyException(
                'No coords have been defined. Slicing or selection'
                ' by location ({}) needs coords definition.'.format(loc))

        coord = self.coord(dim)
        return coord._loc2index(loc)

    # ..................................................................................................................
    def _str_dims(self):
        if self.is_empty:
            return ''
        if len(self.dims) < 1 or not hasattr(self, "_coords"):
            return ''
        if not self._coords or len(self._coords) < 1:
            return ''
        txt = ''
        for dim, idx in zip(self.coords.names, self._get_dims_index(self.coords.names)):
            txt += ' DIMENSION `{}`\n'.format(dim)
            txt += '        index: {}\n'.format(idx)
            if hasattr(self, 'coords') and self.coords:
                # coordinates if available
                coord = self.coord(dim)
                coord._html_output = self._html_output
                txt += '{}\n'.format(coord._cstr())
        txt = txt.rstrip()  # remove the trailing '\n'
        return txt

    _repr_dims = _str_dims

    # ------------------------------------------------------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------------------------------------------------------

    def _dims_update(self, change=None):
        # when notified that a coords names have been updated
        _ = self.dims  # fire an update

        debug_('dims have been updated')

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

        if change['name'] in ["_date", "_modified", "trait_added"]:
            return

        # changes in data -> we must update dates
        if change['name'] == '_data' and self._date == datetime(1970, 1, 1, 0,
                                                                0):
            debug_('changes in NDDataset: ' + change.name)

            self._date = datetime.now()
            self._modified = datetime.now()

            # self._dims_update()

        # all the time -> update modified date
        self._modified = datetime.now()

        return


# ======================================================================================================================
# module function
# ======================================================================================================================

# make some functions also accessible from the scp API
# We want a slightly different docstring so we cannot just make:
#     func = NDDataset.func

sort = make_func_from(NDDataset.sort, first='dataset')
swapaxes = make_func_from(NDDataset.swapaxes, first='dataset')
transpose = make_func_from(NDDataset.transpose, first='dataset')

abs = make_func_from(NDDataset.abs, first='dataset')
conjugate = make_func_from(NDDataset.conjugate, first='dataset')  # defined in ndarray
set_complex = make_func_from(NDDataset.set_complex, first='dataset')

__all__ += ['sort',
            'swapaxes',
            'transpose',
            'abs',
            'conjugate',
            'set_complex',
            ]

# ======================================================================================================================
# Set the operators
# ======================================================================================================================

set_operators(NDDataset, priority=100000)
