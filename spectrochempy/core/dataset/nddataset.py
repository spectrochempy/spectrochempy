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

__all__ = ['NDDataset']

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

# ======================================================================================================================
# Local imports
# ======================================================================================================================

from ...extern.traittypes import Array
from ...core.project.baseproject import AbstractProject
from .ndarray import NDArray, DEFAULT_DIM_NAME
from .ndcomplex import NDComplexArray
from .ndcoord import Coord
from .ndcoordset import CoordSet
from .ndmath import NDMath, set_operators, make_func_from
from .ndio import NDIO
from .ndplot import NDPlot
from ...core import HAS_XARRAY, HAS_PANDAS, info_, debug_, error_, warning_, print_
from ...utils import (INPLACE, TYPE_INTEGER, TYPE_COMPLEX, TYPE_FLOAT,
                      colored_output, convert_to_html,
                      SpectroChemPyWarning, SpectroChemPyDeprecationWarning, SpectroChemPyException,
                      get_user_and_node, docstrings, deprecated)


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
                 history='',
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
        Usage by an end-user :

        >>> from spectrochempy import *

        >>> x = NDDataset([1,2,3])
        >>> print(x.data) # doctest: +NORMALIZE_WHITESPACE
        [       1        2        3]


        """
        super().__init__(data, **kwargs)
        
        self._parent = None
        self._modified = self._date
        
        # eventually set the coordinates with optional units and title
        
        if isinstance(coords, CoordSet):
            self.set_coords(**coords)
            
            if coordunits is not None:
                warning_('When a CoordSet object is passed to set the coordinates, coordunits are ignored. '
                         'To change units, do it in the passed CoordSet')
            
            if coordtitles is not None:
                warning_('When a CoordSet object is passed to set the coordinates, coordtitles are ignored. '
                         'To change titles, do it in the passed CoordSet')
        
        else:
            if coords is None:
                coords = [None] * self.ndim
            
            if coordunits is None:
                coordunits = [None] * self.ndim
            
            if coordtitles is None:
                coordtitles = [None] * self.ndim
            
            _coords = []
            for c, u, t in zip(coords, coordunits, coordtitles):
                if not isinstance(c, CoordSet):
                    coord = Coord(c)
                    if u is not None:
                        coord.units = u
                    if t is not None:
                        coord.title = t
                else:
                    if u:
                        warning_('units have been set for a CoordSet, but this will be ignored '
                                 '(units are only defined at the coordinate level')
                    if t:
                        warning_('title will be ignored as they are only defined at the coordinates level')
                    coord = c
                
                _coords.append(coord)
            
            if _coords and set(_coords) != {Coord()}:  # if they are no coordinates do nothing
                self.set_coords(*_coords)
        
        if description:
            self.description = description

        self._history = []
        if history:
            self.history = history
    
    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    def __dir__(self):
        return ['data', 'dims', 'mask', 'units',
                'meta', 'plotmeta', 'name', 'title', 'coords', 'description',
                'history', 'date', 'modified', 'modeldata', 'origin'] + NDIO().__dir__()
    
    # ..................................................................................................................
    def __getitem__(self, items):
        
        saveditems = items
        
        # coordinate selection to test first
        if isinstance(items, str):
            try:
                return self._coords[items]
            except:
                pass
        
        # slicing
        new, items = super().__getitem__(items, return_index=True)
        
        if new is None:
            return None
        
        if self._coords is not None:
            names = self._coords.names  # all names of the current coordinates
            new_coords = [None] * len(names)
            for i, item in enumerate(items):
                # get the corresponding dimension name in the dims list
                name = self.dims[i]
                # get the corresponding index in the coordinate's names list
                idx = names.index(name)
                if self._coords[idx].is_empty:
                    new_coords[idx] = Coord(None, name=name)
                elif isinstance(item, slice):
                    # add the slice on the corresponding coordinates on the dim to the new list of coordinates
                    if not isinstance(self._coords[idx], CoordSet):
                        new_coords[idx] = self._coords[idx][item]
                    else:
                        # we must slice all internal coordinates
                        newc = []
                        for c in self._coords[idx]:
                            newc.append(c[item])
                        new_coords[idx] = CoordSet(*newc, name=name)
                 
                elif isinstance(item, (np.ndarray, list)):
                    new_coords[idx] = self._coords[idx][item] # Coord(item, name=self._coords[idx].name) (fixes issue #20)
            
            new.set_coords(*new_coords, keepnames=True)
        
        new.history = f'Slice extracted: ({saveditems})'
        return new
    
    # ..................................................................................................................
    def __getattr__(self, item):
        # when the attribute was not found
        if item in ["__numpy_ufunc__", "interface", '_pytestfixturefunction','__dataclass_fields__',
                    '_ipython_canary_method_should_not_exist_',
                    '_baseclass', '_fill_value',
                    '_ax_lines', '_axcb', 'clevels', '__wrapped__',
                    '__await__','__aiter__'] \
                or '_validate' in item or '_changed' in item:
            # raise an error so that traits, ipython operation and more ... will be handled correctly
            raise AttributeError
        
        # syntax such as ds.x, ds.y, etc...
        
        if item[0] in self.dims or self._coords:
            
            # look also properties
            attribute = None
            index = 0
            #print(item)
            if len(item) > 2 and item[1] == '_':
                attribute = item[1:]
                item = item[0]
                index = self.dims.index(item)
            
            if self._coords:
                try:
                    c = self._coords[item]
                    if isinstance(c, str) and c in self.dims:
                        # probaly a reference to another coordinate name
                        c = self._coords[c]
                    
                    if c.name in self.dims or c._parent_dim in self.dims:
                        if attribute is not None:
                            # get the attribute
                            return getattr(c, attribute)
                        else:
                            return c
                    else:
                        raise AttributeError
                
                except Exception as err:
                    if item in self.dims:
                        return None
                    else:
                        raise err
            elif attribute is not None:
                if attribute == 'size':
                    # we want the size but there is no coords, get it from the data shape
                    return self.shape[index]
                else:
                    raise AttributeError(f'Can not find `{attribute}` when no coordinate is defined')
            
            return None
        
        raise AttributeError
    
    def __setattr__(self, key, value):
        #
        #     keyb = key[1:] if key.startswith('_') else key
        #     if keyb in ['copy', 'title', 'modified', 'units', 'meta', 'name', 'parent', 'author', 'dtype',
        #                 'data', 'date', 'filename', 'coords', 'html_output',
        #                 'description', 'history', 'id', 'dims', 'mask', '_mask_metadata',
        #                 'labels', 'plotmeta', 'modeldata', 'modelnames',
        #                 'figsize', 'fig', 'ndaxes', 'clevels', 'divider', 'fignum', 'ax_lines', 'axcb',
        #                 'trait_values', 'trait_notifiers', 'trait_validators', 'cross_validation_lock', 'notify_change']:
        #         super().__setattr__(key, value)
        #         return
        #
        if key in DEFAULT_DIM_NAME:  #:# syntax such as ds.x, ds.y, etc...
            # Note the above test is important to avoid errors with traitlets
            # even if it looks redundant with the folllowing
            if key in self.dims:
                if self._coords is None:
                    # we need to create a coordset first
                    self.set_coords(dict((self.dims[i], None) for i in range(self.ndim)))
                idx = self._coords.names.index(key)
                _coords = self._coords
                _coords[idx] = Coord(value, name=key)
                _coords = self._valid_coords(_coords)
                self._coords.set(_coords)
            else:
                raise AttributeError(f'Coordinate `{key}` is not used.')
        else:
            super().__setattr__(key, value)
    
    # ..................................................................................................................
    def __eq__(self, other, attrs=None):
        attrs = self.__dir__()
        for attr in ('filename', 'plotmeta', 'name', 'description', 'history', 'date', 'modified'):
            # these attibutes are not used for comparison (comparison based on data and units!)
            attrs.remove(attr)
        return super().__eq__(other, attrs)
    
    # ..................................................................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return super().__hash__ + hash(self._coords)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @default('_coords')
    def _coords_default(self):
        return None
    
    # ..................................................................................................................
    @default('_copy')
    def _copy_default(self):
        return False
    
    # ..................................................................................................................
    @default('_modeldata')
    def _modeldata_default(self):
        return None
    
    # ------------------------------------------------------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @validate('_coords')
    def _coords_validate(self, proposal):
        coords = proposal['value']
        return self._valid_coords(coords)
    
    def _valid_coords(self, coords):
        # uses in coords_validate and setattr
        if coords is None:
            return
        
        for k, coord in enumerate(coords):
            
            if coord is not None and not isinstance(coord, CoordSet) and coord.data is None:
                continue
            
            # For coord to be acceptable, we require at least a NDArray, a NDArray subclass or a CoordSet
            if not isinstance(coord, (Coord, CoordSet)):
                if isinstance(coord, NDArray):
                    coord = coords[k] = Coord(coord)
                else:
                    raise TypeError('Coordinates must be an instance or a subclass of Coord class or NDArray, or of '
                                    f' CoordSet class, but an instance of {type(coord)} has been passed')
            
            # This error is not one in previson of NDPanel (more coordinates than dims is possible
            # if self.dims and coord.name not in self.dims:
            #    raise AttributeError(f'The name of a coordinate must have name among the current dims: {self.dims}'
            #                         f' but the name is `{coord.name}`')
            
            if self.dims and coord.name in self.dims:
                # check the validity of the given coordinates in terms of size (if it correspond to one of the dims)
                size = coord.size
                
                if self.implements('NDDataset'):
                    idx = self._get_dims_index(coord.name)[0]  # idx in self.dims
                    if size != self._data.shape[idx]:
                        raise ValueError(
                        f'the size of a coordinates array must None or be equal'
                        f' to that of the respective `{coord.name}`'
                        f' data dimension but coordinate size={size} != data shape[{idx}]={self._data.shape[idx]}')
                else:
                    pass # bypass this checking for any other derived type (should be done in the subclass)
        
        coords._parent = self
        return coords
    
    # ------------------------------------------------------------------------------------------------------------------
    # Read only properties (not in the NDArray base class)
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @property
    def coordtitles(self):
        """
        list - list of the |Coord| titles.

        Read only property. Use set_coordtitle to eventually set titles

        """
        if self._coords is not None:
            return self._coords.titles
    
    # ..................................................................................................................
    @property
    def coordunits(self):
        """
        list - list of the |Coord| units
        
        Read only property. Use set_coordunits to eventually set units

        """
        if self._coords is not None:
            return self._coords.units
    
    # ..................................................................................................................
    @property
    def modified(self):
        """
        `Datetime` object - Date of modification (readonly property).

        """
        return self._modified
    
    # ..................................................................................................................
    @property
    def T(self):
        """
        |NDDataset| - Transposed array.

        The same object is returned if `ndim` is less than 2.

        """
        return self.transpose()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Mutable properties (not in the NDArray base class)
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @property
    def coords(self):
        """
        |CoordSet| instance - Contains the coordinates of the various
        dimensions of the dataset.
        
        It's a readonly property. Use set_coords to change one or more coordinates at once.

        """
        return self._coords
    
    # ..................................................................................................................
    @coords.setter
    def coords(self, coords):
        if isinstance(coords, CoordSet):
            self.set_coords(**coords)
        else:
            self.set_coords(coords)
    
    # ..................................................................................................................
    @property
    def data(self):
        """
        |ndarray|, The `data` array.

        If there is no data but labels, then the labels are returned instead of data.

        """
        return super().data
    
    # ..................................................................................................................
    @data.setter
    def data(self, data):
        # as we can't write super().data = data, we call _set_data
        # see comment in the data.setter of NDArray
        super()._set_data(data)
        
        if HAS_PANDAS:
            
            from pandas.core.generic import NDFrame
            
            if isinstance(data, NDFrame):  # and hasattr(self, "coords"):  # case of the NDDataset subclass
                coords = [Coord(item.values, title=item.name) for item in data.axes]
                self.set_coords(*coords)
    
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
        # There is no label for nd-dataset
        raise NotImplementedError
    
    # ------------------------------------------------------------------------------------------------------------------
    # hidden read_only properties
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @property
    def _dict_dims(self):
        _dict = {}
        for index, dim in enumerate(self.dims):
            if dim not in _dict:
                _dict[dim] = {'size': self.shape[index],
                              'coord': getattr(self, dim)}
        return _dict
    
    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    def add_coords(self, *args, **kwargs):
        
        if not args and not kwargs:
            # reset coordinates
            self._coords = None
            return
        
        if self._coords is None:
            self._coords = CoordSet(*args, **kwargs)
        else:
            self._coords._append(*args, **kwargs)
        
        if self._coords:
            # set a notifier to the updated traits of the CoordSet instance
            HasTraits.observe(self._coords, self._dims_update, '_updated')
            # force it one time after this initialization
            self._coords._updated = True
    
    # ..................................................................................................................
    def coord(self, dim='x'):
        """
        Returns the coordinates along the given dimension.

        Parameters
        ----------
        dim : int or str.
            A dimension index or name, default index = `x`.
            If an integer is provided, it is equivalent to the `axis` parameter for numpy array.

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
            error_(f'could not find this dimenson name: `{name}`')
            return None
    
    # ..................................................................................................................
    def delete_coords(self):
        """
        Delete all coordinate settings
        
        """
        self._coords = None
    
    # ..................................................................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `NDDataset`.
        
        Rather than isinstance(obj, NDDataset) use object.implements('NDDataset').
        
        This is useful to check type without importing the module
        
        """
        
        if name is None:
            return 'NDDataset'
        else:
            return name == 'NDDataset'
    
    # ..................................................................................................................
    def set_coords(self, *args, **kwargs):
        """
        Set one or more coordinates at once
        
        Warnings
        --------
        This method replace all existing coordinates
        
        See Also
        --------
        add_coords, set_coordtitles, set_coordunits

        """
        self._coords = None
        self.add_coords(*args, **kwargs)
    
    # ..................................................................................................................
    def set_coordtitles(self, *args, **kwargs):
        """
        Set titles of the one or more coordinates
        
        """
        self._coords.set_titles(*args, **kwargs)
    
    # ..................................................................................................................
    def set_coordunits(self, *args, **kwargs):
        """
        Set units of the one or more coordinates

        """
        self._coords.set_units(*args, **kwargs)
    
    # ..................................................................................................................
    # def set_real(self, axis):
    #    raise NotImplementedError
    
    # ..................................................................................................................
    # def set_complex(self):
    #    raise NotImplementedError
    
    # ..................................................................................................................
    # def set_quaternion(self):
    #    raise NotImplementedError
    
    # ..................................................................................................................
    @docstrings.dedent
    def sort(self, **kwargs):
        """
        Returns the dataset sorted along a given dimension
        (by default, the last dimension [axis=-1]) using the numeric or label values

        Parameters
        ----------
        dim : str or int, optional, default=-1
            dimension index or name along which to sort.
        pos : int , optional
            If labels are multidimensional  - allow to sort on a define
            row of labels : labels[pos]. Experimental : Not yet checked
        by : str among ['value', 'label'], optional, default=``value``.
            Indicate if the sorting is following the order of labels or
            numeric coord values.
        descend : `bool`, optional, default=`False`.
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
        
        # parameter for selecting the level of labels (default None or 0)
        pos = kwargs.pop('pos', None)
        
        # parameter to say if selection is done by values or by labels
        by = kwargs.pop('by', 'value')
        
        # determine which axis is sorted (dims or axis can be passed in kwargs)
        # it will return a tuple with axis and dim
        axis, dim = self.get_axis(**kwargs)
        if axis is None:
            axis, dim = self.get_axis(axis=0)
        
        # get the corresponding coordinates (remember the their order can be different form the order
        # of dimension  in dims. S we cannot jsut take the coord from the indice.
        coord = getattr(self, dim)  # get the coordinate using the syntax such as self.x
        
        descend = kwargs.pop('descend', None)
        if descend is None:
            # when non specified, default is False (except for reversed coordinates
            descend = coord.reversed
        
        #import warnings
        #warnings.simplefilter("error")
        
        indexes = []
        for i in range(self.ndim):
            if i == axis:
                if not coord.has_data:
                    # sometimes we have only label for Coord objects.
                    # in this case, we sort labels if they exist!
                    if coord.is_labeled:
                        by = 'label'
                    else:
                        # nothing to do for sorting
                        # return self itself
                        return self
                
                args = coord._argsort(by=by, pos=pos, descend=descend)
                setattr(new, dim, coord[args])
                indexes.append(args)
            else:
                indexes.append(slice(None))
        
        new._data = new._data[tuple(indexes)]
        if new.is_masked:
            new._mask = new._mask[tuple(indexes)]
        
        return new
    
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
        # make a copy of the original dims
        old = self.dims[:]
        
        # squeeze the data and determine which axis must be squeezed
        new, axis = super().squeeze(*dims, inplace=inplace, return_axis=True)
        
        if new._coords is not None:
            # if there are coordinates they have to be squeezed as well (remove
            # coordinate for the squeezed axis)
            
            for i in axis:
                dim = old[i]
                del new._coords[dim]
        
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
        
        import pandas as pd
        
        columns = [self.title]  # title
        data = [self.data.reshape(-1)]
        index = self._coords.to_index()
        
        return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)
    
    # ..................................................................................................................
    def to_xarray(self, **kwargs):
        """
        Convert a NDDataset instance to an `~xarray.DataArray` object
        ( the xarray library must be available )

        Parameters

        Returns
        -------
        object : a xarray.DataArray object


        """
        # Information about DataArray from the DataArray docstring
        #
        # Attributes
        # ----------
        # dims: tuple
        #     Dimension names associated with this array.
        # values: np.ndarray
        #     Access or modify DataArray values as a numpy array.
        # coords: dict-like
        #     Dictionary of DataArray objects that label values along each dimension.
        # name: str or None
        #     Name of this array.
        # attrs: OrderedDict
        #     Dictionary for holding arbitrary metadata.
        # Init docstring:
        # Parameters
        # ----------
        # data: array_like
        #     Values for this array. Must be an ``numpy.ndarray``, ndarray like,
        #     or castable to an ``ndarray``. If a self-described xarray or pandas
        #     object, attempts are made to use this array's metadata to fill in
        #     other unspecified arguments. A view of the array's data is used
        #     instead of a copy if possible.
        # coords: sequence or dict of array_like objects, optional
        #     Coordinates (tick labels) to use for indexing along each dimension.
        #     If dict-like, should be a mapping from dimension names to the
        #     corresponding coordinates. If sequence-like, should be a sequence
        #     of tuples where the first element is the dimension name and the
        #     second element is the corresponding coordinate array_like object.
        # dims: str or sequence of str, optional
        #     Name(s) of the data dimension(s). Must be either a string (only
        #     for 1D data) or a sequence of strings with length equal to the
        #     number of dimensions. If this argument is omitted, dimension names
        #     are taken from ``coords`` (if possible) and otherwise default to
        #     ``['dim_0', ... 'dim_n']``.
        # name: str or None, optional
        #     Name of this array.
        # attrs: dict_like or None, optional
        #     Attributes to assign to the new instance. By default, an empty
        #     attribute dictionary is initialized.
        # encoding: dict_like or None, optional
        #     Dictionary specifying how to encode this array's data into a
        #     serialized format like netCDF4. Currently used keys (for netCDF)
        #     include '_FillValue', 'scale_factor', 'add_offset', 'dtype',
        #     'units' and 'calendar' (the later two only for datetime arrays).
        #     Unrecognized keys are ignored.
        if not HAS_XARRAY:
            warnings.warn('Xarray is not available! This function can not be used',
                          SpectroChemPyWarning)
            return None
        
        import xarray as xr
        
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
        
        da.attrs['title'] = self.title
        
        return da

    # ..................................................................................................................
    def to_panel(self, **kwargs):
        """
        Transform the current |NDDataset| to a new |NDPanel| object
        
        Parameters
        ----------
        **kwargs : additional keyword arguments

        Returns
        -------
        object : A |NDPanel| object
        
        """
        import spectrochempy as scp
        return scp.NDPanel(self, **kwargs)
        
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
        new.history = f'Data transposed between dims: {dims}' if dims else ''
        return new
    
    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    def _cstr(self):
        # Display the metadata of the object and partially the data
        out = ''
        out += '         name: {}\n'.format(self.name)
        out += '       author: {}\n'.format(self.author)
        out += '      created: {}\n'.format(self._date)
        out += '     modified: {}\n'.format(self._modified) if (self.modified-self.date).seconds>1 else ''
        
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
            # the three escaped null characters are here to facilitate
            # the generation of html outputs
            hist = '\0\0\0{}\0\0\0\n'.format(hist.rstrip())
            out += hist
        
        out += '{}\n'.format(self._str_value().rstrip())
        out += '{}\n'.format(self._str_shape().rstrip()) if self._str_shape() else ''
        out += '{}\n'.format(self._str_dims().rstrip())
        
        if not out.endswith('\n'):
            out += '\n'
        out += '\n'
        
        if not self._html_output:
            return colored_output(out.rstrip())
        else:
            return out.rstrip()
    
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
        
        self._coords._html_output = self._html_output  # transfert the html flag if necessary: false by default
        
        txt = self._coords._cstr()
        txt = txt.rstrip()  # remove the trailing '\n'
        return txt
    
    _repr_dims = _str_dims
    
    # ------------------------------------------------------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------------------------------------------------------
    
    def _dims_update(self, change=None):
        # when notified that a coords names have been updated
        _ = self.dims  # fire an update
        #debug_('dims have been updated')
        
    
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
        
        # all the time -> update modified date
        self._modified = datetime.now()
        
        return


# ======================================================================================================================
# module function
# ======================================================================================================================

# make some NDDataset operation accessible from the spectrochempy API
# We want a slightly different docstring so we cannot just make:
#     func = NDDataset.func
#

copy = make_func_from(NDDataset.copy, first='dataset')
sort = make_func_from(NDDataset.sort, first='dataset')
squeeze = make_func_from(NDDataset.squeeze, first='dataset')
swapaxes = make_func_from(NDDataset.swapaxes, first='dataset')
transpose = make_func_from(NDDataset.transpose, first='dataset')
to_xarray = make_func_from(NDDataset.to_xarray, first='dataset')
to_dataframe = make_func_from(NDDataset.to_dataframe, first='dataset')
take = make_func_from(NDDataset.take, first='dataset')

__all__ += ['sort',
            'copy',
            'squeeze',
            'swapaxes',
            'transpose',
            'to_xarray',
            'to_dataframe',
            'take',
    ]

# The following operation act only on complex NDDataset
abs = make_func_from(NDDataset.abs, first='dataset')
conjugate = make_func_from(NDDataset.conjugate, first='dataset')  # defined in ndarray
conj = make_func_from(conjugate)
conj.__doc__ = "Short alias of `conjugate` "
set_complex = make_func_from(NDDataset.set_complex, first='dataset')
set_quaternion = make_func_from(NDDataset.set_quaternion, first='dataset')

__all__ += ['abs',
            'conjugate',
            'conj',
            'set_complex',
            'set_quaternion',
            ]

# ======================================================================================================================
# Set the operators
# ======================================================================================================================

set_operators(NDDataset, priority=100000)
