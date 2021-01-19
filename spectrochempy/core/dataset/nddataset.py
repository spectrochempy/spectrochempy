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

import textwrap
import warnings
import sys

import numpy as np
from traitlets import HasTraits, Instance, Bool, Float, validate, default
from traittypes import Array

from spectrochempy.core.project.baseproject import AbstractProject
from spectrochempy.core.dataset.ndarray import NDArray, DEFAULT_DIM_NAME
from spectrochempy.core.dataset.ndcomplex import NDComplexArray
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.ndmath import NDMath, _set_ufuncs, _set_operators
from spectrochempy.core.dataset.ndio import NDIO
from spectrochempy.core.dataset.ndplot import NDPlot
from spectrochempy.core import error_, warning_
from spectrochempy.utils import (colored_output, SpectroChemPyException, SpectroChemPyWarning, )

HAS_XARRAY = False
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    xr = None
    pass


# ======================================================================================================================
# NDDataset class definition
# ======================================================================================================================

class NDDataset(NDIO, NDPlot, NDMath, NDComplexArray):

    # coordinates
    _coordset = Instance(CoordSet, allow_none=True)

    # model data (e.g., for fit)
    _modeldata = Array(Float(), allow_none=True)

    # some setting for NDDataset
    _copy = Bool(False)
    _labels_allowed = Bool(False)  # no labels for NDDataset

    # dataset can be members of a project or panels.
    # we use the abstract class to avoid circular imports.
    _parent = Instance(AbstractProject, allow_none=True)

    # ------------------------------------------------------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------------------------------------------------------
    # ..................................................................................................................
    def __init__(self, data=None, coordset=None, coordunits=None, coordtitles=None, **kwargs):
        """
        The main N-dimensional dataset class used by |scpy|.

        The NDDataset is the main object use by SpectroChemPy. Like numpy ndarrays, NDDataset have the capability to be
        sliced, sorted and subject to mathematical operations. But, in addition, NDDataset may have units, can be masked
        and each dimensions can have coordinates also with units. This make NDDataset aware of unit compatibility, e.g.,
        for binary operation such as additions or subtraction or during the application of mathematical operations.
        In addition or in replacement of numerical data for coordinates, NDDataset can also have labeled coordinates
        where labels can be different kind of objects (strings, datetime, numpy nd.ndarray or othe NDDatasets, etc…).

        Parameters
        ----------
        data : array of floats
            Data array contained in the object. The data can be a list, a tuple, a |ndarray|, a ndarray-like,
            a |NDArray| or any subclass of |NDArray|. Any size or shape of data is accepted. If not given, an empty
            |NDArray| will be inited.
            At the initialisation the provided data will be eventually casted to a numpy-ndarray.
            If a subclass of |NDArray| is passed which already contains some mask, labels, or units, these elements will
            be used to accordingly set those of the created object. If possible, the provided data will not be copied
            for `data` input, but will be passed by reference, so you should make a copy of the `data` before passing
            them if that's the desired behavior or set the `copy` argument to True.
        coordset : An instance of |CoordSet|, optional
            `coords` contains the coordinates for the different dimensions of the `data`. if `coords` is provided,
            it must specified the `coord` and `labels` for all dimensions of the `data`.
            Multiple `coord`'s can be specified in an |CoordSet| instance for each dimension.
        coordunits : list, optional
            A list of units corresponding to the dimensions in the order of the coordset.
        coordtitles : list, optional
            A list of titles corresponding of the dimensions in the order of the coordset.
        **kwargs : dict
            See other parameters.

        Other Parameters
        ----------------
        dtype : str or dtype, optional, default=np.float64
            If specified, the data will be casted to this dtype, else the type of the data will be used.
        dims : list of chars, optional
            If specified the list must have a length equal to the number od data dimensions (ndim) and the chars must be
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
        dlabel :  str, optional
            Alias of `title`.
        meta : dict-like object, optional
            Additional metadata for this object. Must be dict-like but no
            further restriction is placed on meta.
        author : str, optional
            Name(s) of the author(s) of this dataset. BNy default, name of the computer note where this dataset is
            created.
        description : str, optional
            A optional description of the nd-dataset. A shorter alias is `desc`.
        history : str, optional
            A string to add to the object history.
        copy : bool, optional
            Perform a copy of the passed object. Default is False.

        See Also
        --------
        Coord : Explicit coordinates object.
        LinearCoord : Implicit coordinates objet.
        CoordSet : Set of coordinates.

        Notes
        -----
        The underlying array in a |NDDataset| object can be accessed through the `data` attribute, which will return
        a conventional |ndarray|.

        Examples
        --------
        Usage by an end-user

        >>> from spectrochempy import *
        >>> x = NDDataset([1, 2, 3])
        >>> print(x.data)  # doctest: +NORMALIZE_WHITESPACE
        [       1        2        3]
        """
        super().__init__(data, **kwargs)

        self._parent = None

        # eventually set the coordinates with optional units and title

        if isinstance(coordset, CoordSet):
            self.set_coordset(**coordset)

            if coordunits is not None:
                warning_('When a CoordSet object is passed to set the coordinates, coordunits are ignored. '
                         'To change units, do it in the passed CoordSet')

            if coordtitles is not None:
                warning_('When a CoordSet object is passed to set the coordinates, coordtitles are ignored. '
                         'To change titles, do it in the passed CoordSet')

        else:
            if coordset is None:
                coordset = [None] * self.ndim

            if coordunits is None:
                coordunits = [None] * self.ndim

            if coordtitles is None:
                coordtitles = [None] * self.ndim

            _coordset = []
            for c, u, t in zip(coordset, coordunits, coordtitles):
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

                _coordset.append(coord)

            if _coordset and set(_coordset) != {Coord()}:  # if they are no coordinates do nothing
                self.set_coordset(*_coordset)

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def __dir__(self):
        # WARNING: be carefull to keep the present order of the three first elements! Needed for save/load operations
        return ['dims', 'coordset', 'data', 'name', 'title', 'mask', 'units', 'meta', 'preferences', 'author',
                'description', 'history', 'date', 'modified', 'modeldata', 'origin', 'roi', 'offset'] + NDIO().__dir__()

    # ..................................................................................................................
    def __getitem__(self, items):

        saveditems = items

        # coordinate selection to test first
        if isinstance(items, str):
            try:
                return self._coordset[items]
            except Exception:
                pass

        # slicing
        new, items = super().__getitem__(items, return_index=True)

        if new is None:
            return None

        if self._coordset is not None:
            names = self._coordset.names  # all names of the current coordinates
            new_coords = [None] * len(names)
            for i, item in enumerate(items):
                # get the corresponding dimension name in the dims list
                name = self.dims[i]
                # get the corresponding index in the coordinate's names list
                idx = names.index(name)
                if self._coordset[idx].is_empty:
                    new_coords[idx] = Coord(None, name=name)
                elif isinstance(item, slice):
                    # add the slice on the corresponding coordinates on the dim to the new list of coordinates
                    if not isinstance(self._coordset[idx], CoordSet):
                        new_coords[idx] = self._coordset[idx][item]
                    else:
                        # we must slice all internal coordinates
                        newc = []
                        for c in self._coordset[idx]:
                            newc.append(c[item])
                        new_coords[idx] = CoordSet(*newc, name=name)

                elif isinstance(item, (np.ndarray, list)):
                    new_coords[idx] = self._coordset[idx][item]

            new.set_coordset(*new_coords, keepnames=True)

        new.history = f'Slice extracted: ({saveditems})'
        return new

    # ..................................................................................................................
    def __getattr__(self, item):
        # when the attribute was not found
        if item in ["__numpy_ufunc__", "interface", '_pytestfixturefunction', '__dataclass_fields__',
                    '_ipython_canary_method_should_not_exist_', '_baseclass', '_fill_value', '_ax_lines', '_axcb',
                    'clevels', '__wrapped__', 'coords', '__await__',
                    '__aiter__'] or '_validate' in item or '_changed' in item:
            # raise an error so that traits, ipython operation and more ... will be handled correctly
            raise AttributeError

        # syntax such as ds.x, ds.y, etc...

        if item[0] in self.dims or self._coordset:

            # look also properties
            attribute = None
            index = 0
            # print(item)
            if len(item) > 2 and item[1] == '_':
                attribute = item[1:]
                item = item[0]
                index = self.dims.index(item)

            if self._coordset:
                try:
                    c = self._coordset[item]
                    if isinstance(c, str) and c in self.dims:
                        # probaly a reference to another coordinate name
                        c = self._coordset[c]

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

        if key in DEFAULT_DIM_NAME:  # syntax such as ds.x, ds.y, etc...
            # Note the above test is important to avoid errors with traitlets
            # even if it looks redundant with the folllowing
            if key in self.dims:
                if self._coordset is None:
                    # we need to create a coordset first
                    self.set_coordset(dict((self.dims[i], None) for i in range(self.ndim)))
                idx = self._coordset.names.index(key)
                _coordset = self._coordset
                listcoord = False
                if isinstance(value, list):
                    listcoord = all([isinstance(item, Coord) for item in value])
                if listcoord:
                    _coordset[idx] = list(CoordSet(value).to_dict().values())[0]
                    _coordset[idx].name = key
                    _coordset[idx]._is_same_dim = True
                elif isinstance(value, CoordSet):
                    if len(value) > 1:
                        value = CoordSet(value)
                    _coordset[idx] = list(value.to_dict().values())[0]
                    _coordset[idx].name = key
                    _coordset[idx]._is_same_dim = True
                else:
                    _coordset[idx] = Coord(value, name=key)
                _coordset = self._valid_coordset(_coordset)
                self._coordset.set(_coordset)
            else:
                raise AttributeError(f'Coordinate `{key}` is not used.')
        else:
            super().__setattr__(key, value)

    # ..................................................................................................................
    def __eq__(self, other, attrs=None):
        attrs = self.__dir__()
        for attr in (
                'filename', 'preferences', 'name', 'description', 'history', 'date', 'modified', 'modeldata', 'origin',
                'roi', 'offset'):
            # these attibutes are not used for comparison (comparison based on data and units!)
            attrs.remove(attr)
        return super().__eq__(other, attrs)

    # ..................................................................................................................
    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return super().__hash__ + hash(self._coordset)

    # ------------------------------------------------------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @default('_coordset')
    def _coordset_default(self):
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
    @validate('_coordset')
    def _coordset_validate(self, proposal):
        coords = proposal['value']
        return self._valid_coordset(coords)

    def _valid_coordset(self, coords):
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
                        raise ValueError(f'the size of a coordinates array must be None or be equal'
                                         f' to that of the respective `{coord.name}`'
                                         f' data dimension but coordinate size={size} != data shape[{idx}]='
                                         f'{self._data.shape[idx]}')
                else:
                    pass  # bypass this checking for any other derived type (should be done in the subclass)

        coords._parent = self
        return coords

    # ..................................................................................................................
    @property
    def _dict_dims(self):
        _dict = {}
        for index, dim in enumerate(self.dims):
            if dim not in _dict:
                _dict[dim] = {'size': self.shape[index], 'coord': getattr(self, dim)}
        return _dict

    # ------------------------------------------------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def add_coordset(self, *coords, dims=None, **kwargs):
        """
        Add one or a set of coordinates from a dataset or panel.

        Parameters
        ----------
        *coords : iterable
            Coordinates object(s).
        dims : list
            Name of the coordinates.
        **kwargs : dict
            Keywords passed to the coordset.
        """
        if not coords and not kwargs:
            # reset coordinates
            self._coordset = None
            return

        if self._coordset is None:
            # make the whole coordset at once
            self._coordset = CoordSet(*coords, dims=dims, **kwargs)
        else:
            # add one coordinate
            self._coordset._append(*coords, **kwargs)

        if self._coordset:
            # set a notifier to the updated traits of the CoordSet instance
            HasTraits.observe(self._coordset, self._dims_update, '_updated')
            # force it one time after this initialization
            self._coordset._updated = True

    # ..................................................................................................................
    def coord(self, dim='x'):
        """
        Return the coordinates along the given dimension.

        Parameters
        ----------
        dim : int or str
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

        if self._coordset is None:
            return None

        # idx is not necessarily the position of the coordinates in the CoordSet
        # indeed, transposition may have taken place. So we need to retrieve the coordinates by its name
        name = self.dims[idx]
        if name in self._coordset.names:
            idx = self._coordset.names.index(name)
            return self._coordset[idx]
        else:
            error_(f'could not find this dimenson name: `{name}`')
            return None

    # ..................................................................................................................
    @property
    def coordset(self):
        """
        |CoordSet| instance.

        Contains the coordinates of the various dimensions of the dataset.
        It's a readonly property. Use set_coords to change one or more coordinates at once.
        """
        if self._coordset and all(c.is_empty for c in self._coordset):
            # all coordinates are empty, this is equivalent to None for the coordset
            return None
        return self._coordset

    # ..................................................................................................................
    @coordset.setter
    def coordset(self, coords):
        if isinstance(coords, CoordSet):
            self.set_coordset(**coords)
        else:
            self.set_coordset(coords)

    # ..................................................................................................................
    @property
    def coordtitles(self):
        """
        List of the |Coord| titles.

        Read only property. Use set_coordtitle to eventually set titles.
        """
        if self._coordset is not None:
            return self._coordset.titles

    # ..................................................................................................................
    @property
    def coordunits(self):
        """
        List of the |Coord| units.

        Read only property. Use set_coordunits to eventually set units.
        """
        if self._coordset is not None:
            return self._coordset.units

    # ..................................................................................................................
    @property
    def data(self):
        """
        The ``data`` array.

        If there is no data but labels, then the labels are returned instead of data.
        """
        return super().data

    # ..................................................................................................................
    @data.setter
    def data(self, data):
        # as we can't write super().data = data, we call _set_data
        # see comment in the data.setter of NDArray
        super()._set_data(data)

    # ..................................................................................................................
    def delete_coordset(self):
        """
        Delete all coordinate settings.
        """
        self._coordset = None

    # ..................................................................................................................
    def implements(self, name=None):
        """
        Check if the current object implements `NDDataset`.

        Rather than isinstance(obj, NDDataset) use object.implements('NDDataset').
        This is useful to check type without importing the module

        Parameters
        ----------
        name : str
            Name of the object class. If None, the function returns the class name.
            If name is given, it checks if it correspond to the current class name.

        Returns
        -------
        str or bool
            If name is given, a bool is returned
            If name is None, the classname is returned

        Examples
        --------
        >>> from spectrochempy import NDDataset, Coord
        >>> co = Coord([1., 2., 3.])
        >>> co.implements('NDDataset')
        False
        >>> co.implements('Coord')
        True
        >>> ds = NDDataset([1., 2., 3.])
        >>> ds.implements()
        'NDDataset'
        """

        if name is None:
            return 'NDDataset'
        else:
            return name == 'NDDataset'

    # ..................................................................................................................
    @property
    def labels(self):
        # not valid for NDDataset
        # There is no label for nd-dataset
        raise NotImplementedError

    # ..................................................................................................................
    @property
    def modeldata(self):
        """
        |ndarray| - models data.

        Data eventually generated by modelling of the data.
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
        |Project| instance

        The parent project of the dataset.
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

    # ..................................................................................................................
    def set_coordset(self, *args, **kwargs):
        """
        Set one or more coordinates at once.

        Warnings
        --------
        This method replace all existing coordinates.

        See Also
        --------
        add_coords, set_coordtitles, set_coordunits
        """
        self._coordset = None
        self.add_coordset(*args, dims=self.dims, **kwargs)

    # ..................................................................................................................
    def set_coordtitles(self, *args, **kwargs):
        """
        Set titles of the one or more coordinates.
        """
        self._coordset.set_titles(*args, **kwargs)

    # ..................................................................................................................
    def set_coordunits(self, *args, **kwargs):
        """
        Set units of the one or more coordinates.
        """
        self._coordset.set_units(*args, **kwargs)

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
    def sort(self, **kwargs):
        """
        Returns the dataset sorted along a given dimension.

        (by default, the last dimension [axis=-1]) using the numeric or label values.

        Parameters
        ----------
        dim : str or int, optional, default=-1
            dimension index or name along which to sort.
        pos : int , optional
            If labels are multidimensional  - allow to sort on a define
            row of labels : labels[pos]. Experimental : Not yet checked.
        by : str among ['value', 'label'], optional, default=``value``
            Indicate if the sorting is following the order of labels or
            numeric coord values.
        descend : `bool`, optional, default=`False`
            If true the dataset is sorted in a descending direction. Default is False  except if coordinates
            are reversed.
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        sorted_dataset
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

        # import warnings
        # warnings.simplefilter("error")

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
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        squeezed
            The input array, but with all or a subset of the
            dimensions of length 1 removed.

        Raises
        ------
        ValueError
            If `dim` is not `None`, and the dimension being squeezed is not
            of length 1.
        """
        # make a copy of the original dims
        old = self.dims[:]

        # squeeze the data and determine which axis must be squeezed
        new, axis = super().squeeze(*dims, inplace=inplace, return_axis=True)

        if axis is not None and new._coordset is not None:
            # if there are coordinates they have to be squeezed as well (remove
            # coordinate for the squeezed axis)

            for i in axis:
                dim = old[i]
                del new._coordset[dim]

        return new

    def expand_dims(self, dim=None):
        """
        Expand the shape of an array.

        Insert a new axis that will appear at the `axis` position in the expanded array shape.

        Parameters
        ----------
        dim : int or str
            Position in the expanded axes where the new axis (or axes) is placed.

        Returns
        -------
        result : ndarray
            View of `a` with the number of dimensions increased.

        See Also
        --------
        squeeze : The inverse operation, removing singleton dimensions
        """
        # TODO
    # ..................................................................................................................
    def swapdims(self, dim1, dim2, inplace=False):
        """
        Interchange two dimensions of a NDDataset.

        Parameters
        ----------
        dim1 : int
            First axis.
        dim2 : int
            Second axis.
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        swaped_dataset

        See Also
        --------
        transpose
        """

        new = super().swapdims(dim1, dim2, inplace=inplace)
        new.history = f'Data swapped between dims {dim1} and {dim2}'
        return new

    # ..................................................................................................................
    @property
    def T(self):
        """
        Transposed |NDDataset|.

        The same object is returned if `ndim` is less than 2.
        """
        return self.transpose()

    # ..................................................................................................................
    def take(self, indices, **kwargs):
        """
        Take elements from an array

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

    # # ................................................................................................................
    # def to_panel(self, **kwargs):
    #     """
    #     Transform the current |NDDataset| to a new |NDPanel| object
    #
    #     Parameters
    #     ----------
    #     **kwargs : additional keyword arguments
    #
    #     Returns
    #     -------
    #     object : A |NDPanel| object
    #
    #     """
    #     import spectrochempy as scp
    #
    #     return scp.NDPanel(self, **kwargs)

    def to_array(self):
        """
        Return a numpy masked array (i.e., other NDDataset attributes are lost.

        Examples
        ========
        >>> a = scp.to_array(dataset)

        equivalent to:

        >>> a = np.ma.array(dataset)
        or
        >>> a= dataset.masked_data
        """
        return np.ma.array(self)

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
        # Init docstring
        #
        # Parameters
        # ----------
        # data: array_like
        #     Values for this array. Must be an ``numpy.ndarray``, ndarray like,
        #     or castable to an ``ndarray``.
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
            warnings.warn('Xarray is not available! This function can not be used', SpectroChemPyWarning)
            return None

        x, y = self.x, self.y
        tx = x.title
        if y:
            ty = y.title
            da = xr.DataArray(np.array(self.data, dtype=np.float64), coords=[(ty, y.data), (tx, x.data)], )

            da.attrs['units'] = self.units
        else:
            da = xr.DataArray(np.array(self.data, dtype=np.float64), coords=[(tx, x.data)], )

            da.attrs['units'] = self.units

        da.attrs['title'] = self.title

        return da

    # ..................................................................................................................
    def transpose(self, *dims, inplace=False):
        """
        Permute the dimensions of a NDDataset.

        Parameters
        ----------
        dims : sequence of dimension indexes or names, optional
            By default, reverse the dimensions, otherwise permute the dimensions
            according to the values given.
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        transposed_array

        See Also
        --------
        swapdims : Interchange two dimensions of a NDDataset.
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
        out += '     modified: {}\n'.format(self._modified) if (self.modified - self.date).seconds > 1 else ''

        wrapper1 = textwrap.TextWrapper(initial_indent='', subsequent_indent=' ' * 15, replace_whitespace=True,
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
    def _loc2index(self, loc, dim=-1):
        # Return the index of a location (label or coordinates) along the dim
        # This can work only if `coords` exists.

        if self._coordset is None:
            raise SpectroChemPyException('No coords have been defined. Slicing or selection'
                                         ' by location ({}) needs coords definition.'.format(loc))

        coord = self.coord(dim)

        return coord._loc2index(loc)

    # ..................................................................................................................
    def _str_dims(self):
        if self.is_empty:
            return ''
        if len(self.dims) < 1 or not hasattr(self, "_coordset"):
            return ''
        if not self._coordset or len(self._coordset) < 1:
            return ''

        self._coordset._html_output = self._html_output  # transfert the html flag if necessary: false by default

        txt = self._coordset._cstr()
        txt = txt.rstrip()  # remove the trailing '\n'
        return txt

    _repr_dims = _str_dims

    # ------------------------------------------------------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------------------------------------------------------

    def _dims_update(self, change=None):
        # when notified that a coords names have been updated
        _ = self.dims  # fire an update  # debug_('dims have been updated')

    # ..................................................................................................................


# ======================================================================================================================
# module function
# ======================================================================================================================

# make some NDDataset operation accessible from the spectrochempy API
thismodule = sys.modules[__name__]

api_funcs = ['sort',
             'copy',
             'squeeze',
             'swapdims',
             'transpose',
             'to_array',
             'to_xarray',
             'take',
             'set_complex',
             'set_quaternion',
             'set_hypercomplex',
             'part',
             'to',
             'to_base_units',
             'to_reduced_units',
             'ito',
             'ito_base_units',
             'ito_reduced_units',
             'is_units_compatible',
             'remove_masks',

             ]

# todo: chack the fact that some function are defined also in ndmath
for funcname in api_funcs:
    setattr(thismodule, funcname, getattr(NDDataset, funcname))

    thismodule.__all__.append(funcname)

# ======================================================================================================================
# Set the operators
# ======================================================================================================================

_set_operators(NDDataset, priority=100000)
_set_ufuncs(NDDataset)
