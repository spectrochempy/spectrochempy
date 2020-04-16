# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module implements the |NDPanel| class.

"""

__all__ = ['NDPanel']

__dataset_methods__ = []

# ======================================================================================================================
# Standard python imports
# ======================================================================================================================

import itertools
import textwrap
from datetime import datetime
import warnings
import uuid

# ======================================================================================================================
# third-party imports
# ======================================================================================================================
import numpy as np
from traitlets import (HasTraits, Dict, Set, List, Union, Any,
                       Unicode, Instance, Bool, All,
                       Float, validate, observe, default, )

try:
    import xarray as xr
    
    HAS_XARRAY = True
except:
    HAS_XARRAY = False

# ======================================================================================================================
# Local imports
# ======================================================================================================================

from ...extern.orderedset import OrderedSet
from ..processors.align import can_merge_or_align
from .ndarray import DEFAULT_DIM_NAME
from .nddataset import NDDataset
from .ndcoord import Coord
from .ndcoordset import CoordSet
from .ndio import NDIO

from ...utils import colored_output

# ======================================================================================================================
# NDPanel class definition
# ======================================================================================================================

class NDPanel(
    NDDataset
):
    """
    A multi-dimensional array container.

    A NDPanel consists of ndarrays (|NDArray| objects), coordinates (|Coord|
    objets) and metadata.

    A NDPanel can contains heterogeneous data (e.g., 1D and ND arrays with various data units), but their
    dimensions must be compatible or subject to alignement.

    """
    
    _datasets = Any()
    
    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    def __init__(self, *datasets, merge=True, align='outer', **kwargs):
        """

        Parameters
        ----------
        *datasets : |NDARRAY| or subclass of |NDARRAY| objects
            NDArray, NDComplexArray, NDDataset and/or Coord objects
            can be provided as arguments : Casting if possible to a NDArray
            or NDComplexArray type object will provide the underlining data,
            and if present the data coordinates.
            Units for data and coordinates will also be extracted.
            Compatibilities between the data is expected.
        merge : bool, optional, default=True
            if set to True, dimensions are merged automatically when they are compatible
        align : str among [False, ‘outer’, ‘inner’, ‘first’, ‘last’, 'interpolate'], optional, default=False.
            by default there is no alignment of the dimensions, if they can not be merged, a new dimension name and
            eventually the associated coordinates are created for each different dimensions.
     
            If align is defined :
            
            * 'outer' means that a union of the different coordinates is achieved (missing values are masked)
            * 'inner' means that the intersection of the coordinates is used
            * 'first' means that the first dataset is used as reference
            * 'last' means that the last dataset is used as reference
            * 'interpolate' means that interpolation is performed relative to the first dataset.
        **kwargs : additional keyword arguments

        Warnings
        --------
        The dimension can be aligned only if they have compatible units and the same title.
        
        Examples
        --------

        

        """
        
        datasets = kwargs.pop('datasets', datasets)
        
        # options
        super().__init__(**kwargs)
        
        self._set_datasets(datasets, merge=merge, align=align)
    
    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    def __dir__(self):
        return ['datasets', 'dims', 'coords', 'meta', 'plotmeta',
                'name', 'title', 'description', 'history', 'date', 'modified'] + NDIO().__dir__()
    
    # ..................................................................................................................
    def __getitem__(self, item):
        
        # dataset names selection first
        if isinstance(item, str):
            try:
                dataset = self._datasets[item]
                if self._coords:
                    c = {}
                    for dim in dataset.dims:
                        c[dim] = self.coords[dim]
                    dataset.set_coords(c)
                return dataset
            
            except:
                pass
        
        return super().__getitem__(item)
    
    # ------------------------------------------------------------------------------------------------------------------
    # validators
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @validate('_datasets')
    def _datasets_validate(self, proposal):
        
        datasets = proposal['value']
        
        return datasets
    
    
    # ------------------------------------------------------------------------------------------------------------------
    # Default setting
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @default('_datasets')
    def _datasets_default(self):
        return {}
    
    # ..................................................................................................................
    @default('_dims')
    def _dims_default(self):
        return []
    
    # ------------------------------------------------------------------------------------------------------------------
    # Read Only Properties
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @property
    def data(self):
        """alias of datasets"""
        
        return self._datasets
    
    # ..................................................................................................................
    @property
    def ndim(self):
        """
        int - The number of dimensions present in the panel.
        
        """
        return len(self.dims)
    
    # ..................................................................................................................
    @property
    def dims(self):
        """
        Dimension names
        
        Returns
        -------
        out : list
            A list with the current dimension names
            
        """
        dims = []
        for dataset in self._datasets.values():
            dims += list(dataset.dims)
        
        return sorted(list(set(dims)))
    
    # ..................................................................................................................
    @property
    def names(self):
        """
        dataset names
        
        Returns
        -------
        out : list
            A list with the current dataset names
            
        """
        return sorted(list(self._datasets.keys()))

    # ------------------------------------------------------------------------------------------------------------------
    # hidden read_only properties
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @property
    def _dict_dims(self):
        _dict = {}
        for dataset in self._datasets.values():
            for index, dim in enumerate(dataset.dims):
                if dim not in _dict:
                    _dict[dim] = {'datasets': [dataset.name],
                                  'size': dataset.shape[index],
                                  'coord': getattr(self, dim)}
        return _dict

    # ------------------------------------------------------------------------------------------------------------------
    # Mutable Properties
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    @property
    def datasets(self):
        """
        Dictionary of NDDatasets contained in the current NDPanel
        
        Returns
        -------
        out : dict
            Dictionary of NDDataset
            
        """
        
        return self._datasets
    
    @datasets.setter
    def datasets(self, datasets):
        
        self._set_datasets(datasets)

    # ..................................................................................................................
    @property
    def is_empty(self):
        """
        bool - True if the `datasets` array is empty
        (Readonly property).
        """
        if len(self.datasets.items())==0:
            return True
    
        return False
    
    # ------------------------------------------------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def _str_value(self):
        prefix = ['']
        
        if self.is_empty:
            return '{}'.format(textwrap.indent('empty', ' ' * 9))

        out = f'         size: {len(self.names)} datasets\n'
        
        for name, dataset in self.datasets.items():
            out += f'       DATASET `{name}`'
            out += '{}\n'.format(dataset._str_value().replace('DATA','').rstrip())
            out += '{}\n'.format(dataset._str_shape().rstrip())
        return out.rstrip()

    # ..................................................................................................................
    def _repr_value(self):
        return type(self).__name__ + ': '
    
    # ..................................................................................................................
    def _str_shape(self):
        return ''

    # ..................................................................................................................
    def _repr_shape(self):
    
        if not self.is_empty:
            out = f"size: {len(self.names)} datasets"
        else:
            out = 'empty'
        return out
    
    # ..................................................................................................................
    def _set_data(self, data):
        # go to set_datasets
        if data:
            self._set_datasets(data)

    # ..................................................................................................................
    def _set_datasets(self, datasets, merge=True, align='outer'):
    
        if datasets:
            # we assume a sequence of objects have been provided
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
    
        for dataset in datasets:
        
            if isinstance(dataset, NDPanel):
                for name in dataset.names:
                    self.add_dataset(dataset[name], name=name, merge=merge, align=align)
        
            elif isinstance(dataset, (NDDataset, CoordSet, Coord)):
                self.add_dataset(dataset, merge=merge, align=align)
                continue
        
            else:
                # create a NDataset
                self.add_dataset(NDDataset(dataset), merge=merge, align=align)

    # ..................................................................................................................
    def _make_dim_and_coord(self, dataset, dim, new_name=True):
        if new_name:
            new_dim_name = (OrderedSet(DEFAULT_DIM_NAME) - self._dims).pop()
        else:
            new_dim_name = dim
        index = dataset._dims.index(dim)
        dataset._dims[index] = new_dim_name
        self._dims.insert(0, new_dim_name)
        if dataset.coords:
            dataset.coords[dim].name = new_dim_name
            self.add_coords(dataset.coords[new_dim_name])

    # ..................................................................................................................
    def _equal_dim_properties(self, this, other):
        
        if this['coord'] is None or other['coord'] is None:
            # no coordinates, or coord exists only in the dataset or in the panel:
            # we can merge if the size are equal
            # but not align as there is no coordinate information for one of the objects
            can_merge = (this['size'] == other['size'])   # merge is obvious if same size
            can_align = False   # we can not really align if there is no coord information onto base the alignment
            return can_merge, can_align
        
        return can_merge_or_align(this['coord'], other['coord'])

    # ..................................................................................................................
    @staticmethod
    def _set_dataset_name(dataset, dim, newdim):
        if newdim != dim:
            index = dataset._dims.index(dim)
            dataset._dims[index] = newdim
            
    # ..................................................................................................................
    def _do_merge_or_align(self, dataset, dim, merge, align):
    
        # get the dim information from the dataset to align
        prop = dataset._dict_dims[dim]
        
        # we must now check on which existing dimensions we can possibly merge or align

        # check existing dimensions in the Panel, starting with the dimension having the same name
        dims = OrderedSet()
        if dim in self.dims:
            dims.add(dim)
        dims = dims.union(self.dims)
        for curdim in dims:
            curprop = self._dict_dims[curdim]
            
            # Check if dimension properties are the same for merging or can be aligned
            can_merge, can_align = self._equal_dim_properties(prop, curprop)
            
            if can_merge:
                # yes
                # nothing else to do as the dim properties are the same and merge is allowed,
                # except to set the dim name
                self._set_dataset_name(dataset, dim, curdim)
                return True
            
            elif align is not None and can_align:
                # can not merge but alignment may be possible
                self._set_dataset_name(dataset, dim, curdim)
                _, dataset = self.align(dataset, dim=curdim, method=align)
                self._dataset_to_be_added = dataset
                return True
        
        # if it was possible to merge or align, return already happened: Thus return False
        return False

    def _valid_coords(self, coords):
        coords = super()._valid_coords(coords)
        # TODO add size checking
        return coords
    
    # ------------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------------
    
    # ..................................................................................................................
    def add_dataset(self, dataset, name=None, merge=True, align=False):
        """
        Add a single dataset (or coordinates) to the NDPanel
        
        Parameters
        ----------
        dataset : ndarray-like object
            The array must be any object that can be transformed into a NDDataset
        name : str, optional
            If not provided, a name will be created automatically
        merge : bool, optional, default=True
            Whether or not ot merge the dataset dimensions and eventually the corresponding coordinates
            with the existing one in the NDPanel
        align : bool, optional, default=False
            Whether or not to align compatible coordinates (compatible units and coordinates values)
        
        """
        self._dataset_to_be_added = dataset.copy(keepname=True)
        
        if not isinstance(self._dataset_to_be_added, (NDPanel, NDDataset, CoordSet, Coord)):
            self._set_datasets(self._dataset_to_be_added, merge=merge, align=align)
            return
        
        if self._dataset_to_be_added.has_defined_name:
            name = self._dataset_to_be_added.name
        
        if name is None or name in self.names:
            name = f"data_{len(self._datasets)}"
            self._dataset_to_be_added.name = name
        
        # handle the dimension names
        dims = self._dataset_to_be_added.dims
        
        for index, dim in enumerate(dims[::-1]):    # [::-1] is necessary to respect dataset dim orders
            
            if merge and self._do_merge_or_align(self._dataset_to_be_added, dim, merge, align):
                # merge allowed
                # if can merge or do alignement:  use the same dimension
                # nothing else to do as dim and coords are already set for this dimension
                # either naturally or inside the '_do_merge_or_align' function
                continue
            else:
                # can not merge or align, if this case create a new dimension name in the panel
                # or the dimension does not yet exist, in this case simply add it to the dims of the panel
                self._make_dim_and_coord(self._dataset_to_be_added, dim, new_name=(dim in self.dims))
        
        # datasets in panel have no internal coordinates, so delete it
        self._dataset_to_be_added.delete_coords()
        
        # eventually store the dataset
        self._datasets[name] = self._dataset_to_be_added.copy(keepname=True)
        self._dataset_to_be_added = None   # reset
        
    # ..................................................................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `NDPanel`.
        
        Rather than isinstance(obj, NDPanel) use object.implements('NDPanel').
        
        This is useful to check type without importing the module
        
        """
        if name is None:
            return 'NDPanel'
        else:
            return name == 'NDPanel'
    
    # ..................................................................................................................
    def to_dataframe(self):
        """
        Convert a NDPanel to a pandas DataFrame

        Returns
        -------

        """
