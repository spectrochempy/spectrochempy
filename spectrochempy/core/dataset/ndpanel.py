# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
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
import matplotlib.pyplot as plt

try:
    import xarray as xr
    
    HAS_XARRAY = True
except:
    HAS_XARRAY = False

# ======================================================================================================================
# Local imports
# ======================================================================================================================

from ...core import log
from ...extern.orderedset import OrderedSet

from .ndarray import NDArray, DEFAULT_DIM_NAME
from .nddataset import NDDataset
from .ndcomplex import NDComplexArray
from .ndcoord import Coord
from .ndcoordset import CoordSet
from .ndio import NDIO


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
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args : |NDARRAY| or subclass of |NDARRAY| objects
            NDArray, NDComplexArray, NDDataset and/or Coord objects
            can be provided as arguments: Casting if possible to a NDArray
            or NDComplexArray type object will provide the underlining data,
            and if present the data coordinates.
            Units for data and coordinates will also be extracted.
            Compatibilities between the data is expected.
        kwargs


        Examples
        ========

        

        """
        
        datasets = kwargs.pop('datasets', args)
        merge = kwargs.pop('merge', True)
        align = kwargs.pop('align', False)
        
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
    # Private methods
    # ------------------------------------------------------------------------------------------------------------------
    
    def _set_data(self, data):
        # go to set_datasets
        if data:
            self._set_datasets(data)
    
    # ..................................................................................................................
    def _set_datasets(self, datasets, merge=True, align=False):
        
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
        if self._datasets is None:
            return 0
        
        else:
            dims = []
            for dataset in self._datasets:
                dims += dataset.dims
                
            return len(set(dims))
    
    @property
    def dims(self):
        """
        Dimension names
        
        Returns
        -------
        out: list
            A list with the current dimension names
            
        """
        dims = []
        for dataset in self._datasets.values():
            dims += list(dataset.dims)
            
        return sorted(list(set(dims)))
    
    @property
    def dim_properties(self):
        """
        Dimension properties
        
        Returns
        -------
        out: dict
            A dictionary with the current dimension properties
            
        """
        properties = {}
        for dataset in self._datasets.values():
            properties.update(dataset.dim_properties)
    
        return properties

    @property
    def names(self):
        """
        dataset names
        
        Returns
        -------
        out: list
            A list with the current dataset names
            
        """
        return sorted(list(self._datasets.keys()))
    
    
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
        out: dict
            Dictionary of NDDataset
            
        """
    
        return self._datasets

    @datasets.setter
    def datasets(self, datasets):
    
        self._set_datasets(datasets)
        
    # ------------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    def add_dataset(self, dataset, name=None, merge=True, align=False):
        """
        Add a single dataset (or coordinates) to the NDPanel
        
        Parameters
        ----------
        dataset
        name
        merge

        Returns
        -------

        """
        dataset = dataset.copy(keepname=True)
        
        if not isinstance(dataset, (NDPanel, NDDataset, CoordSet, Coord)):
            self._set_datasets(dataset)
            return
           
        if dataset.has_defined_name:
            name = dataset.name
            
        if name is None or name in self.names :
            name = f"data_{len(self._datasets)}"
        
        # handle the dimension names
        dims = dataset.dims
        coords = dataset.coords # get the dataset Coordset
        dataset.delete_coords() # datasets in panel have no internal coordinates
        
        def make_dim_and_coord(dim):
            new_dim_name = (OrderedSet(DEFAULT_DIM_NAME)-self._dims).pop()
            index = dataset._dims.index(dim)
            dataset._dims[index] = new_dim_name
            self._dims.insert(0, new_dim_name)
            if coords:
                coords[dim].name = new_dim_name
                self.add_coords(coords[new_dim_name])
            
        def can_merge(dataset, dim, merge):
            prop = self.dim_properties # current dim properties
            datasetprop = dataset.dim_properties
            if merge:
                # same dimension properties ?
                return datasetprop[dim] == prop[dim]
            return False
            
        for index, dim in enumerate(dims[::-1]):
    
            if dim in self.dim_properties:  # current dim properties
                # already in current dims
                
                if can_merge(dataset, dim, merge):
                    # can merge: so it's natural to use the same dimension
                    # nothing else to do as dim and coords are already set for this dimension
                    continue
                else:
                    # not the same dim or not merge allowed
                    # create a new one using the next available names
                    make_dim_and_coord(dim)
                    
            else:
                # not yet in self.dims, we can safely add it to the dims and coordinates
                self._dims.insert(0, dim)
                if coords:
                    coords[dim].name = dim
                    self.add_coords(coords[dim])

        self._datasets[name] = dataset.copy()
        


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
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }
        
        log.debug('changes in NDPanel: ' + change.name)
