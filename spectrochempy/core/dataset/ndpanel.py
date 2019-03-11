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
from spectrochempy.utils import (SpectroChemPyWarning,
                                 get_user_and_node, set_operators, docstrings,
                                 make_func_from, )
from spectrochempy.extern.traittypes import Array
from spectrochempy.core.project.baseproject import AbstractProject
from spectrochempy.units import Unit, ur, Quantity
from spectrochempy.utils.meta import Meta
from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.dataset.ndcomplex import NDComplexArray
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndmath import NDMath
from spectrochempy.core.dataset.ndio import NDIO
from spectrochempy.core.dataset.ndplot import NDPlot
from spectrochempy.core.dataset.nddataset import NDDataset

from spectrochempy.core import log
from spectrochempy.utils import INPLACE


# ======================================================================================================================
# NDPanel class definition
# ======================================================================================================================

class NDPanel(
    NDIO,
    NDPlot,
    NDMath,
    HasTraits
):
    """
    A multi-dimensional array container.

    A dataset consists of ndarrays (|NDArray| objects), coordinates (|Coord|
    objets) and metadata.

    A dataset can contains heterogeneous data (e.g., 1D and ND arrays with various data units), but their
    dimensions must be compatible or subject to alignment.

    """

    # main data
    _datasets = Any()
    _coords = Instance(CoordSet, allow_none=True)
    _meta = Instance(Meta, allow_none=True)

    # main attributes
    _id = Unicode()
    _name = Unicode(allow_none=True)
    _date = Instance(datetime)
    _title = Unicode(allow_none=True)

    # some settings
    _copy = Bool()

    # ..................................................................................................................
    @validate('_datasets')
    def _datasets_validate(self, proposal):
    
        datasets = proposal['value']
    
        if datasets:
            # we assume a sequence of objects have been provided
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
        else:
            return {}
        
        for d in datasets:
            pass
        
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

        self._datasets = kwargs.get('datasets', args)
        self._coords = kwargs.get('coords', None)
        self._name = kwargs.get('name', None)
        self._meta = kwargs.get('meta', None)



    # ------------------------------------------------------------------------------------------------------------------
    # Read Only Properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def datasets(self):
        return self._datasets

    # ..................................................................................................................
    @property
    def coords(self):
        return self._coords
    
    # ..................................................................................................................
    @default('_id')
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    # ..................................................................................................................
    @property
    def id(self):
        """
        str - Object identifier (Readonly property).
        """
        return self._id


    # ------------------------------------------------------------------------------------------------------------------
    # Mutable Properties
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return self._id

    @name.setter
    def name(self, value):
        if value is not None:
            self._name = value

        # ..................................................................................................................
    @property
    def meta(self):
        """
        |Meta| instance object - Additional metadata.
        """
        return self._meta

    # ..................................................................................................................
    @meta.setter
    def meta(self, meta):
        # property.setter for meta
        if meta is not None:
            self._meta.update(meta)
            
    # ------------------------------------------------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------------------------------------------------

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
