# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module defines functions related to NDDataset or NDPanel alignment.

"""

__all__ = ['align']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import scipy.interpolate
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------

from ...utils import NOMASK, MASKED, UnitsCompatibilityError
from ...extern.orderedset import OrderedSet
from .. import warning_, error_


# ..................................................................................................................
def can_merge_or_align(coord1, coord2):
    """
    Check if two coordinates can be merged or aligned
    
    Parameters
    ----------
    coord1, coord2 : |Coord|
        coordinates to merge or align

    Returns
    -------
    can_merge, can_align : tuple of bools
        Two flags about merge and alignment possibility
        
    """
    if (coord1 == coord2):
        # same coordinates
        can_merge = True  # merge is obvious
        can_align = True  # of course as it is the same coordinate
    
    else:
        # no the same coordinates
        can_merge = False # we need to do alignment to merge
        # can align only if data exists, units compatibles, and title are the same
        can_align = True
        can_align &= not coord1.is_empty
        can_align &= not coord2.is_empty
        can_align &= coord1.title == coord2.title
        if can_align and (coord1.has_units or coord2.has_units):
            if coord1.has_units:
                can_align &= coord1.is_units_compatible(coord2)
            else:
                can_align &= coord2.is_units_compatible(coord1)
    
    return can_merge, can_align
    
# ............................................................................
def align(dataset, *others, **kwargs):
    """
    Align individual |NDDataset| or |NDPanel| along given dimensions using various methods.

    Parameters
    -----------
    dataset : |NDDataset|.
        Dataset on which we want to salign other objects
    *others : |NDDataset| or |NDPanel|.
        Objects to align.
        If NDPanel objects are provided, internal datasets will be aligned along the given dimension.
        Aligning more than one panel is not implemented.
    dim : str. Optional, default='x'.
        Along which axis to perform the alignment.
    dims : list of str, optional, default=None
        Align along all dims defined in dims (if dim or axis is also defined, then dims have higher priority).
    method : enum ['outer', 'inner', 'first', 'last', 'interpolate'], optional, default='outer'
        Which method to use for the alignment.
        
        If align is defined :
        
        * 'outer' means that a union of the different coordinates is achieved (missing values are masked)
        * 'inner' means that the intersection of the coordinates is used
        * 'first' means that the first dataset is used as reference
        * 'last' means that the last dataset is used as reference
        * 'interpolate' means that interpolation is performed relative to the first dataset.
    interpolate_method : enum ['linear','pchip']. Optional, default='linear'.
        Method of interpolation to performs for the alignment.
    interpolate_sampling : 'auto', int or float. Optional, default='auto'.
    
        * 'auto' : sampling is determined automatically from the existing data.
        * int :  if an integer values is specified, then the
          sampling interval for the interpolated data will be splitted in this number of points.
        * float : If a float value is provided, it determines the interval between the interpolated data
    coord : |Coord|, optional, default=None
        coordinates to use for alignment. Ignore those corresponding to the dimensions to align
    copy : bool, optional, default=True
        If False then the returned objects will share memory with the original objects, whenever it is possible :
        in principle only if reindexing is not necessary.
    
    Returns
    --------
    aligned_datasets : tuple of |NDDataset| or a |NDPanel|.
        Same objects as datasets with dimensions aligned

    Raises
    ------
    ValueError
        issued when the dimensions given in `dim` or `dims` argument are not compatibles (units, titles, etc...)
    
    """
    # TODO: Perform an alignment along numeric labels
    # TODO: add example in docs
    
    # copy objects?
    copy = kwargs.pop('copy', True)
   
    # make a single list with dataset and the remaining object
    objects = [dataset] + list(others)
    
    # should we align on given external coordinates
    extern_coord = kwargs.pop('coord', None)
    
    # what's the method to use (by default='outer')
    method = kwargs.pop('method','outer')
    
    # trivial cases where alignment is not possible or unecessary
    if not objects:
        warning_('No object provided for alignment!')
        return None

    if len(objects)==1 and objects[0].implements('NDDataset') and extern_coord is None:
        # no necessary alignment
        return objects
    
    # get all objets to align
    _objects = {}
    _nobj = 0
    
    for idx, object in enumerate(objects):
        
        if not object.implements('NDDataset') and not object.implements('NDPanel'):
            error_(f'Bad object(s) found: {object}. Note that only NDDataset or NDPanel objects are accepted '
                   f'for alignment')
            return None
        
        if object.implements('NDPanel'):
            
            for k,v in object.datasets.items():
                # set the coordset into the NDDataset object (temporary: this will be unset at the end)
                v.coords = object.coords
                _objects[_nobj]={'obj':v, 'idx':idx, 'is_panel':True, 'key':k}
                _nobj += 1

        else:
            _objects[_nobj]={'obj':object, 'idx':idx, 'is_panel':False}
            _nobj += 1
    
    _last = _nobj-1
    
    # get the reference object (by default the fist, except if method if set  to 'last'
    ref_obj_index = 0
    if method == 'last':
        ref_obj_index = _last
    ref_obj = _objects[ref_obj_index]['obj']
    
    # get the relevant dimension names
    ref_dims = ref_obj.dims
    axis = kwargs.pop('axis', -1)
    dim = kwargs.pop('dim', ref_dims[axis])
    dims = kwargs.pop('dims', [dim])
    
    # check compatibility of the dims and prepare the dimension for alignment
    for dim in dims:
        
        # as we will sort their coordinates at some point, we need to know if the coordinates need to be reversed at
        # the end of the alignment process
        reversed = ref_obj.coords[dim].reversed
        if reversed:
            ref_obj.sort(descend = False, dim=dim, inplace=True)
            
        # get the coordset corresponding to the reference object
        ref_obj_coords = ref_obj.coords
        
        # get the coordinate for the reference dimension
        ref_coord = ref_obj_coords[dim]

        # as we will sort their coordinates at some point, we need to know if the coordinates need to be reversed at
        # the end of the alignment process
        reversed = ref_coord.reversed
        
        # prepare a new Coord object to store the final new dimenson
        new_coord = ref_coord.copy()
        
        # loop on all object
        for index, object in _objects.items():
            
            obj = object['obj']
            
            if obj is ref_obj:
                # not necessary to compare with itself!
                continue

            if reversed:
                obj.sort(descend = False, dim=dim, inplace=True)

            # get the current objet coordinates and check compatibility
            coord = obj.coords[dim]
            if not coord.is_units_compatible(ref_coord):
                # not compatible, stop everything
                raise UnitsCompatibilityError('NDataset to align must have compatible units!')

            # do units transform if necesssary so coords can be compared
            if coord.units != ref_coord.units:
                coord.ito(ref_coord)

            # adjust the new_cord depending on the method of alignement
            new_coord_data = set(new_coord.data)
            coord_data = set(coord.data)
            
            if method in ['outer','interpolate']:
                # in this case we do a union of the coords (masking the missing values)
                # For method=`interpolate`, the interpolation will be performed in a second step
                new_coord._data = sorted(coord_data | new_coord_data)
                
            elif method == 'inner':
                # take only intersection of the coordinates
                # and generate a warning if it result something null or
                new_coord._data = sorted(coord_data & new_coord_data)
            
            elif method in ['first','last'] :
                # we take the reference coordinates already determined as basis (masking the missing values)
                continue
                
            else:
                raise NotImplementedError(f'The method {method} is unknown!')
        
        # Now perform alignment of all objects on the new coordinates
        for index, object in _objects.items():
            
            obj = object['obj']
            
            # get the dim index for the given object
            dim_index = obj.dims.index(dim)
            
            # prepare slicing keys ; set slice(None) for the untouched dimensions preceeding the dimension of interest
            prepend_keys = [slice(None)]*dim_index

            # New objects for obj must be created with the new coordinates
            
            # change the data shape
            new_obj_shape = list(obj.shape)
            new_obj_shape[dim_index] = len(new_coord)
            new_obj_data = np.full(new_obj_shape, np.NaN)
        
            # create new dataset for obj and ref_objects
            if copy:
                new_obj = obj.copy()
            else:
                new_obj = obj
            
            # update the data and mask
            coord = obj.coords[dim]
            coord_data = set(coord.data)
            
            dim_loc = new_coord._loc2index(sorted(coord_data))
            loc = tuple(prepend_keys+[dim_loc])
            
            new_obj.data = new_obj_data
            #new_obj.mask = MASKED  # mask all the data -we will unmask later the relevant data in the next step
            #new_obj[loc] = False   # remove the mask for the selected part of the array
            new_obj.data[loc] = obj.data
        
            # update the coordinates
            new_coords = obj.coords.copy()
            if coord.is_labeled:
                label_shape = list(coord.labels.shape)
                label_shape[0] = new_coord.size
                new_coord._labels = np.zeros(tuple(label_shape)).astype(coord.labels.dtype)
                new_coord._labels[:] = '--'
                new_coord._labels[dim_loc] = coord.labels
            setattr(new_coords, dim, new_coord)
            new_obj._coords = new_coords
            
            
            # reversed?
            if reversed:
                # we must reverse the given coordinates
                new_obj.sort(descend = reversed, dim=dim, inplace=True)
                
            # update the _objects
            _objects[index]['obj'] = new_obj
            
            if method == 'interpolate':
                warning_('Interpolation not yet implemented - for now equivalent to `outer`')
          
    # now return the new transformed object in the same order as the passed objects
    # and mask the maissing values (for the moment they are defined to NaN
    
    for index, object in _objects.items():
        is_panel = object['is_panel']
        obj = object['obj']
        obj[np.where(np.isnan(obj))]=MASKED  # mask NaN values
        obj[np.where(np.isnan(obj))]=99999999999999.  # replace NaN values (to simplify comparisons)
        idx = int(object['idx'])
        if not is_panel:
            objects[idx] = obj
        else:
            key = object['key']
            coords = obj.coords
            obj.delete_coords()  # NDDataset in NDPanel are stored without coords
            objects[idx].datasets[key] =  obj
            for dim in obj.dims:
                setattr(objects[idx], dim, getattr(coords, dim))
    
    return  tuple(objects)
                    
                    

    if method == 'interpolate':

        # reorders dataset and reference in ascending order
        is_sorted = False
        if dataset.coords(axis).reversed:
            datasetordered = dataset.sort(axis, descend=False)
            refordered = ref.sort(refaxis, descend=False)
            is_sorted = True
        else:
            datasetordered = dataset.copy()
            refordered = ref.copy()
    
        try:
            datasetordered.coords(axis).to(refordered.coords(refaxis).units)
        except:
            raise ValueError(
                'units of the dataset and reference axes on which interpolate are not compatible')
    
        oldaxisdata = datasetordered.coords(axis).data
        refaxisdata = refordered.coords(refaxis).data  # TODO: at the end restore the original order
    
        method = kwargs.pop('method', 'linear')
        fill_value = kwargs.pop('fill_value', np.NaN)
    
        if method == 'linear':
            interpolator = lambda data, ax=0: scipy.interpolate.interp1d(
                oldaxisdata, data, axis=ax, kind=method, bounds_error=False, fill_value=fill_value, assume_sorted=True)
    
        elif method == 'pchip':
            interpolator = lambda data, ax=0: scipy.interpolate.PchipInterpolator(
                oldaxisdata, data, axis=ax, extrapolate=False)
        else:
            raise AttributeError(f'{method} is not a recognised option method for `align`')
    
        interpolate_data = interpolator(datasetordered.data, axis)
        newdata = interpolate_data(refaxisdata)
    
        if datasetordered.is_masked:
            interpolate_mask = interpolator(datasetordered.mask, axis)
            newmask = interpolate_mask(refaxisdata)
        else:
            newmask = NOMASK
    
        # interpolate_axis = interpolator(datasetordered.coords(axis).data)
        # newaxisdata = interpolate_axis(refaxisdata)
        newaxisdata = refaxisdata.copy()
    
        if method == 'pchip' and not np.isnan(fill_value):
            index = np.any(np.isnan(newdata))
            newdata[index] = fill_value
    
            index = np.any(np.isnan(newaxisdata))
            newaxisdata[index] = fill_value
    
        # create the new axis
        newaxes = dataset.coords.copy()
        newaxes[axis]._data = newaxisdata
        newaxes[axis]._labels = np.array([''] * newaxisdata.size)
    
        # transform the dataset
        inplace = kwargs.pop('inplace', False)
    
        if inplace:
            out = dataset
        else:
            out = dataset.copy()
    
        out._data = newdata
        out._coords = newaxes
        out._mask = newmask
    
        out.name = dataset.name
        out.title = dataset.title
    
        out.history = '{}: Aligned along dim {} with respect to dataset {} using coords {} \n'.format(
            str(dataset.modified), axis, ref.name, ref.coords[refaxis].title)
    
        if is_sorted and out.coords(axis).reversed:
            out.sort(axis, descend=True, inplace=True)
            ref.sort(refaxis, descend=True, inplace=True)

    return out
