# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie,
#  Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in
#  the root directory                         =
# ======================================================================================================================
"""
This module defines functions related to NDDataset alignment.
"""

__all__ = ["align"]
__dataset_methods__ = __all__

# import scipy.interpolate
import numpy as np

from spectrochempy.utils import MASKED, UnitsCompatibilityError, get_n_decimals
from spectrochempy.core import warning_, error_
from spectrochempy.core.dataset.coord import Coord


# ..........................................................................
def can_merge_or_align(coord1, coord2):
    """
    Check if two coordinates can be merged or aligned.

    Parameters
    ----------
    coord1, coord2 : |Coord|
        Coordinates to merge or align.

    Returns
    -------
    can_merge, can_align : tuple of bools
        Two flags about merge and alignment possibility.
    """
    if coord1 == coord2:
        # same coordinates
        can_merge = True  # merge is obvious
        can_align = True  # of course as it is the same coordinate

    else:
        # no the same coordinates
        can_merge = False  # we need to do alignment to merge
        # can align only if data exists, units compatibles, and title are
        # the same
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
    Align individual |NDDataset| along given dimensions using various methods.

    Parameters
    -----------
    dataset : |NDDataset|
        Dataset on which we want to salign other objects.
    *others : |NDDataset|
        Objects to align.
    dim : str. Optional, default='x'
        Along which axis to perform the alignment.
    dims : list of str, optional, default=None
        Align along all dims defined in dims (if dim is also
        defined, then dims have higher priority).
    method : enum ['outer', 'inner', 'first', 'last', 'interpolate'], optional, default='outer'
        Which method to use for the alignment.

        If align is defined :

        * 'outer' means that a union of the different coordinates is
        achieved (missing values are masked).
        * 'inner' means that the intersection of the coordinates is used.
        * 'first' means that the first dataset is used as reference.
        * 'last' means that the last dataset is used as reference.
        * 'interpolate' means that interpolation is performed relative to
        the first dataset.
    interpolate_method : enum ['linear','pchip']. Optional, default='linear'
        Method of interpolation to performs for the alignment.
    interpolate_sampling : 'auto', int or float. Optional, default='auto'

        * 'auto' : sampling is determined automatically from the existing data.
        * int :  if an integer values is specified, then the
          sampling interval for the interpolated data will be splitted in
          this number of points.
        * float : If a float value is provided, it determines the interval
        between the interpolated data.
    coord : |Coord|, optional, default=None
        Coordinates to use for alignment. Ignore those corresponding to the
        dimensions to align.
    copy : bool, optional, default=True
        If False then the returned objects will share memory with the
        original objects, whenever it is possible :
        in principle only if reindexing is not necessary.

    Returns
    --------
    aligned_datasets : tuple of |NDDataset|
        Same objects as datasets with dimensions aligned.

    Raises
    ------
    ValueError
        Issued when the dimensions given in `dim` or `dims` argument are not
        compatibles (units, titles, etc.).
    """
    # DEVELOPPER NOTE
    # There is probably better methods, but to simplify dealing with
    # LinearCoord, we transform them in Coord before treatment (going back
    # to linear if possible at the end of the process)

    # TODO: Perform an alignment along numeric labels
    # TODO: add example in docs

    # copy objects?
    copy = kwargs.pop("copy", True)

    # make a single list with dataset and the remaining object
    objects = [dataset] + list(others)

    # should we align on given external coordinates
    extern_coord = kwargs.pop("coord", None)
    if extern_coord and extern_coord.implements("LinearCoord"):
        extern_coord = Coord(extern_coord, linear=False, copy=True)

    # what's the method to use (by default='outer')
    method = kwargs.pop("method", "outer")

    # trivial cases where alignment is not possible or unecessary
    if not objects:
        warning_("No object provided for alignment!")
        return None

    if (
        len(objects) == 1
        and objects[0].implements("NDDataset")
        and extern_coord is None
    ):
        # no necessary alignment
        return objects

    # evaluate on which axis we align
    axis, dims = dataset.get_axis(only_first=False, **kwargs)

    # check compatibility of the dims and prepare the dimension for alignment
    for axis, dim in zip(axis, dims):

        # get all objets to align
        _objects = {}
        _nobj = 0

        for idx, object in enumerate(objects):

            if not object.implements("NDDataset"):
                error_(
                    f"Bad object(s) found: {object}. Note that only NDDataset "
                    f"objects are accepted "
                    f"for alignment"
                )
                return None

            _objects[_nobj] = {
                "obj": object.copy(),
                "idx": idx,
            }
            _nobj += 1

        _last = _nobj - 1

        # get the reference object (by default the first, except if method if
        # set to 'last'
        ref_obj_index = 0
        if method == "last":
            ref_obj_index = _last

        ref_obj = _objects[ref_obj_index]["obj"]

        # as we will sort their coordinates at some point, we need to know
        # if the coordinates need to be reversed at
        # the end of the alignment process
        reversed = ref_obj.coordset[dim].reversed
        if reversed:
            ref_obj.sort(descend=False, dim=dim, inplace=True)

        # get the coordset corresponding to the reference object
        ref_obj_coordset = ref_obj.coordset

        # get the coordinate for the reference dimension
        ref_coord = ref_obj_coordset[dim]

        # as we will sort their coordinates at some point, we need to know
        # if the coordinates need to be reversed at
        # the end of the alignment process
        reversed = ref_coord.reversed

        # prepare a new Coord object to store the final new dimension
        new_coord = ref_coord.copy()

        ndec = get_n_decimals(new_coord.data.max(), 1.0e-5)

        if new_coord.implements("LinearCoord"):
            new_coord = Coord(new_coord, linear=False, copy=True)

        # loop on all object
        for index, object in _objects.items():

            obj = object["obj"]

            if obj is ref_obj:
                # not necessary to compare with itself!
                continue

            if reversed:
                obj.sort(descend=False, dim=dim, inplace=True)

            # get the current objet coordinates and check compatibility
            coord = obj.coordset[dim]
            if coord.implements("LinearCoord") or coord.linear:
                coord = Coord(coord, linear=False, copy=True)

            if not coord.is_units_compatible(ref_coord):
                # not compatible, stop everything
                raise UnitsCompatibilityError(
                    "NDataset to align must have compatible units!"
                )

            # do units transform if necesssary so coords can be compared
            if coord.units != ref_coord.units:
                coord.ito(ref_coord)

            # adjust the new_cord depending on the method of alignement

            new_coord_data = set(np.around(new_coord.data, ndec))
            coord_data = set(np.around(coord.data, ndec))

            if method in ["outer", "interpolate"]:
                # in this case we do a union of the coords (masking the
                # missing values)
                # For method=`interpolate`, the interpolation will be
                # performed in a second step
                new_coord._data = sorted(coord_data | new_coord_data)

            elif method == "inner":
                # take only intersection of the coordinates
                # and generate a warning if it result something null or
                new_coord._data = sorted(coord_data & new_coord_data)

            elif method in ["first", "last"]:
                # we take the reference coordinates already determined as
                # basis (masking the missing values)
                continue

            else:
                raise NotImplementedError(f"The method {method} is unknown!")

        # Now perform alignment of all objects on the new coordinates
        for index, object in _objects.items():

            obj = object["obj"]

            # get the dim index for the given object
            dim_index = obj.dims.index(dim)

            # prepare slicing keys ; set slice(None) for the untouched
            # dimensions preceeding the dimension of interest
            prepend_keys = [slice(None)] * dim_index

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
            coord = obj.coordset[dim]
            coord_data = set(np.around(coord.data, ndec))

            dim_loc = new_coord._loc2index(sorted(coord_data))
            loc = tuple(prepend_keys + [dim_loc])

            new_obj._data = new_obj_data

            # mask all the data then unmask later the relevant data in
            # the next step

            if not new_obj.is_masked:
                new_obj.mask = MASKED
                new_obj.mask[loc] = False
            else:
                mask = new_obj.mask.copy()
                new_obj.mask = MASKED
                new_obj.mask[loc] = mask

            # set the data for the loc
            new_obj._data[loc] = obj.data

            # update the coordinates
            new_coordset = obj.coordset.copy()
            if coord.is_labeled:
                label_shape = list(coord.labels.shape)
                label_shape[0] = new_coord.size
                new_coord._labels = np.zeros(tuple(label_shape)).astype(
                    coord.labels.dtype
                )
                new_coord._labels[:] = "--"
                new_coord._labels[dim_loc] = coord.labels
            setattr(new_coordset, dim, new_coord)
            new_obj._coordset = new_coordset

            # reversed?
            if reversed:
                # we must reverse the given coordinates
                new_obj.sort(descend=reversed, dim=dim, inplace=True)

            # update the _objects
            _objects[index]["obj"] = new_obj

            if method == "interpolate":
                warning_(
                    "Interpolation not yet implemented - for now equivalent "
                    "to `outer`"
                )

        # the new transformed object must be in the same order as the passed
        # objects
        # and the missing values must be masked (for the moment they are defined to NaN

        for index, object in _objects.items():
            obj = object["obj"]
            # obj[np.where(np.isnan(obj))] = MASKED  # mask NaN values
            obj[
                np.where(np.isnan(obj))
            ] = 99999999999999.0  # replace NaN values (to simplify
            # comparisons)
            idx = int(object["idx"])
            objects[idx] = obj

            # we also transform into linear coord if possible ?
            # TODO:

    # Now return

    return tuple(objects)

    # if method == 'interpolate':  #  #     # reorders dataset and reference  # in ascending order  #     is_sorted
    # = False  #     if  # dataset.coordset(axis).reversed:  #         datasetordered =  # dataset.sort(axis,
    # descend=False)  #         refordered = ref.sort(  # refaxis, descend=False)  #         is_sorted = True  #
    # else:  #  # datasetordered = dataset.copy()  #         refordered = ref.copy()  #  #     try:  #
    # datasetordered.coordset(axis).to(  #     refordered.coordset(refaxis).units)  #     except:  #  #     raise
    # ValueError(  #             'units of the dataset and  #     reference axes on which interpolate are not
    # compatible')  #  #  #     oldaxisdata = datasetordered.coordset(axis).data  #  #     refaxisdata =
    # refordered.coordset(refaxis).data  # TODO: at the  #      end restore the original order  #  #     method =
    #  kwargs.pop(  #      'method', 'linear')  #     fill_value = kwargs.pop('fill_value',  #      np.NaN)  #  #
    #  if method == 'linear':  #         interpolator  #      = lambda data, ax=0: scipy.interpolate.interp1d(  #  #
    #  oldaxisdata, data, axis=ax, kind=method, bounds_error=False,  #             fill_value=fill_value,
    #  assume_sorted=True)  #  #     elif  #             method == 'pchip':  #         interpolator = lambda data,
    #             ax=0: scipy.interpolate.PchipInterpolator(  #  #             oldaxisdata, data, axis=ax,
    #             extrapolate=False)  #  #             else:  #         raise AttributeError(f'{method} is not a  #
    #             recognised option method for `align`')  #  #  #             interpolate_data = interpolator(
    #             datasetordered.data,  #             axis)  #     newdata = interpolate_data(refaxisdata)  #  #  #
    #             if datasetordered.is_masked:  #         interpolate_mask =  #             interpolator(
    #             datasetordered.mask, axis)  #         newmask  #             = interpolate_mask(refaxisdata)  #
    #             else:  #  #             newmask = NOMASK  #  #     # interpolate_axis =  #
    #             interpolator(datasetordered.coordset(axis).data)  #     #  #             newaxisdata =
    #             interpolate_axis(refaxisdata)  #  #             newaxisdata = refaxisdata.copy()  #  #     if
    #             method ==  #             'pchip' and not np.isnan(fill_value):  #         index =  #
    #             np.any(np.isnan(newdata))  #         newdata[index] =  #             fill_value  #  #
    #             index = np.any(np.isnan(  #             newaxisdata))  #         newaxisdata[index] = fill_value
    #  #     # create the new axis  #     newaxes = dataset.coords.copy()  #  #  newaxes[axis]._data = newaxisdata
    #     newaxes[axis]._labels =  #  np.array([''] * newaxisdata.size)  #  #     # transform the dataset  #
    #     inplace = kwargs.pop('inplace', False)  #  #     if inplace:  #  #     out = dataset  #     else:  #
    #     out = dataset.copy()  #  #  #     out._data = newdata  #     out._coords = newaxes  #     out._mask  #
    #     = newmask  #  #     out.name = dataset.name  #     out.title =  #     dataset.title  #  #     out.history
    #     = '{}: Aligned along dim {}  #     with respect to dataset {} using coords {} \n'.format(  #  #     str(
    #     dataset.modified), axis, ref.name, ref.coords[refaxis].title)  #  #     if is_sorted and out.coordset(
    #     axis).reversed:  #  #  out.sort(axis, descend=True, inplace=True)  #         ref.sort(  #  refaxis,
    #     descend=True, inplace=True)  #  # return out
