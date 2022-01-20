# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Plugin module to extend NDDataset with a netCDF export method.
"""
# import numpy as np

from spectrochempy.optional import import_optional_dependency

# from spectrochempy.core.dataset.ndplot import PreferencesSet
from spectrochempy.core import info_

HAS_NETCDF4 = True
netcdf_file = import_optional_dependency("netCDF4.Dataset", errors="ignore")
if netcdf_file is None:
    HAS_NETCDF4 = False
    info_(
        "netCDF4 package was not found. Using Scipy.io instead but then "
        "writing is limited to netCDF3 format as scipy does not support "
        "netCDF4 format."
    )
    from scipy.io import netcdf_file

from spectrochempy.core.writers.exporter import Exporter, exportermethod

__all__ = ["write_netcdf"]
__dataset_methods__ = __all__


# ...............................................................................
def write_netcdf(dataset, filename, **kwargs):
    """
    Write a dataset or a list of datasets in netCDF format.

    See https://www.unidata.ucar.edu/software/netcdf/ for information about
    this format

    Parameters
    ----------
    filename: str or pathlib object, optional
        If not provided, a dialog is opened to select a file for writing
    directory : str, optional
        Where to write the specified `filename`. If not specified, write in the current directory.
    description: str, optional
        A Custom description.

    Returns
    -------
    out : `pathlib` object
        path of the saved file.

    Examples
    --------

    The extension `nc` will be added automatically
    >>> X.write_netcdf('myfile')
    """
    exporter = Exporter()
    kwargs["filetypes"] = ["NetCDF - Network Common Data Form (*.nc)"]
    kwargs["suffix"] = ".nc"
    return exporter(dataset, filename, **kwargs)


write_nc = write_netcdf
write_nc.__doc__ = "This method is an alias of `write_netcdf`."


@exportermethod
def _write_netcdf(*args, **kwargs):

    # raise NotImplementedError("WIP")

    # To create a NetCDF file:
    #
    # >>> from netCDF4 import Dataset
    # >>> rootgrp = Dataset("test.nc", "w", format="NETCDF4")
    # >>> print(rootgrp.data_model)
    # NETCDF4
    # >>> rootgrp.close()

    # f = netcdf_file('simple2.nc', 'w')
    # f.history = 'Created for a nD test'
    # f.createDimension('time', 10)
    # time = f.createVariable('time', 'f8', ('time',))
    # f.createDimension('wavenumbers', 20)
    # wave = f.createVariable('wavenumbers', 'f8', ('wavenumbers',))
    # time[:] = np.arange(10)
    # time.units = 'seconds'
    # wave[:] = np.arange(20)
    # wave.units = '1/centimeter'
    # data = f.createVariable("data", 'f8', ('time', 'wavenumbers'))
    # data.units = 'absorbance'
    # print(f)
    # f.close()
    #
    # f2 = netcdf_file('simple2.nc', 'r')
    # print(f2.history)
    # time = f2.variables['time']
    # print(time.units)
    # print(time.shape)
    # wave = f2.variables['wavenumbers']
    # print(wave.units)
    # print(wave.shape)
    # data = f2.variables['data']
    # print(data.units)
    # print(data.shape)
    # print(data.__repr__())
    # f2.close()

    dataset, filename = args
    dataset.filename = filename

    f = netcdf_file(filename, "w")
    f = _to_netCDF(f, dataset)
    f.close()


def _to_netCDF(cdf_obj, obj):

    if obj.implements("NDDataset"):

        # Get the name of the dimensions and their order
        dims = obj.dims  # this give the order of dimensions

        # Create the corresponding dimensions
        coords = []
        for index, dim in enumerate(dims):
            coord = getattr(obj, dim)
            cdf_obj.createDimension(coord.title, coord.size)
            dim_ = cdf_obj.createVariable(dim, "f8", (coord.title,))
            coords.append(coord.title)
            if coord.implements("LinearCoord"):
                dim_.linear = True
                dim_.offset = coord.offset
                dim_.increment = coord.increment
            else:
                dim_[:] = coord.data
            dim_.units = coord.units
            # dim_.labels = coord.labels  # TODO: handle labels and more (
            #  modeldata ....)
            if coord.meta:
                dim_.meta = coord.meta
            dim_.roi = coord.roi

        cdf_obj.history = obj.history
        cdf_obj.author = obj.author
        cdf_obj.description = obj.description
        cdf_obj.date = obj.date
        cdf_obj.modified = obj.modified
        cdf_obj.origin = obj.origin
        if obj.meta:
            cdf_obj.meta = obj.meta
        cdf_obj.roi = obj.roi
        data = cdf_obj.createVariable("data", "f8", coords)
        data[:] = obj.data
        # TODO: handle masks
        return cdf_obj

        # dataset __dir__

        # ['dims', 'coordset', 'data', 'name', 'title', 'mask', 'units',
        # 'meta', 'preferences', 'author', 'description', 'history',
        # 'date', 'modified', 'origin', 'roi', 'transposed',
        # 'modeldata', 'referencedata', 'state','ranges', 'filename']

        # coord : ['data', 'labels', 'units', 'meta', 'title', 'name', 'offset', 'increment', 'linear', 'roi']
