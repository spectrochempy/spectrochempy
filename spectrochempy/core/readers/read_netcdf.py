# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module extend NDDataset with netCDF import methods.

It uses the CF convention : http://cfconventions.org/cf-conventions/cf-conventions.html

"""
__all__ = ["read_netcdf", "read_nc"]
__dataset_methods__ = __all__

# import io
# import re
# import numpy as np

# from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.readers.importer import Importer, importermethod

# from spectrochempy.utils.exceptions import deprecated
from spectrochempy.optional import import_optional_dependency


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_netcdf(*paths, **kwargs):
    """
    Open netCDF files with extension ``.nc``.

    Parameters
    ----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    --------
    read_jcamp
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For examples on how to use this feature, one can look in the ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    ---------
    read : Generic read method.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.
    read_nc: Alias of read_netcdf
    """
    kwargs["filetypes"] = ["Xarray / NetCDF - Network Common Data Form (*.nc)"]
    kwargs["protocol"] = ["netcdf"]
    importer = Importer()
    return importer(*paths, **kwargs)


read_nc = read_netcdf

# ======================================================================================================================
# private functions
# ======================================================================================================================


@importermethod
def _read_netcdf(*args, **kwargs):

    # read netcdf file
    dataset, filename = args
    _ = kwargs.get("content", None)

    # if content is not None:
    #     fid = io.StringIO(content.decode("utf-8"))
    # else:
    #     fid = open(filename, "r")

    xr = import_optional_dependency("xarray")
    xarr = xr.open_dataarray(filename)
    dataset = NDDataset._from_xarray(xarr)  # convert

    return dataset


# ..............................................................................
@importermethod
def _read_nc(*args, **kwargs):  # pragma: no cover
    return _read_netcdf(*args, **kwargs)


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
