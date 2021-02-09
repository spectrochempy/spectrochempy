# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""Plugin module to extend NDDataset with the import methods method.
"""

__all__ = ['read_asc']
__dataset_methods__ = __all__

import io
from warnings import warn
from datetime import datetime, timezone
import re

import numpy as np
import scipy.io as sio

from spectrochempy.core.dataset.nddataset import NDDataset, Coord
from spectrochempy.core.readers.importer import Importer, importermethod

# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_quadera(*paths, **kwargs):
    """
    Read a Pfeiffer Vacuum's QUADERA® mass spectrometer software file with extension ``.asc`` and return its
    content as dataset.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_asc
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel', 'asc'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False)
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False)

    Examples
    ---------

    >>> scp.read_asc('ascdata/sample.asc')
    NDDataset: [float64] unitless (shape: (y:20, x:426))

    See Also
    --------
    read : Read generic files.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    """
    kwargs['filetypes'] = ['Quadera files (*.asc)']
    kwargs['protocol'] = ['asc']
    importer = Importer()
    return importer(*paths, **kwargs)

# ......................................................................................................................



# ----------------------------------------------------------------------------------------------------------------------
# Private methods
# ----------------------------------------------------------------------------------------------------------------------

@importermethod
def _read_quadera(*args, **kwargs):
    _, filename = args
    content = kwargs.get('content', False)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, 'rb')



    return datasets




# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
