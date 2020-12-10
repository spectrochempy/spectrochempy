# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""This module extend NDDataset with the import methods.

"""
__all__ = ['read_json']
__dataset_methods__ = __all__

import json
import io

from spectrochempy.core import debug_
from spectrochempy.utils import json_decoder
from spectrochempy.core.readers.importer import docstrings, Importer, importermethod


# ======================================================================================================================
# Public functions
# ======================================================================================================================

# ......................................................................................................................
@docstrings.dedent
def read_json(*args, **kwargs):
    """
    Read a single file or a byte string in JSON format

    Parameters
    ----------
    filename : `str`, optional
        Filename of the file to load.
    content : str, optional
        The optional contents of the file to be loaded as a binary string
    directory : str, optional
        From which directory to read the specified filename.
        If not specified, read in the defaults datadir.

    Returns
    -------
    dataset : |NDDataset|

    Examples
    --------
    >>> import spectrochempy as scp

    Read some data
    >>> nd = scp.read_omnic('wodger.spg')

    Now write it in JSON format
    >>> filename = nd.write_json()

    Check the existence of this file

    >>> assert filename.is_file()
    >>> assert filename.name == 'wodger.json'

    Read the JSON file

    >>> new_nd = scp.read_json('wodger.json')
    >>> assert new_nd == nd


    Remove this file
    >>> filename.unlink()

    """
    kwargs['filetypes'] = ['JSON files (*.json)']
    kwargs['protocol'] = ['json']
    importer = Importer()
    return importer(*args, **kwargs)


# ======================================================================================================================
# private functions
# ======================================================================================================================

@importermethod
def _read_json(*args, **kwargs):
    debug_("reading a json file")

    # read json file
    dataset, filename = args
    content = kwargs.get('content', None)

    if content is not None:
        fid = io.StringIO(content.decode("utf-8"))
    else:
        fid = open(filename, 'rb')

    js = json.loads(fid.read(), object_hook=json_decoder)

    dataset = type(dataset).from_json(js)
    dataset.filename = filename
    dataset.name = filename.stem

    return dataset
