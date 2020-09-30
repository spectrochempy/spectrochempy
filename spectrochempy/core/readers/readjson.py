# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""This module extend NDDataset with the import methods.

"""
__all__ = ['read_json']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import json
import base64
import io
from datetime import datetime
import pickle

import numpy as np

from spectrochempy.core import debug_
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.utils.meta import Meta
from spectrochempy.units import Unit, Quantity
from spectrochempy.core.dataset.ndio import NDIO
from spectrochempy.utils import readfilename, pathclean
from spectrochempy.core import general_preferences as prefs

def json_decoder(dic):
    if "__class__" in dic:
        if dic["__class__"] == str(datetime):
            return datetime.fromisoformat(dic["isoformat"])

        if dic["__class__"] == str(np.ndarray):
            return pickle.loads(base64.b64decode(dic['serialized']))

        raise TypeError("numpy array or datetime expected.")

    return dic

def read_json(dataset=None, **kwargs):
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
    >>> A = NDDataset.read_json('nh4.json')

    """
    debug_("reading a json file")

    # filename will be given by a keyword parameter except if the first parameters
    # is already the filename
    filename = kwargs.get('filename', None)

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, (str, list)) and dataset != '':
            filename = dataset

        dataset = NDDataset()  # create an instance of NDDataset

    content = kwargs.get('content', None)
    directory = pathclean(kwargs.get("directory", prefs.datadir))

    if content is not None:
        f = io.BytesIO(content)
    else:
        f = open(filename, 'rb')

    # read file content
    obj = json.loads(f.read(),object_hook=json_decoder)

    # interpret
    coords = None

    def setattributes(clss, key, val):
        # utility function to set the attributes

        if key in ['modified', 'date']:
            setattr(clss, f"_{key}", val)

        elif key == 'meta':
            # handle the case were quantity were saved
            for k, v in val.items():
                if isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, (list, tuple)):
                            try:
                                v[i] = Quantity.from_tuple(item)
                            except TypeError:
                                # not a quantity
                                pass
                    val[k] = v
            clss.meta.update(val)

        elif key == 'plotmeta':
            # handle the case were quantity were saved
            for k, v in val.items():
                if isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, (list, tuple)):
                            try:
                                v[i] = Quantity.from_tuple(item)
                            except TypeError:
                                # not a quantity
                                pass
                    val[k] = v
            clss.plotmeta.update(val)

        elif key in ['units']:
            setattr(clss, key, val)

        else:
            setattr(clss, f"_{key}", val)

    for key, val in list(obj.items()):

        if key.startswith('coord_'):
            if not coords:
                coords = {}
            els = key.split('_')
            dim = els[1]
            if dim not in coords.keys():
                coords[dim] = Coord()
            setattributes(coords[dim], els[2], val)

        elif key.startswith('coordset_'):
            els = key.split('_')
            dim = els[1]
            if key.endswith("is_same_dim"):
                setattributes(coords[dim], "is_same_dim", val)
            elif key.endswith("name"):
                setattributes(coords[dim], "name", val)
            elif key.endswith("references"):
                setattributes(coords[dim], "references", val)
            else:
                idx = "_" + els[3]
                setattributes(coords[dim][idx], els[4], val)
        else:
            setattributes(dataset, key, val)

    if filename:
        dataset._filename = filename

    if coords:
        dataset.set_coords(coords)

    return dataset
