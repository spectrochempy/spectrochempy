# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""This module extend NDDataset with the import methods.

"""
__all__ = ['read_json', 'from_json']
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
from spectrochempy.utils import get_filename, json_decoder
from spectrochempy.core import general_preferences as prefs
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
    >>> A = NDDataset.read_json('nh4.json')

    """
    kwargs['filetypes'] = ['JSON files (*.json)']
    kwargs['protocol'] = ['json']
    importer = Importer()
    return importer(*args, **kwargs)


@docstrings.dedent
def from_json(dic):
    """
    Transform a JSON object into a NDDataset

    Parameters
    ----------
    args
    kwargs

    Returns
    -------

    """

    dataset = NDDataset()
    return _from_json(dataset, dic)


# ======================================================================================================================
# private functions
# ======================================================================================================================

@importermethod
def _read_json(*args, **kwargs):
    debug_("reading a json file")

    # read jdx file
    dataset, filename = args
    content = kwargs.get('content', None)

    if content is not None:
        fid = io.StringIO(content.decode("utf-8"))
    else:
        fid = open(filename, 'rb')

    js = json.loads(fid.read(), object_hook=json_decoder)

    dataset = _from_json(dataset, js)
    dataset.filename = filename
    dataset.name = filename.stem

    return dataset


def _from_json(dataset, obj):
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

    if coords:
        dataset.set_coords(coords)

    return dataset
