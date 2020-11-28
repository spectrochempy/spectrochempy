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
from spectrochempy.utils import get_filename, json_decoder
from spectrochempy.core import general_preferences as prefs


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
    debug_("reading a json file")

    dataset, files = check_filename_to_open(*args, **kwargs)

    if not files:
        # open a file selector dialog
        directory = kwargs.get('directory', prefs.datadir)
        files = get_filename(directory=directory,
                             filetypes=['JSON files (*.json)'])

    if not files:
        # still, there is no files, we thus return nothing
        return None


    files = files['.json']  # select only file with the correct extension
    datasets = []
    for filename in files:
        f = open(filename, 'rb')
        js = json.loads(f.read(), object_hook=json_decoder)
        datasets.append(NDDataset.from_json(js))

    if len(datasets) == 1:
        return datasets[0]
    else:
        return datasets


def _read(f, dataset, filename=None):

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
