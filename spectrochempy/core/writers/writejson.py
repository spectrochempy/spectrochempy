# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================


"""Plugin module to extend NDDataset with json export method.

"""
import os
import json
import pickle
import base64
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.utils.meta import Meta
from spectrochempy.units import Unit, Quantity
from spectrochempy.utils import savefilename
from spectrochempy.core import debug_    # info_, error_, warning_

__all__ = ['write_json']

__dataset_methods__ = __all__


def json_serialiser(byte_obj):
   if isinstance(byte_obj, datetime):
       return {"isoformat": byte_obj.isoformat(), "__class__": str(byte_obj.__class__)}
   if isinstance(byte_obj, np.ndarray):
       #return {"ndarray":byte_obj.tolist(), "dtype": byte_obj.dtype.name}
       return {"serialized":base64.b64encode(pickle.dumps(byte_obj)).decode(), "__class__": str(byte_obj.__class__)}

   raise ValueError('No encoding handler for data type ' + type(byte_obj))


def write_json(*args, **kwargs):
    """Writes a dataset in JSON format

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset
    filename : `None`, `str`
        Filename of the file to write. If `None`: opens a dialog box to save files.
    directory: str, optional, default="".
        Where to save the file. If not specified, write in the current directory.

    Returns
    -------
    None

    Examples
    --------
    >>> X.write_json('myfile.json')

    The format is also infered from the filename extension

    >>> X.write('myfile.json')

    """
    debug_("writing json file")

    # filename will be given by a keyword parameter except if the first parameters is already the filename
    filename = kwargs.get('filename', None)

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(args[0], NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(args[0], str) and args[0] != '':
            filename = args[0]
    else:  # then the dataset is the first and the filename might be the second parameter:
        dataset = args[0]
        if isinstance(args[1], str) and args[0] != '':
            filename = args[1]

    directory = kwargs.get('directory', None)

    filename = savefilename(filename=filename,
                            directory=directory,
                            filters="json (*.json) ;; All files (*)")
    if filename is None:
        # no filename from the dialogbox
        return

    dic = {}
    objnames = dataset.__dir__()

    def _loop_on_obj(_names, obj=dataset, level=''):
        """Recursive scan on NDDataset objects"""

        for key in _names:
            val = getattr(obj, f"_{key}")

            if isinstance(val, np.ndarray):
                dic[level + key] = val

            elif isinstance(val, CoordSet):
                for v in val._coords:
                    _objnames = dir(v)
                    if isinstance(v, Coord):
                        _loop_on_obj(_objnames, obj=v, level=f"coord_{v.name}_")
                    elif isinstance(v, CoordSet):
                        _objnames.remove('coords')
                        _loop_on_obj(_objnames, obj=v, level=f"coordset_{v.name}_")
                        for vi in v:
                            _objnames = dir(vi)
                            _loop_on_obj(_objnames, obj=vi, level=f"coordset_{v.name}_coord_{vi.name[1:]}_")

            elif isinstance(val, Unit):
                dic[level + key] = str(val)

            elif isinstance(val, Meta):
                d = val.to_dict()
                # we must handle Quantities
                for k, v in d.items():
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            if isinstance(item, Quantity):
                                item = list(item.to_tuple())
                                if isinstance(item[0], np.ndarray):
                                    item[0] = item[0].tolist()
                                d[k][i] = tuple(item)
                dic[level + key] = d

            elif val is None:
                continue

            elif isinstance(val, dict) and key == 'axes':
                # do not save the matplotlib axes
                continue

            elif isinstance(val, (plt.Figure, plt.Axes)):
                # pass the figures and Axe
                continue

            else:
                dic[level + key] = val

    _loop_on_obj(objnames)

    with open(filename, 'wb') as f:
        js = json.dumps(dic, default=json_serialiser)
        f.write(js.encode('utf-8'))
