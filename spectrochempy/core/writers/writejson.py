# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================


"""Plugin module to extend NDDataset with json export method.

"""

import json
import numpy as np


import matplotlib.pyplot as plt

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.utils.meta import Meta
from spectrochempy.units import Unit, Quantity
from spectrochempy.utils.qtfiledialogs import savedialog
from spectrochempy.core import debug_    # info_, error_, warning_
from spectrochempy.utils import json_serialiser, check_filename_to_save

__all__ = ['write_json']
__dataset_methods__ = __all__


def write_json(*args, **kwargs):
    """Writes a dataset in JSON format

    Parameters
    ----------
    dataset : |NDDataset|.
        The dataset to export with a JSON format
    filename : str, optional, default=None.
        Filename of the file to write.
        If not specified, open a dialog box except if `to_string` is True
    directory: str, optional, default=None.
        Where to save the file.
        If not specified, write in the current directory.
    to_string: bool, optional, default=False.
        If True, return a JSON string

    Returns
    -------
    string:
        a JSON string if to_string, else None

    Examples
    --------
    >>> X.write_json('myfile.json')

    The format is also infered from the filename extension

    >>> X.write('myfile.json')

    """

    dataset, filename = check_filename_to_save(*args, **kwargs)

    dic = dataset.to_json()

    # make the json string
    js = json.dumps(dic, default=json_serialiser)

    directory = kwargs.get('directory', None)
    filename = savedialog(filename=filename,
                            directory=directory,
                            filters="json (*.json) ;; All files (*)")
    if filename is None:
        # no filename from the dialogbox
        return

    with open(filename, 'wb') as f:
        f.write(js.encode('utf-8'))


