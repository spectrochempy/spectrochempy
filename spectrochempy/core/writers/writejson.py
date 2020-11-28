# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""Plugin module to extend NDDataset with json export method.

"""

import json

from spectrochempy.core import debug_
from spectrochempy.utils import json_serialiser
from spectrochempy.core.writers.exporter import docstrings, Exporter, exportermethod

__all__ = ['write_json']
__dataset_methods__ = __all__


# ......................................................................................................................
@docstrings.dedent
def write_json(*args, **kwargs):
    """
    Writes a dataset in JSON format

    Parameters
    ----------


    Returns
    -------
    out : `pathlib` object
        path of the saved file

    Examples
    --------
    >>> X.write_json('myfile.json')

    The format is also infered from the filename extension

    >>> X.write('myfile.json')

    """
    exporter = Exporter()
    kwargs['filetypes'] = ['JSON format(*.json)']
    kwargs['suffix'] = '.json'
    return exporter(*args, **kwargs)


@exportermethod
def _write_json(*args, **kwargs):
    # Writes a dataset in JSON format

    debug_("writing jcamp_dx file")

    dataset, filename = args
    dataset.filename = filename

    js = dataset.to_json(to_string=True)

    with open(filename, 'wb') as f:
        f.write(js.encode('utf-8'))

    return filename
