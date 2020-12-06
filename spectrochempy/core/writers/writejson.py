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

    Read some data from a TOPSPIN nmr file
    >>> import spectrochempy as scp
    >>> nd = scp.read_topspin( 'nmrdata/bruker/tests/nmr/topspin_1d', name = "NMR_1D")
    >>> filename1 = nd.write_json()
    >>> assert filename1.is_file()

    The format is also infered from the filename extension
    >>> filename2 = nd.write('myfile.json')
    >>> assert filename2.name == 'myfile.json'

    Check that we can read back a JSON file
    >>> new_nd = scp.read_json('myfile.json')
    >>> assert new_nd = nd

    Remove these files
    >>> filename1.unlink()
    >>> filename2.unlink()

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

    # make the json string
    dic = dataset.to_json()
    js = json.dumps(dic, default=json_serialiser, indent=2)

    with open(filename, 'wb') as f:
        f.write(js.encode('utf-8'))

    return filename
