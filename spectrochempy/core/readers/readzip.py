# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

__all__ = ['read_zip']
__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard and other imports
# ----------------------------------------------------------------------------------------------------------------------

import io

import numpy as np
from numpy.lib.npyio import NpzFile

# ----------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.core.readers.importer import _Importer

# ======================================================================================================================
# Public functions
# ======================================================================================================================

def read_zip(*args, **kwargs):
    """Open a zipped list of data files and set data/metadata in the
    current dataset

    Parameters
    ----------
    %(read_method.parameters)s

    Other Parameters
    ----------------
    %(read_method.other_parameters)s

    Returns
    -------
    out : NDDataset| or list of |NDDataset|
        The dataset or a list of dataset corresponding to a (set of) .zip file(s).

    Examples
    --------
    >>> from spectrochempy import NDDataset
    >>> A = NDDataset.read_zip('agirdata/P350/FTIR/FTIR.zip', only=50, origin='omnic')
    >>> print(A)
    NDDataset: [float64] a.u. (shape: (y:50, x:2842))

    See Also
    --------
    read : Generic read method
    read_csv, read_jdx, read_matlab, read_omnic, read_opus, read_topspin


    """
    if 'filetypes' not in kwargs.keys():
        kwargs['filetypes'] = ['Compressed files (*.zip)']
    # TODO: allows other type of compressed files
    kwargs['protocol'] = ['.zip']
    importer = _Importer()
    return importer(*args, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================

def _read_zip(*args, **kwargs):

    # Below we assume that files to read are in a unique directory
    from spectrochempy import NDDataset
    import zipfile

    # read zip file
    _ , filename = args
    content = kwargs.pop('content', None)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, 'rb')

    with zipfile.ZipFile(fid) as zf:

        filelist = zf.filelist
        only = kwargs.pop('only', len(filelist))

        datasets = []

        for file in zf.filelist:

            file = zipfile.Path(zf, at=file.filename)

            if file.name.startswith('__MACOSX'):
                continue  # bypass non-data files

            # seek the parent directory containing the files to read
            if not file.is_dir():
                continue

            parent = file
            break


        for count, children in enumerate(parent.iterdir()):

            if count == only:
                # limits to only this number of files
                break

            _ , extension = children.name.split('.')
            if extension == 'DS_Store':
                only += 1
                continue

            read_ = getattr(NDDataset, f"read_{extension}")

            datasets.append(read_(children.name, content=children.read_bytes(), **kwargs))

    return datasets


# Register the readers
# ----------------------------------------------------------------------------------------------------------------------
_Importer._read_zip = staticmethod(_read_zip)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
