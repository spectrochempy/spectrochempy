# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""This module extend NDDataset with the import method for OPUS generated data files.

"""
__all__ = ['read_opus']
__dataset_methods__ = __all__

import io
import numpy as np
from datetime import datetime, timezone, timedelta

from numpy import linspace
from brukeropusreader.opus_parser import parse_data, parse_meta

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.readers.importer import docstrings, Importer, importermethod
from spectrochempy.core import debug_


# ======================================================================================================================
# Public functions
# ======================================================================================================================

# ......................................................................................................................
@docstrings.dedent
def read_opus(*args, **kwargs):
    """
    Open Bruker OPUS file(s) and eventually group them in a single dataset. Only Absorbance spectra are
    extracted ("AB" field). Returns an error if dimensions are incompatibles.

    Parameters
    -----------
    %(read_method.parameters.no_origin|csv_delimiter)s


    Returns
    --------
    out : NDDataset| or list of |NDDataset|
        The dataset or a list of dataset corresponding to a (set of) OPUS file(s).

    Examples
    ---------

    >>> import spectrochempy as scp

    Reading a single OPUS file  (providing a windows type filename relative to the default ``Datadir``)

    >>> scp.read_opus('irdata\\\\OPUS\\\\test.0000')
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative to the default ``Datadir``)
    Note that here read_opus is called as a classmethod of the NDDataset class

    >>> NDDataset.read_opus('irdata/OPUS/test.0000')
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata/OPUS')
    >>> p = folder / 'test.0000'
    >>> read_opus(p)
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Multiple files not merged (return a list of datasets). Note that a directory is specified

    >>> l = scp.read_opus('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS')
    >>> len(l)
    3
    >>> l[0]
    NDDataset: [float32] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read_opus('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS', merge=True)
    NDDataset: [float32] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword `merge`

    >>> scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS')
    NDDataset: [float32] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> l = scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS', merge=False)
    >>> len(l)
    3

    Read without a filename. This has the effect of opening a dialog for file(s) selection

    >>> scp.read_opus() # doctest: +ELLIPSIS
    ...

    Read in a directory (assume that only OPUS files are present in the directory
    (else we must use the generic `read` function instead)

    >>> l = scp.read_opus(directory='irdata/OPUS')
    >>> len(l)
    4

    Again we can use merge to stack all 4 spectra if thet have compatible dimensions.

    >>> scp.read_opus(directory='irdata/OPUS', merge=True)
    NDDataset: [float32] a.u. (shape: (y:4, x:2567))

    See Also
    --------
    read : Generic read method
    read_topspin, read_omnic, read_spg, read_spa, read_srs, read_csv, read_matlab, read_zip

    """

    kwargs['filetypes'] = ['Bruker OPUS files (*.[0-9]*)']
    kwargs['protocol'] = ['opus']
    importer = Importer()
    return importer(*args, **kwargs)


# ======================================================================================================================
# Private Functions
# ======================================================================================================================

# ......................................................................................................................
@importermethod
def _read_opus(*args, **kwargs):
    debug_('Bruker OPUS import')

    dataset, filename = args
    content = kwargs.get('content', None)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, 'rb')

    opus_data = _read_data(fid)

    # data
    try:
        npt = opus_data['AB Data Parameter']['NPT']
        data = opus_data["AB"][:npt]
        dataset.data = np.array(data[np.newaxis], dtype='float32')
    except KeyError:
        raise IOError(f"{filename} is not an Absorbance spectrum. It cannot be read with the `read_opus` import method")

    # xaxis
    fxv = opus_data['AB Data Parameter']['FXV']
    lxv = opus_data['AB Data Parameter']['LXV']
    xdata = linspace(fxv, lxv, npt)
    xaxis = Coord(xdata, title='Wavenumbers', units='cm^-1')

    # yaxis
    name = opus_data["Sample"]['SNM']
    acqdate = opus_data["AB Data Parameter"]["DAT"]
    acqtime = opus_data["AB Data Parameter"]["TIM"]
    gmt_offset_hour = float(acqtime.split('GMT')[1].split(')')[0])
    date_time = datetime.strptime(acqdate + '_' + acqtime.split()[0], '%d/%m/%Y_%H:%M:%S.%f')
    utc_dt = date_time - timedelta(hours=gmt_offset_hour)
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    timestamp = utc_dt.timestamp()
    yaxis = Coord([timestamp],
                  title='Acquisition timestamp (GMT)',
                  units='s',
                  labels=([utc_dt], [name]))

    # set dataset's Coordset
    dataset.set_coords(y=yaxis, x=xaxis)
    dataset.units = 'absorbance'
    dataset.title = 'Absorbance'

    # Set name, origin, description and history
    dataset.name = filename.name
    dataset.origin = "opus"
    dataset.description = 'Dataset from opus files. \n'
    dataset.history = str(datetime.now()) + ': import from opus files \n'
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    return dataset


# ......................................................................................................................
def _read_data(fid):
    data = fid.read()
    meta_data = parse_meta(data)
    opus_data = parse_data(data, meta_data)
    return opus_data


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
