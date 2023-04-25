# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module extend NDDataset with the import method for OPUS generated data files.
"""
__all__ = ["read_opus"]
__dataset_methods__ = __all__

from datetime import datetime, timedelta, timezone

import numpy as np
from brukeropusreader.opus_parser import parse_data, parse_meta

from spectrochempy.core import debug_
from spectrochempy.core.dataset.coord import Coord, LinearCoord
from spectrochempy.core.readers.importer import Importer, _importer_method, _openfid
from spectrochempy.utils.docstrings import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_opus")


@_docstring.dedent
def read_opus(*paths, **kwargs):
    """
    Open Bruker OPUS file(s).

    Eventually group them in a single dataset. Only Absorbance spectra are
    extracted ("AB" field). Returns an error if dimensions are incompatibles.

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    --------
    %(Importer.returns)s

    Other Parameters
    ----------------
    %(Importer.other_parameters)s

    See Also
    ---------
    %(Importer.see_also.no_read_opus)s

    Examples
    ---------
    Reading a single OPUS file  (providing a windows type filename relative to
    the default `datadir` )

    >>> scp.read_opus('irdata\\\\OPUS\\\\test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative to the
    default `datadir` )
    Note that here read_opus is called as a classmethod of the NDDataset class

    >>> scp.NDDataset.read_opus('irdata/OPUS/test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata/OPUS')
    >>> p = folder / 'test.0000'
    >>> scp.read_opus(p)
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files not merged (return a list of datasets). Note that a directory is
    specified

    >>> le = scp.read_opus('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS')
    >>> len(le)
    3
    >>> le[0]
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read_opus('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword `
    merge`

    >>> scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS')
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> le = scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS', merge=False)
    >>> len(le)
    3

    Read without a filename. This has the effect of opening a dialog for file(s)
    selection

    >>> nd = scp.read_opus()

    Read in a directory (assume that only OPUS files are present in the directory
    (else we must use the generic `read` function instead)

    >>> le = scp.read_opus(directory='irdata/OPUS')
    >>> len(le)
    4

    Again we can use merge to stack all 4 spectra if thet have compatible dimensions.

    >>> scp.read_opus(directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:4, x:2567))
    """

    kwargs["filetypes"] = ["Bruker OPUS files (*.[0-9]*)"]
    kwargs["protocol"] = ["opus"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private Functions
# ======================================================================================
@_importer_method
def _read_opus(*args, **kwargs):
    debug_("Bruker OPUS import")

    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    opus_data = _read_data(fid)

    # data
    try:
        npt = opus_data["AB Data Parameter"]["NPT"]
        data = opus_data["AB"][:npt]
        dataset.data = np.array(data[np.newaxis], dtype="float32")
    except KeyError:
        raise KeyError(
            f"{filename} is not an Absorbance spectrum. It cannot be read with the `read_opus` import method"
        )
    # todo: read background

    # xaxis
    fxv = opus_data["AB Data Parameter"]["FXV"]
    lxv = opus_data["AB Data Parameter"]["LXV"]
    # xdata = linspace(fxv, lxv, npt)
    xaxis = LinearCoord.linspace(fxv, lxv, npt, title="wavenumbers", units="cm^-1")

    # yaxis
    name = opus_data["Sample"]["SNM"]
    acqdate = opus_data["AB Data Parameter"]["DAT"]
    acqtime = opus_data["AB Data Parameter"]["TIM"]
    gmt_offset_hour = float(acqtime.split("GMT")[1].split(")")[0])
    if len(acqdate.split("/")[0]) == 2:
        date_time = datetime.strptime(
            acqdate + "_" + acqtime.split()[0], "%d/%m/%Y_%H:%M:%S.%f"
        )
    elif len(acqdate.split("/")[0]) == 4:
        date_time = datetime.strptime(
            acqdate + "_" + acqtime.split()[0], "%Y/%m/%d_%H:%M:%S"
        )
    utc_dt = date_time - timedelta(hours=gmt_offset_hour)
    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    timestamp = utc_dt.timestamp()

    yaxis = Coord(
        [timestamp],
        title="acquisition timestamp (GMT)",
        units="s",
        labels=([utc_dt], [name], [filename]),
    )

    # set dataset's Coordset
    dataset.set_coordset(y=yaxis, x=xaxis)
    dataset.units = "absorbance"
    dataset.title = "absorbance"

    # Set name, origin, description and history
    dataset.name = filename.name
    dataset.origin = "opus"
    dataset.description = "Dataset from opus files. \n"
    dataset.history = str(datetime.now(timezone.utc)) + ": import from opus files \n"

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset


def _read_data(fid):
    data = fid.read()
    meta_data = parse_meta(data)
    opus_data = parse_data(data, meta_data)
    return opus_data
