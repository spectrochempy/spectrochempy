# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module extend NDDataset with the import method for OPUS generated data files.
"""
__all__ = ["read_opus"]
__dataset_methods__ = __all__

import io
import numpy as np
from datetime import datetime, timezone, timedelta

from brukeropusreader.opus_parser import parse_data, parse_meta
from spectrochempy.core.dataset.coord import LinearCoord, Coord
from spectrochempy.core.readers.importer import Importer, importermethod
from spectrochempy.core import debug_


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_opus(*paths, **kwargs):
    """
    Open Bruker OPUS file(s).

    Eventually group them in a single dataset. Only Absorbance spectra are
    extracted ("AB" field). Returns an error if dimensions are incompatibles.

    Parameters
    -----------
    *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
        The data source(s) can be specified by the name or a list of name for the file(s) to be loaded:

        *e.g.,( file1, file2, ...,  **kwargs )*

        If the list of filenames are enclosed into brackets:

        *e.g.,* ( **[** *file1, file2, ...* **]**, **kwargs *)*

        The returned datasets are merged to form a single dataset,
        except if `merge` is set to False. If a source is not provided (i.e. no `filename`, nor `content`),
        a dialog box will be opened to select files.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_opus
        The dataset or a list of dataset corresponding to a (set of) OPUS file(s).

    Other Parameters
    -----------------
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description: str, optional
        A Custom description.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True).
    recursive : bool, optional
        Read also in subfolders. (default=False).

    See Also
    --------
    read : Generic read method.
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.

    Examples
    ---------
    Reading a single OPUS file  (providing a windows type filename relative to the default ``Datadir``)

    >>> scp.read_opus('irdata\\\\OPUS\\\\test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative to the default ``Datadir``)
    Note that here read_opus is called as a classmethod of the NDDataset class

    >>> scp.NDDataset.read_opus('irdata/OPUS/test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Single file specified with pathlib.Path object

    >>> from pathlib import Path
    >>> folder = Path('irdata/OPUS')
    >>> p = folder / 'test.0000'
    >>> scp.read_opus(p)
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files not merged (return a list of datasets). Note that a directory is specified

    >>> le = scp.read_opus('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS')
    >>> len(le)
    3
    >>> le[0]
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read_opus('test.0000', 'test.0001', 'test.0002', directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword `merge`

    >>> scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS')
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> le = scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS', merge=False)
    >>> len(le)
    3

    Read without a filename. This has the effect of opening a dialog for file(s) selection

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


# ======================================================================================================================
# Private Functions
# ======================================================================================================================

# ..............................................................................
@importermethod
def _read_opus(*args, **kwargs):
    debug_("Bruker OPUS import")

    dataset, filename = args
    content = kwargs.get("content", None)

    if content:
        fid = io.BytesIO(content)
    else:
        fid = open(filename, "rb")

    opus_data = _read_data(fid)

    # data
    try:
        npt = opus_data["AB Data Parameter"]["NPT"]
        data = opus_data["AB"][:npt]
        dataset.data = np.array(data[np.newaxis], dtype="float32")
    except KeyError:
        raise IOError(
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
    date_time = datetime.strptime(
        acqdate + "_" + acqtime.split()[0], "%d/%m/%Y_%H:%M:%S.%f"
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
    dataset._date = datetime.now(timezone.utc)
    dataset._modified = dataset.date

    return dataset


# ..............................................................................
def _read_data(fid):
    data = fid.read()
    meta_data = parse_meta(data)
    opus_data = parse_data(data, meta_data)
    return opus_data


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
