# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Extend NDDataset with the import method for OPUS generated data files."""

__all__ = ["read_opus"]
__dataset_methods__ = __all__

from datetime import datetime
from datetime import timedelta

import numpy as np

from spectrochempy.application import debug_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid

# from brukeropusreader.opus_parser import parse_data
# from brukeropusreader.opus_parser import parse_meta
from spectrochempy.extern.brukeropus import OPUSFile
from spectrochempy.utils.datetimeutils import UTC
from spectrochempy.utils.docreps import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_opus")


@_docstring.dedent
def read_opus(*paths, **kwargs):
    r"""
    Open Bruker OPUS file(s).

    Eventually group them in a single dataset. Returns an error if dimensions are incompatibles.


    Parameters
    ----------
    %(Importer.parameters)s
    type : str, optional
        The type of data to be read. Possible values are:

        - "AB": Absorbance (default if present in the file)
        - "TR": Transmittance
        - "KM": Kubelka-Munk
        - "RAM": Raman
        - "EMI": Emission
        - "RFL": Reflectance
        - "LRF": log(Reflectance)
        - "ATR": ATR
        - "PAS": Photoacoustic
        - "RF": Single-channel reference spectra
        - "SM": Single-channel sample spectra
        - "IGRF": Reference interferogram
        - "IGSM": Sample interferogram
        - "PHRF": Reference phase
        - "PHSM": Sample phase

        An error is raised if the specified type is not present in the file.

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    %(Importer.other_parameters)s

    See Also
    --------
    %(Importer.see_also.no_read_opus)s

    Examples
    --------
    Reading a single OPUS file  (providing a windows type filename relative to
    the default `datadir` )

    >>> scp.read_opus('irdata\\OPUS\\test.0000')
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

    >>> le = scp.read_opus('test.0000', 'test.0001', 'test.0002',
    >>>                    directory='irdata/OPUS')
    >>> len(le)
    3
    >>> le[0]
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Multiple files merged as the `merge` keyword is set to true

    >>> scp.read_opus('test.0000', 'test.0001', 'test.0002',
    directory='irdata/OPUS', merge=True)
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files to merge : they are passed as a list instead of using the keyword `
    merge`

    >>> scp.read_opus(['test.0000', 'test.0001', 'test.0002'],
    >>>               directory='irdata/OPUS')
    NDDataset: [float64] a.u. (shape: (y:3, x:2567))

    Multiple files not merged : they are passed as a list but `merge` is set to false

    >>> le = scp.read_opus(['test.0000', 'test.0001', 'test.0002'],
    >>>                    directory='irdata/OPUS', merge=False)
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

_data_types = {
    "sm": "Single-channel sample spectra",
    "rf": "Single-channel reference spectra",
    "igsm": "Sample interferogram",
    "igrf": "Reference interferogram",
    "phsm": "Sample phase",
    "phrf": "Reference phase",
    "a": "Absorbance",
    "t": "Transmittance",
    "r": "Reflectance",
    "km": "Kubelka-Munk",
    "tr": "Trace (Intensity over Time)",
    "gcig": "gc File (Series of Interferograms)",
    "gcsc": "gc File (Series of Spectra)",
    "ra": "Raman",
    "e": "Emission",
    "pw": "Power",
    "logr": "log(Reflectance)",
    "atr": "ATR",
    "pas": "Photoacoustic",
}
_spectra_types = {
    "RF": "Single-channel reference spectra",
    "SM": "Single-channel sample spectra",
    "IGRF": "Reference interferogram",
    "IGSM": "Sample interferogram",
    "PHRF": "Reference phase",
    "PHSM": "Sample phase",
    "AB": "Absorbance",
    "TR": "Transmittance",
    "KM": "Kubelka-Munk",
    "RAM": "Raman",
    "EMI": "Emission",
    "RFL": "Reflectance",
    "LRF": "log(Reflectance)",
    "ATR": "ATR",
    "PAS": "Photoacoustic",
}

_units = {
    "WN": ("wavenumber", "cm^-1"),
    "WL": ("wavelength", "µm"),
    "Absorbance": "absorbance",  # defined in core/units
    "Transmittance": "transmittance",
    "Kubelka-Munk": "Kubelka_Munk",
}


def _data_from_value(dic, value):
    return [k for k, v in dic.items() if v == value][0]


def _get_timestamp_from(params):
    # get the acquisition timestamp
    acqdate = params.dat
    acqtime = params.tim
    gmt_offset_hour = float(acqtime.split("GMT")[1].split(")")[0])
    if len(acqdate.split("/")[0]) == 2:
        date_time = datetime.strptime(
            acqdate + "_" + acqtime.split()[0],
            "%d/%m/%Y_%H:%M:%S.%f",
        )
    elif len(acqdate.split("/")[0]) == 4:
        date_time = datetime.strptime(
            acqdate + "_" + acqtime.split()[0],
            "%Y/%m/%d_%H:%M:%S",
        )
    else:  # pragma: no cover
        raise ValueError("acqdate can not be interpreted.")
    utc_dt = date_time - timedelta(hours=gmt_offset_hour)
    utc_dt = utc_dt.replace(tzinfo=UTC)
    return utc_dt, utc_dt.timestamp()


@_importer_method
def _read_opus(*args, **kwargs):
    debug_("Bruker OPUS import")

    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    opus_data = OPUSFile(fid)

    # data present
    all_data_types = opus_data.all_data_keys  # type of data present in the file

    # check if the data type is specified
    typ = kwargs.get("type", "AB")
    spectra_type = _spectra_types.get(typ)
    if spectra_type is None:
        raise ValueError(
            f"Unknown data type {typ}. Possible values are: {list(_spectra_types.keys())}",
        )
    data_type = _data_from_value(_data_types, spectra_type)

    if data_type not in all_data_types:
        available_data_types = ".".join(
            [k for k, v in _spectra_types.items() if v in all_data_types]
        )
        raise ValueError(
            f"The data type {typ} is not present in the file. "
            f"Available data types are: {available_data_types}",
        )

    d = getattr(opus_data, data_type)

    # data
    dataset.data = d.y if d.y.ndim > 1 else d.y[np.newaxis]
    dataset.title = spectra_type.lower()
    dataset.units = _units.get(spectra_type)

    # xaxis
    title, units = _units.get(getattr(d.params, "dxu"))  # noqa: B009
    xaxis = Coord(d.x, title=title, units=units)

    # yaxis (in case this is not a data series)
    # TODO: check if this is a data series and read eventually 2D data
    name = opus_data.params.snm
    dt, timestamp = _get_timestamp_from(d.params)

    yaxis = Coord(
        [timestamp],
        title="acquisition timestamp (GMT)",
        units="s",
        labels=([dt], [name], [filename]),
    )

    # set dataset's Coordset
    dataset.set_coordset(y=yaxis, x=xaxis)

    # Set name, origin, description and history
    dataset.name = filename.name
    dataset.filename = filename
    dataset.origin = "opus"
    dataset.description = "Dataset from opus files. \nSpectra type: " + spectra_type
    dataset.history = str(datetime.now(UTC)) + ": import from opus files \n"

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset
