# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Extend NDDataset with the import method for OPUS generated data files."""

__all__ = ["read_opus"]

from datetime import datetime
from datetime import timedelta

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid
from spectrochempy.extern.brukeropus import OPUSFile
from spectrochempy.extern.brukeropus.file.utils import get_block_type_label
from spectrochempy.extern.brukeropus.file.utils import get_param_label
from spectrochempy.utils._logging import debug_
from spectrochempy.utils.datetimeutils import UTC
from spectrochempy.utils.meta import Meta

# ======================================================================================
# Public functions
# ======================================================================================


def read_opus(*paths, **kwargs):
    r"""
    Open Bruker OPUS file(s).

    Eventually group them in a single dataset. Returns an error if dimensions are incompatibles.


    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ...,  kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to `False`.
    **kwargs : keyword parameters, optional
        See Other Parameters.
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
        - "TRACE": Trace (intensity over time) for time-resolved files
        - "GCIG": GC file (series of interferograms)
        - "GCSC": GC file (series of spectra)

        An error is raised if the specified type is not present in the file.

    Returns
    -------
    object : `NDDataset` or list of `NDDataset`
        The returned dataset(s).

    Other Parameters
    ----------------
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a GUI Dash application to handle drag and drop of files into a
        Browser.
    csv_delimiter : `str`, optional, default: `~spectrochempy.preferences.csv_delimiter`
        Set the column delimiter in CSV file.
    description : `str`, optional
        A Custom description.
    directory : `~pathlib.Path` object objects or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `True` and several filenames or a ``directory`` have been provided as
        arguments, then a single `NDDataset` with merged dataset (stacked along the first
        dimension) is returned. In the case not all datasets have compatible dimensions or types/origins,
        then several NDDatasets can be returned for different groups of compatible datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used for instance whn reading directory with different types of files, for merging
        the datasets with compatible dimensions and different origin into different groups.

        It is also used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading. It can be one of {``'scp'``, ``'omnic'``,
        ``'opus'``, ````, ``'matlab'``, ``'jcamp'``, ``'csv'``,
        ``'excel'``}. If not provided, the correct protocol
        is inferred (whenever it is possible) from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic reader inferring protocol from the filename extension.
    read_zip : Read Zip archives (containing spectrochempy readable files)
    read_dir : Read an entire directory.
    read_labspec : Read Raman LABSPEC spectra (:file:`.txt`).
    read_omnic : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    read_soc : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    read_galactic : Read Galactic files (:file:`.spc`).
    read_quadera : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.

    read_csv : Read CSV files (:file:`.csv`).
    read_matlab : Read Matlab files (:file:`.mat`, :file:`.dso`).
    read_wire : Read Renishaw Wire files (:file:`.wdf`).

    Examples
    --------
    Reading a single OPUS file  (providing a windows type filename relative to
    the default `datadir` )

    >>> scp.read_opus('irdata\\OPUS\\test.0000')
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

    Reading a single OPUS file  (providing a unix/python type filename relative to
    the default ``datadir`` )

    >>> scp.read_opus('irdata/OPUS/test.0000')
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

    Multiple files to merge : they are passed as a list instead of using the keyword
    `merge`

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
    2

    Again we can use merge to stack all 4 spectra if they have compatible dimensions.

    >>> scp.read_opus(directory='irdata/OPUS', merge=True)
    [NDDataset: [float64] a.u. (shape: (y:1, x:2567)), NDDataset: [float64] a.u. (shape: (y:4, x:2567))]

    Bruker OPUS files often contain several types of data (AB, RF, IGSM ...).
    In some cases, the type of data can be inferred from the file content.
    For instance, if the file contains a single background spectrum, it is inferred
    as a reference spectrum. In the following example, the file `background.0` is
    correctly inferred as a reference spectrum.

    >>> B = scp.read_opus('irdata/OPUS/background.0')
    >>> B
    NDDataset: [float64] a.u. (shape: (y:1, x:4096))

    If the type of data can not be inferred, an error is raised. In the following
    example, if the file `test.0000` contains only sample spectra (SM), the type of data
    must be specified.

    >>> A = scp.read_opus('irdata/OPUS/test.0000', type='SM')
    >>> A
    NDDataset: [float64] a.u. (shape: (y:1, x:2567))

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

_types_parameters = {
    "RF": "rf",
    "SM": "sm",
    "IGRF": "igrf",
    "IGSM": "igsm",
    "PHRF": "phrf",
    "PHSM": "phsm",
    "AB": "a",
    "TR": "t",
    "KM": "km",
    "RAM": "ra",
    "EMI": "e",
    "RFL": "r",
    "LRF": "logr",
    "ATR": "atr",
    "PAS": "pas",
    "TRACE": "tr",
    "GCIG": "gcig",
    "GCSC": "gcsc",
}

_types_parameters_inv = {v: k for k, v in _types_parameters.items()}

_blocks_parameters = {
    0: "",
    1: "Data Status",  # 'Data Status Parameters',
    2: "Instrument Status",  # 'Instrument Status Parameters',
    3: "Acquisition",  # 'Acquisition Parameters',
    4: "Fourier Transform",  # 'Fourier Transform Parameters',
    5: "Plot and Display",  # 'Plot and Display Parameters',
    6: "Optical",  # 'Optical Parameters',
    7: "GC",  # 'GC Parameters',
    8: "Library Search",  # 'Library Search Parameters',
    9: "Communication",  # 'Communication Parameters',
    10: "Sample Origin",  # 'Sample Origin Parameters',
    11: "Lab and Process",  # 'Lab and Process Parameters',
}

_blocks_parameters_inv = {v: k for k, v in _blocks_parameters.items()}

_units = {
    "WN": ("wavenumber", "cm^-1"),
    "WL": ("wavelength", "µm"),
    "Absorbance": "absorbance",  # defined in core/units
    "Transmittance": "transmittance",
    "Kubelka-Munk": "Kubelka_Munk",
}


def _get_timestamp_from(params):
    # get the acquisition timestamp
    acqdate = params.dat
    acqtime = params.tim
    gmt_offset_hour = float(acqtime.split("GMT")[1].split(")")[0])
    if len(acqdate.split("/")[0]) == 2:
        dt_str = acqdate + "_" + acqtime.split()[0]
        try:
            date_time = datetime.strptime(dt_str, "%d/%m/%Y_%H:%M:%S.%f")
        except ValueError:
            # Some OPUS files store a malformed sub-second field, e.g.
            # "10:31:19.-70"; fall back to whole-second precision instead of
            # failing the whole read (#1036).
            date_time = datetime.strptime(dt_str.split(".")[0], "%d/%m/%Y_%H:%M:%S")
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


def _load_parameters_into_meta(opus_data):
    # Load the parameters in the metadata.
    meta = Meta()
    param_infos = [
        ("Sample/Result Parameters", "params"),
        ("Reference Parameters", "rf_params"),
    ]
    for title, attr in param_infos:
        meta[attr] = Meta(name=title)
        blocks = getattr(opus_data, attr).blocks
        for block in blocks:
            id = f"{_blocks_parameters[block.type[2]].lower().replace(' ', '_')}"
            meta[attr][id] = Meta(name=get_block_type_label(block.type))
            for key in block.keys:
                name = get_param_label(key)
                value = getattr(getattr(opus_data, attr), key)
                meta[attr][id][key] = Meta(name=name, value=value)

    return meta


@_importer_method
def _read_opus(*args, **kwargs):
    debug_("Bruker OPUS import")

    dataset, filename = args

    fid, kwargs = _openfid(filename, **kwargs)

    opus_data = OPUSFile(fid)

    # Which data are present in the file?
    all_data_types = opus_data.all_data_keys  # type of data present in the file
    possible_type_parameters = [_types_parameters_inv[k] for k in all_data_types]

    # Get type parameter
    type_parameter = kwargs.get("type")

    # Check if the data type is specified
    if type_parameter is None:
        # if AB in possible_type_parameters, take it as default (backward compatibility)
        if "AB" in possible_type_parameters:
            type_parameter = "AB"

        # if RAM, it is a raman spectra
        elif "RAM" in possible_type_parameters:
            type_parameter = "RAM"

        # if EMI, it is an emission spectra
        elif "EMI" in possible_type_parameters:
            type_parameter = "EMI"

        # it may be a single background
        elif possible_type_parameters == ["RF", "IGRF"]:
            type_parameter = "RF"

        else:
            raise ValueError(
                f"Please specify the type of data to read. Possible values for this file are: {possible_type_parameters}"
            )
        # warning_(
        #     f"Default type parameter {type_parameter} is used.\n"
        #     "Please specify the type of data to read if this is not the correct data type to read.\n"
        #     f"Possible values for this file are: {possible_type_parameters}"
        # )

    elif type_parameter is not None and type_parameter not in possible_type_parameters:
        raise ValueError(
            f"The data type {type_parameter} is not present in the file. Possible values for this file are: {possible_type_parameters}"
        )

    data_type = _types_parameters[type_parameter]

    d = getattr(opus_data, data_type)

    # data
    dataset.data = d.y if d.y.ndim > 1 else d.y[np.newaxis]
    desc = _data_types[data_type]
    dataset.title = desc.lower()
    dataset.units = _units.get(desc)  # None if not found in _units

    # xaxis
    dxu = getattr(d.params, "dxu", None)  # noqa: B009
    xu = _units.get(dxu) if dxu else None
    if xu is None:
        title, units = "wavenumber", "cm^-1"
    else:
        title, units = xu
    xaxis = Coord(d.x, title=title, units=units)

    # yaxis
    name = opus_data.params.snm
    if d.y.ndim > 1 and d.y.shape[0] > 1:
        # assembled / time-resolved data series
        if "ert" in d.params:
            yaxis = Coord(
                d.params.ert,
                title="elapsed time",
                units="s",
            )
        else:
            yaxis = Coord(
                np.arange(d.y.shape[0]),
                title="spectrum index",
            )
    else:
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
    dataset.name = filename.stem
    dataset.filename = filename
    dataset.origin = f"opus-{type_parameter}"
    dataset.description = "Dataset from opus files. \nSpectra type: " + desc
    dataset.history = str(datetime.now(UTC)) + ": import from opus files \n"

    # add other parameters in metadata
    dataset.meta = _load_parameters_into_meta(opus_data)
    dataset.meta.name = "OPUS Parameters"

    # add info about other type present in th file and which could be alternatively read.
    dataset.meta["other_data_types"] = possible_type_parameters

    # set the dataset as readonly now that all the metadata is set
    dataset.meta.readonly = True

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset
