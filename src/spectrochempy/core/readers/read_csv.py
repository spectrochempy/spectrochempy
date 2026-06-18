# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["read_csv"]

import csv

# --------------------------------------------------------------------------------------
# standard and other imports
# --------------------------------------------------------------------------------------
import locale
import warnings
from datetime import datetime

import numpy as np

from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.importer import _openfid

try:
    locale.setlocale(locale.LC_ALL, "en_US")  # to avoid problems with date format
except Exception:  # pragma: no cover
    try:
        locale.setlocale(
            locale.LC_ALL,
            "en_US.utf8",
        )  # to avoid problems with date format
    except Exception:
        warnings.warn("Could not set locale: en_US or en_US.utf8", stacklevel=2)


# ======================================================================================
# Public functions
# ======================================================================================
def read_csv(*paths, **kwargs):
    r"""
    Open CSV (comma-separated values) files.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to ``False``.
    **kwargs : keyword parameters, optional
        See Other Parameters.

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
        A custom description.
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

        It is used, for instance, when reading a directory with different types of
        files and merging compatible datasets into separate groups by origin.

        It is also used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading, for example ``'scp'``, ``'omnic'``,
        ``'opus'``, ``'matlab'``, ``'jcamp'``, ``'csv'``, or ``'excel'``.
        If not provided, the correct protocol is inferred whenever possible
        from the filename extension.
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
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra (:file:`.txt`).
    read_omnic : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    read_soc : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    read_galactic : Read Galactic files (:file:`.spc`).
    read_quadera : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.

    read_matlab : Read Matlab files (:file:`.mat`, :file:`.dso`).
    read_jcamp : Read Infrared JCAMP-DX files (:file:`.jdx`, :file:`.dx`).
    read_wire : Read Renishaw Wire files (:file:`.wdf`).

    Examples
    --------
    Reading a single CSV file

    >>> scp.read_csv('irdata/csv/iris.csv')
    NDDataset: [float64] a.u. (shape: (y:150, x:4))

    """
    kwargs["filetypes"] = ["CSV files (*.csv)"]
    kwargs["protocol"] = ["csv"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_csv(*args, **kwargs):
    # read csv file
    dataset, filename = args
    delimiter = kwargs.get("csv_delimiter", prefs.csv_delimiter)

    fid, kwargs = _openfid(filename, mode="r", **kwargs)

    txt = fid.read()
    fid.close()

    # We assume this csv file contains only numbers # TODO: write a more general reader
    if ";" in txt:
        # look like the delimiter is ;
        # if comma is also present, it could be that french writer was used.
        txt = txt.replace(",", ".")
        delimiter = ";"

    d = list(csv.reader(txt.splitlines(), delimiter=delimiter))

    # Skip header row if present (non-numeric first row from write_csv)
    def _is_numeric_row(row):
        """Check if a row contains only numeric values."""
        for item in row:
            try:
                float(item)
            except (ValueError, TypeError):
                return False
        return True

    if d and not _is_numeric_row(d[0]):
        d = d[1:]

    d = np.array(d, dtype=float).T

    # Handle both single-column (data only) and multi-column (x + data) CSV files
    if d.shape[0] == 1:
        # Single column: data only, create synthetic x coordinates
        data = d[0].reshape((1, d.shape[1]))
        coordx = Coord(np.arange(d.shape[1]))
    else:
        # Multiple columns: first is x, second is data
        coordx = Coord(d[0])
        data = d[1].reshape((1, coordx.size))

    # Create a second coordinate for dimension y of size 1
    coordy = Coord([0])

    # try:
    #     d = np.loadtxt(fid, unpack=True, delimiter=delimiter)
    #     fid.close()
    #
    # except ValueError:
    #     # it might be that the delimiter is not correct (default is ','), but
    #     # french excel export with the french locale for instance, use ";".
    #     _delimiter = ";"
    #     try:
    #         if fid:
    #             fid.close()
    #         fid, kwargs = _openfid(filename, mode="r", **kwargs)
    #         d = np.loadtxt(fid, unpack=True, delimiter=_delimiter)
    #         fid.close()
    #
    #     except Exception:  # pragma: no cover
    #         # in french, very often the decimal '.' is replaced by a
    #         # comma:  Let's try to correct this
    #         if fid:
    #             fid.close()
    #         fid, kwargs = _openfid(filename, mode="r", **kwargs)
    #         txt = fid.read()
    #         fid.close()
    #
    #         txt = txt.replace(",", ".")
    #
    #         fid = io.StringIO(txt)
    #         try:
    #             d = np.loadtxt(fid, unpack=True, delimiter=delimiter)
    #         except Exception:
    #             raise IOError(
    #                 "{} is not a .csv file or its structure cannot be recognized"
    #             )

    # Update the dataset
    dataset.data = data
    dataset.set_coordset(y=coordy, x=coordx)

    # set the additional attributes
    name = filename.stem
    dataset.filename = filename
    dataset.name = kwargs.get("name", name)
    dataset.title = kwargs.get("title", None)
    dataset.units = kwargs.get("units", None)
    dataset.description = kwargs.get("description", '"name" ' + "read from .csv file")
    dataset.history = "Read from .csv file"

    # here we can check some particular format
    origin = kwargs.get("origin", "")
    if "omnic" in origin:
        # this will be treated as csv export from omnic (IR data)
        dataset = _add_omnic_info(dataset, **kwargs)
    elif "tga" in origin:
        # this will be treated as csv export from tga analysis
        dataset = _add_tga_info(dataset, **kwargs)
    elif origin:
        raise NotImplementedError(
            f"Sorry, but reading a csv file with '{origin}' origin is not implemented. "
            "Please, remove or set the keyword 'origin'\n "
            "(Up to now implemented csv files are: `omnic` , `tga` )",
        )

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset


def _add_omnic_info(dataset, **kwargs):
    # get the time and name
    name = desc = dataset.name

    # modify the dataset metadata
    dataset.units = "absorbance"
    dataset.title = "absorbance"
    dataset.name = name
    dataset.description = f"Dataset from .csv file: {desc}\n"
    dataset.history = "Read from omnic exported csv file."
    dataset.origin = "omnic"

    # x axis
    dataset.x.units = "cm^-1"

    # y axis ?
    if "_" in name:
        try:
            name, dat = name.split("_")
            # if needed convert weekday name to English
            dat = dat.replace("Lun", "Mon")
            dat = dat[:3].replace("Mar", "Tue") + dat[3:]
            dat = dat.replace("Mer", "Wed")
            dat = dat.replace("Jeu", "Thu")
            dat = dat.replace("Ven", "Fri")
            dat = dat.replace("Sam", "Sat")
            dat = dat.replace("Dim", "Sun")
            # convert month name to English
            dat = dat.replace("Aout", "Aug")

            # get the dates
            acqdate = datetime.strptime(dat, "%a %b %d %H-%M-%S %Y")

            # Transform back to timestamp for storage in the Coord object
            # use datetime.fromtimestamp(d, timezone.utc))
            # to transform back to datetime obkct
            timestamp = acqdate.timestamp()

            dataset.y = Coord(np.array([timestamp]), name="y")
            dataset.set_coordtitles(y="acquisition timestamp (GMT)", x="wavenumbers")
            dataset.y.labels = np.array([[acqdate], [name]])
            dataset.y.units = "s"
        except (ValueError, AttributeError):
            # If date parsing fails, just keep default y coordinate
            pass

    # reset modification date to cretion date
    dataset._modified = dataset._created

    return dataset


def _add_tga_info(dataset, **kwargs):
    # for TGA, some information are needed.
    # we add them here
    dataset.x.units = "hour"
    dataset.units = "percent"
    dataset.x.title = "time-on-stream"
    dataset.title = "mass change"
    dataset.origin = "tga"

    return dataset
