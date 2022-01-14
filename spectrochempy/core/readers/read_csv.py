# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

__all__ = ["read_csv"]
__dataset_methods__ = __all__

# ------------------------------------------------------------------
# standard and other imports
# ------------------------------------------------------------------

import warnings
import locale
import io
from datetime import datetime, timezone

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core import preferences as prefs
from spectrochempy.core.readers.importer import Importer, importermethod

try:
    locale.setlocale(locale.LC_ALL, "en_US")  # to avoid problems with date format
except Exception:  # pragma: no cover
    try:
        locale.setlocale(
            locale.LC_ALL, "en_US.utf8"
        )  # to avoid problems with date format
    except Exception:
        warnings.warn("Could not set locale: en_US or en_US.utf8")


# ======================================================================================================================
# Public functions
# ======================================================================================================================
def read_csv(*paths, **kwargs):
    """
    Open a *.csv file or a list of *.csv files.

    This is limited to 1D array - csv file must have two columns [index, data]
    without header.

    Parameters
    ----------
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
    read_csv
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
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
    origin : {'omnic', 'tga'}, optional
        in order to properly interpret CSV file it can be necessary to set the origin of the spectra.
        Up to now only 'omnic' and 'tga' have been implemented.
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy ``Preferences``.
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
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.
    read : Generic file reading.

    Examples
    ---------

    >>> scp.read_csv('agirdata/P350/TGA/tg.csv')
    NDDataset: [float64] unitless (shape: (y:1, x:3247))

    Additional information can be stored in the dataset if the origin is given
    (known origin for now : tga or omnic)
    # TODO: define some template to allow adding new origins

    >>> scp.read_csv('agirdata/P350/TGA/tg.csv', origin='tga')
    NDDataset: [float64] wt.% (shape: (y:1, x:3247))

    Sometimes the delimiteur needs to be adjusted

    >>> prefs = scp.preferences
    >>> scp.read_csv('irdata/IR.CSV', directory=prefs.datadir, origin='omnic', csv_delimiter=',')
    NDDataset: [float64] a.u. (shape: (y:1, x:3736))
    """
    kwargs["filetypes"] = ["CSV files (*.csv)"]
    kwargs["protocol"] = ["csv"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================


@importermethod
def _read_csv(*args, **kwargs):
    # read csv file
    dataset, filename = args
    content = kwargs.get("content", None)
    delimiter = kwargs.get("csv_delimiter", prefs.csv_delimiter)

    def _open():
        if content is not None:
            f = io.StringIO(content.decode("utf-8"))
        else:
            f = open(filename, "r")
        return f

    try:
        fid = _open()
        d = np.loadtxt(fid, unpack=True, delimiter=delimiter)
        fid.close()
    except ValueError:
        # it might be that the delimiter is not correct (default is ','), but
        # french excel export with the french locale for instance, use ";".
        _delimiter = ";"
        try:
            fid = _open()
            if fid:
                fid.close()
            fid = _open()
            d = np.loadtxt(fid, unpack=True, delimiter=_delimiter)
            fid.close()

        except Exception:  # pragma: no cover
            # in french, very often the decimal '.' is replaced by a
            # comma:  Let's try to correct this
            if fid:
                fid.close()
            if not isinstance(fid, io.StringIO):
                with open(fid, "r") as fid_:
                    txt = fid_.read()
            else:
                txt = fid.read()
            txt = txt.replace(",", ".")
            fil = io.StringIO(txt)
            try:
                d = np.loadtxt(fil, unpack=True, delimiter=delimiter)
            except Exception:
                raise IOError(
                    "{} is not a .csv file or its structure cannot be recognized"
                )

    # First column is the x coordinates
    coordx = Coord(d[0])

    # create a second coordinate for dimension y of size 1
    coordy = Coord([0])

    # and data is the second column -  we make it a vector
    data = d[1].reshape((1, coordx.size))

    # update the dataset
    dataset.data = data
    dataset.set_coordset(y=coordy, x=coordx)

    # set the additional attributes
    name = filename.stem
    dataset.filename = filename
    dataset.name = kwargs.get("name", name)
    dataset.title = kwargs.get("title", None)
    dataset.units = kwargs.get("units", None)
    dataset.description = kwargs.get("description", '"name" ' + "read from .csv file")
    dataset.history = str(datetime.now(timezone.utc)) + ":read from .csv file \n"
    dataset._date = datetime.now(timezone.utc)
    dataset._modified = dataset.date

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
            "(Up to now implemented csv files are: `omnic`, `tga`)"
        )
    return dataset


# .............................................................................
def _add_omnic_info(dataset, **kwargs):
    # get the time and name
    name = desc = dataset.name

    # modify the dataset metadata
    dataset.units = "absorbance"
    dataset.title = "absorbance"
    dataset.name = name
    dataset.description = "Dataset from .csv file: {}\n".format(desc)
    dataset.history = (
        str(datetime.now(timezone.utc)) + ":read from omnic exported csv file \n"
    )
    dataset.origin = "omnic"

    # Set the NDDataset date
    dataset._date = datetime.now(timezone.utc)
    dataset._modified = dataset.date

    # x axis
    dataset.x.units = "cm^-1"

    # y axis ?
    if "_" in name:
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

    return dataset


def _add_tga_info(dataset, **kwargs):
    # for TGA, some information are needed.
    # we add them here
    dataset.x.units = "hour"
    dataset.units = "weight_percent"
    dataset.x.title = "time-on-stream"
    dataset.title = "mass change"
    dataset.origin = "tga"

    return dataset


# ------------------------------------------------------------------
if __name__ == "__main__":
    pass
