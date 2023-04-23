# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["read_csv"]
__dataset_methods__ = __all__

import csv

# --------------------------------------------------------------------------------------
# standard and other imports
# --------------------------------------------------------------------------------------
import locale
import warnings
from datetime import datetime

import numpy as np

from spectrochempy.core import preferences as prefs
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer, _importer_method, _openfid
from spectrochempy.utils.docstrings import _docstring

try:
    locale.setlocale(locale.LC_ALL, "en_US")  # to avoid problems with date format
except Exception:  # pragma: no cover
    try:
        locale.setlocale(
            locale.LC_ALL, "en_US.utf8"
        )  # to avoid problems with date format
    except Exception:
        warnings.warn("Could not set locale: en_US or en_US.utf8")


# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_csv")


@_docstring.dedent
def read_csv(*paths, **kwargs):
    """
    Open a :file:`.csv` file or a list of :file:`.csv` files.

    This is limited to 1D array - csv file must have two columns [index, data]
    without header.

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
    --------
    %(Importer.see_also.no_read_csv)s

    Examples
    ---------

    >>> scp.read_csv('agirdata/P350/TGA/tg.csv')
    NDDataset: [float64] unitless (shape: (y:1, x:3247))

    Additional information can be stored in the dataset if the origin is given
    (known origin for now : tga or omnic)
    # TODO: define some template to allow adding new origins

    >>> A = scp.read_csv('agirdata/P350/TGA/tg.csv', origin='tga')

    Sometimes the delimiteur needs to be adjusted

    >>> prefs = scp.preferences
    >>> B = scp.read_csv('irdata/IR.CSV', origin='omnic', csv_delimiter=',')
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

    d = [row for row in csv.reader(txt.splitlines(), delimiter=delimiter)]
    d = np.array(d).T

    # First column is the x coordinates
    coordx = Coord(d[0])

    # Create a second coordinate for dimension y of size 1
    coordy = Coord([0])

    # and data is the second column -  we make it a vector
    data = d[1].reshape((1, coordx.size))

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
            "(Up to now implemented csv files are: `omnic` , `tga` )"
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
    dataset.description = "Dataset from .csv file: {}\n".format(desc)
    dataset.history = "Read from omnic exported csv file."
    dataset.origin = "omnic"

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

    # reset modification date to cretion date
    dataset._modified = dataset._created

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
