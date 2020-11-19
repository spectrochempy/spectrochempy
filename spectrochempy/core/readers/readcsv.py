# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

__all__ = ['read_csv']
__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard and other imports
# ----------------------------------------------------------------------------------------------------------------------

import warnings
import locale
import io
from datetime import datetime

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core import general_preferences as prefs
from spectrochempy.core.readers.importer import docstrings, _Importer


try:
    locale.setlocale(locale.LC_ALL, 'en_US')  # to avoid problems with date format
except Exception:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.utf8')  # to avoid problems with date format
    except Exception:
        warnings.warn('Could not set locale: en_US or en_US.utf8')

# ======================================================================================================================
# Public functions
# ======================================================================================================================

# NOTE: Warning @docstrings doesnt't work if % sign are present in the docstring. We need to double it %% !

@docstrings.dedent
def read_csv(*args, **kwargs):
    """
    Open a *.csv file or a list of *.csv files and set data/metadata
    in the current dataset

    Parameters
    ------
    %(read_method.parameters)s

    Other Parameters
    -----------------
    %(read_method.other_parameters)s

    Returns
    --------
    out : NDDataset| or list of |NDDataset|
        The dataset or a list of dataset corresponding to a (set of) .csv file(s).

    Examples
    ---------

    >>> from spectrochempy import NDDataset
    >>> NDDataset.read_csv('agirdata/P350/TGA/tg.csv')
    NDDataset: [float64] unitless (shape: (y:1, x:3247))

    Additional information can be stored in the dataset if the origin is given
    (known origin for now : tga or omnic)
    # TODO: define some template to allow adding new origins

    >>> NDDataset.read_csv('agirdata/P350/TGA/tg.csv', origin='tga')
    NDDataset: [float64] wt.%% (shape: (y:1, x:3247))

    Sometimes the delimiteur needs to be adjusted

    >>> NDDataset.read_csv('irdata/IR.CSV', directory=prefs.datadir, origin='omnic', csv_delimiter=',')
    NDDataset: [float64] a.u. (shape: (y:1, x:3736))

    Notes
    -----
    This is limited to 1D array - csv file must have two columns [index, data]
    without header

    See Also
    --------
    read : Generic read method
    read_zip, read_jdx, read_matlab, read_omnic, read_opus, read_topspin


    """
    if 'filetypes' not in kwargs.keys():
        kwargs['filetypes'] = ['CSV files (*.csv)']
    kwargs['protocol'] = ['.csv']
    importer = _Importer()
    return importer(*args, **kwargs)


# ======================================================================================================================
# Private functions
# ======================================================================================================================


def _read_csv(*args, **kwargs):

    # read csv file
    dataset, filename = args
    content = kwargs.get('content', None)
    delimiter = kwargs.get("csv_delimiter", prefs.csv_delimiter)

    def _open():
        if content is not None:
            f = io.StringIO(content.decode("utf-8"))
        else:
            f = open(filename, 'r')
        return f

    try:
        fid = _open()
        d = np.loadtxt(fid, unpack=True, delimiter=delimiter)
        fid.close()
    except ValueError:
        # it might be that the delimiter is not correct (default is ','), but
        # french excel export with the french locale for instance, use ";".
        _delimiter = ';'
        try:
            fid = _open()
            if fid:
                fid.close()
            fid = _open()
            d = np.loadtxt(fid, unpack=True, delimiter=_delimiter)
            fid.close()
        except Exception as e:
            # in french, very often the decimal '.' is replaced by a
            # comma:  Let's try to correct this
            if fid:
                fid.close()
            if not isinstance(fid, io.StringIO):
                with open(fid, "r") as fid_:
                    txt = fid_.read()
            else:
                txt = fid.read()
            txt = txt.replace(',', '.')
            fil = io.StringIO(txt)
            try:
                d = np.loadtxt(fil, unpack=True, delimiter=delimiter)
            except Exception:
                raise IOError(
                        '{} is not a .csv file or its structure cannot be recognized')

    # First column is the x coordinates
    coordx = Coord(d[0])

    # create a second coordinate for dimension y of size 1
    coordy = Coord([0])

    # and data is the second column -  we make it a vector
    data = d[1].reshape((1, coordx.size))

    # update the dataset
    dataset.data = data
    dataset.set_coords(y=coordy, x=coordx)

    # set the additional attributes
    name = filename.stem
    dataset.name = kwargs.get('name', name)
    dataset.title = kwargs.get('title', None)
    dataset.units = kwargs.get('units', None)
    dataset.description = kwargs.get('description',
                                     '"name" ' + 'read from .csv file')
    dataset.history = str(datetime.now()) + ':read from .csv file \n'
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    # here we can check some particular format
    origin = kwargs.get('origin', '')
    if 'omnic' in origin:
        # this will be treated as csv export from omnic (IR data)
        dataset = _add_omnic_info(dataset, **kwargs)
    elif 'tga' in origin:
        # this will be treated as csv export from tga analysis
        dataset = _add_tga_info(dataset, **kwargs)
    elif origin:
        origin = kwargs.get('origin', None)
        raise NotImplementedError(f"Sorry, but reading a csv file with '{origin}' origin is not implemented. "
                                  "Please, remove or set the keyword 'origin'\n "
                                  '(Up to now implemented csv files are: `omnic`, `tga`)')
    return dataset


# .............................................................................
def _add_omnic_info(dataset, **kwargs):

    # get the time and name
    name = desc = dataset.name

    # modify the dataset metadata
    dataset.units = 'absorbance'
    dataset.title = 'Absorbance'
    dataset.name = name
    dataset.description = ('Dataset from .csv file: {}\n'.format(desc))
    dataset.history = str(datetime.now()) + ':read from omnic exported csv file \n'

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    # x axis
    dataset.x.units = 'cm^-1'

    # y axis ?
    if '_' in name:
        name, dat = name.split('_')
        # if needed convert weekday name to English
        dat = dat.replace('Lun', 'Mon')
        dat = dat[:3].replace('Mar', 'Tue') + dat[3:]
        dat = dat.replace('Mer', 'Wed')
        dat = dat.replace('Jeu', 'Thu')
        dat = dat.replace('Ven', 'Fri')
        dat = dat.replace('Sam', 'Sat')
        dat = dat.replace('Dim', 'Sun')
        # convert month name to English
        dat = dat.replace('Aout', 'Aug')

        # get the dates
        acqdate = datetime.strptime(dat, "%a %b %d %H-%M-%S %Y")

        # Transform back to timestamp for storage in the Coord object
        # use datetime.fromtimestamp(d, timezone.utc))
        # to transform back to datetime obkct
        timestamp = acqdate.timestamp()

        dataset.y = Coord(np.array([timestamp]), name='y')
        dataset.set_coordtitles(y='Acquisition timestamp (GMT)', x='Wavenumbers')
        dataset.y.labels = np.array([[acqdate], [name]])
        dataset.y.units = 's'

    return dataset


def _add_tga_info(dataset, **kwargs):

    # for TGA, some information are needed.
    # we add them here
    dataset.x.units = 'hour'
    dataset.units = 'weight_percent'
    dataset.x.title = 'Time-on-stream'
    dataset.title = 'Mass change'

    return dataset


# Register the readers
# ----------------------------------------------------------------------------------------------------------------------
_Importer._read_csv = staticmethod(_read_csv)



# make also classmethod
# ----------------------------------------------------------------------------------------------------------------------
# NDIO.read_csv = read_csv


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
