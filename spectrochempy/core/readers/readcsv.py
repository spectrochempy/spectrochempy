# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['read_zip', 'read_csv']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import shutil
import warnings
from io import StringIO
from datetime import datetime
import locale

try:
    locale.setlocale(locale.LC_ALL, 'en_US')  # to avoid problems with date format
except:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.utf8')  # to avoid problems with date format
    except:
        warnings.warn('Could not set locale: en_US or en_US.utf8')

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from numpy.lib.npyio import NpzFile

# ----------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.processors.concatenate import stack
from spectrochempy.core import general_preferences as prefs
from spectrochempy.utils import (readfilename, is_sequence)

# ======================================================================================================================
# Public functions
# ======================================================================================================================

def read_zip(*args, **kwargs):
    """Open a zipped list of .csv files  and set data/metadata in the
    current dataset

    Parameters
    ----------
    filename : str
        filename of the file to load
    directory : str, optional, default="".
        From where to read the specified filename. If not sperfied, read in
        the current directory.

    Returns
    -------
    dataset : |NDDataset|

    Examples
    --------
    >>> A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip', origin='omnic')
    >>> print(A)
    <BLANKLINE>
      name/id :  ...

    """
    #debug_("reading zipped folder of *.csv files")

    # filename will be given by a keyword parameter except the first parameters
    # is already the filename
    if args and args[0]:
        filename = args[0]
    else:
        filename = None
    filename = kwargs.pop('filename', filename)

    nd = _read(filename=filename, filter='zip file (*.zip);', **kwargs)

    return nd


# .............................................................................
def read_csv(*args, **kwargs):
    """Open a \*.csv file or a list of \*.csv files and set data/metadata
    in the current dataset

    Parameters
    ----------
    filename : str
        filename of the file to load
    directory : str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the current directory then it the test directory.

    Returns
    -------
    dataset : |NDDataset|

    Examples
    --------
    >>> A = NDDataset.read_csv('agirdata/A350/TGA/tg.csv')
    >>> print(A)
    <BLANKLINE>
      name/id : ...

    Notes
    -----
    This is limited to 1D array - csv file must have two columns [index, data]
    without header

    """
    # TODO: to allow header and nd-data

    # debug_("reading csv files")

    # filename will be given by a keyword parameter except the first parameters
    # is already the filename
    if args and args[0]:
        filename = args[0]
    else:
        filename = None
    filename = kwargs.pop('filename', filename)

    nd = _read(filename=filename, filters=['csv files (*.csv)', 'All files (*)'])

    return nd


# ======================================================================================================================
# private functions
# ======================================================================================================================

# .............................................................................
def _read(filename, **kwargs):
    # check if directory was specified
    directory = kwargs.get("directory", None)

    # returns a list of files to read
    filters = kwargs.get('filters', ['All files (*)'])
    files = readfilename(filename,
                         directory=directory,
                         filters=filters)

    if not files:
        # there is no files, return nothing
        return None

    datasets = []
    for extension in files.keys():

        for filename in files[extension]:
            if extension == '.zip':
                # zip returns a list, so we extend the list of datasets
                datasets.extend(_read_zip(filename, **kwargs))

            elif extension == '.csv':
                csv = _read_csv(filename, **kwargs)
                # check is it is a list of datasets or a single
                if isinstance(csv, NDDataset):
                    datasets.append(csv)
                elif is_sequence(csv):
                    datasets.extend(csv)
            else:
                # try another format!
                dat = NDDataset.read(filename, protocol=extension[1:],
                                     sortbydate=True, **kwargs)
                if isinstance(dat, NDDataset):
                    datasets.append(dat)
                elif is_sequence(dat):
                    datasets.extend(dat)

    # and stack them along the y-dimension into a single file - this assume they are compatible
    if len(datasets) > 1:
        new = stack(datasets)
    else:
        # else, squeeze a unidimensional dataset
        new = datasets[0].squeeze()

    # now we return the results
    return new


# .............................................................................
def _read_zip(filename, **kwargs):
    if not os.path.exists(filename):
        print('Sorry but this filename (%s) does not exists!' % filename)
        return None

    origin = kwargs.get('origin', None)
    if origin is None:
        origin = 'unknown'
        raise NotImplementedError("Sorry, but reading a zip file with origin of "
                                  "type '%s' is not implemented. Please"
                                  "set the keyword 'origin'." % origin)

    temp = os.path.join(os.path.dirname(filename), '~temp')
    basename = os.path.splitext(os.path.basename(filename))[0]

    obj = NpzFile(filename)
    # unzip(filename, temp)
    # unzipfilename = os.path.join(temp, basename, basename)

    # get all .csv in the zip
    # filelist = os.listdir(unzipfilename)
    filelist = sorted(obj.files)

    # read all .csv files?
    only = kwargs.pop('only', len(filelist))

    datasets = []

    for f in filelist:

        if not f.endswith('.csv') or f.startswith('__MACOSX'):
            continue  # bypass non-csv files

        #debug_('reading %s ...' % (f))

        datasets.append(_read_csv(filename=f, fid=obj[f], **kwargs))
        if len(datasets) + 1 > only:
            break

    try:
        shutil.rmtree(temp)
    except:
        pass

    return datasets


# .............................................................................
def _read_csv(filename='', fid=None, **kwargs):
    # this is limited to 1D array (two columns reading!)
    # TODO: improve this for 2D with header

    if not isinstance(fid, bytes) and not os.path.exists(filename):
        raise IOError("{} file doesn't exists!".format(filename))

    delimiter = kwargs.get("delimiter", prefs.csv_delimiter)
    try:
        if isinstance(fid, bytes):
            f = StringIO(fid.decode("utf-8"))
        else:
            f = filename
        d = np.loadtxt(f, delimiter=delimiter)
    except ValueError:
        # it might be that the delimiter is not correct (default is ','), but
        # french excel export with the french locale for instance, use ";".
        _delimiter = ';'
        try:
            d = np.loadtxt(f, delimiter=_delimiter)
        except:
            # in french, very often the decimal '.' is replaced by a
            # comma:  Let's try to correct this
            if not isinstance(f, StringIO):
                with open(f, "r") as f_:
                    txt = f_.read()
            else:
                txt = f.read()
            txt = txt.replace(',', '.')
            fil = StringIO(txt)
            try:
                d = np.loadtxt(fil, delimiter=delimiter)
            except:
                raise IOError(
                    '{} is not a .csv file or its structure cannot be recognized')

    # transpose d so the the rows becomes the last dimensions
    d = d.T

    # First column should now be the x coordinates
    coordx = Coord(d[0])

    # create a second coordinate for dimension x of size 1
    coordy = Coord([0])

    # and data is the second column -  we make it a vector
    data = d[1].reshape((1, coordx.size))

    # create the dataset
    new = NDDataset(data, coords=(coordy, coordx))

    # set the additional attributes
    name = os.path.splitext(os.path.basename(filename))[0]
    new.name = kwargs.get('name', name)
    new.title = kwargs.get('title', None)
    new.units = kwargs.get('units', None)
    new.description = kwargs.get('description',
                                 '"name" ' + 'read from .csv file')
    new.history = str(datetime.now()) + ':read from .csv file \n'
    new._date = datetime.now()
    new._modified = new.date

    # here we can check some particular format
    origin = kwargs.get('origin', '')
    if 'omnic' in origin:
        # this will be treated as csv export from omnic (IR data)
        new = _add_omnic_info(new, **kwargs)
    elif 'tga' in origin:
        # this will be treated as csv export from tga analysis
        new = _add_tga_info(new, **kwargs)

    return new


# .............................................................................
def _add_omnic_info(dataset, **kwargs):
    # get the time and name
    name = desc = dataset.name
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

    # modify the dataset metadata
    dataset.units = 'absorbance'
    dataset.title = 'Absorbance'
    dataset.name = name
    dataset.y = Coord(np.array([timestamp]), name='y')
    dataset.set_coordtitles(y='Acquisition timestamp (GMT)', x='Wavenumbers')
    dataset.x.units = 'cm^-1'
    dataset.y.labels = np.array([[acqdate], [name]])
    dataset.y.units = 's'

    # Set description and history
    dataset.description = (
        'Dataset from .csv file: {}\n'.format(desc))

    dataset.history = str(datetime.now()) + ':read from spg file \n'

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    return dataset


def _add_tga_info(dataset, **kwargs):
    # for TGA, some information are needed.
    # we add them here
    dataset.x.units = 'hour'
    dataset.units = 'weight_percent'
    dataset.x.title = 'Time-on-stream'
    dataset.title = 'Mass change'

    return dataset


# ======================================================================================================================
# tests
# ======================================================================================================================
if __name__ == '__main__':
    pass
