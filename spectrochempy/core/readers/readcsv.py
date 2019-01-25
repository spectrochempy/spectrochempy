# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

__all__ = ['read_zip', 'read_csv']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------

import os
import shutil
import warnings
from io import StringIO
from datetime import datetime
import locale

try:
    locale.setlocale(locale.LC_ALL, 'en_US')  # to avoid problems with date format
except:
<<<<<<< Updated upstream
    warnings.warn('Could not set locale: en_US')
=======
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.utf-8')  # idem
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # idem
        except:
            warnings.warn('Could not set locale: en_US')
>>>>>>> Stashed changes

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np
from numpy.lib.npyio import zipfile_factory, NpzFile

# -----------------------------------------------------------------------------
# Local imports
# -----------------------------------------------------------------------------
from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.core.processors.concatenate import concatenate
from spectrochempy.application import log, general_preferences as prefs
from spectrochempy.utils import (readfilename, unzip, is_sequence,
                                 SpectroChemPyWarning)


# =============================================================================
# read_zip
# =============================================================================

def read_zip(dataset=None, **kwargs):
    """Open a zipped list of .csv files  and set data/metadata in the
    current dataset

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to store the data and the metadata read from the spg file
    filename: str
        filename of the file to load
    directory: str, optional, default="".
        From where to read the specified filename. If not sperfied, read in
        the current directory.

    Returns
    -------
    |NDDataset|

    Examples
    --------
    >>> A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip', origin='omnic')
    >>> print(A)
    <BLANKLINE>
      name/id:  ...

    """
    log.debug("reading zipped folder of *.csv files")

    # filename will be given by a keyword parameter except the first parameters
    # is already the filename
    filename = kwargs.pop('filename', None)

    nd = _read(dataset, filename=filename, filter='zip file (*.zip);', **kwargs)

    return nd


# =============================================================================
# Public functions
# =============================================================================

# .............................................................................
def read_csv(dataset=None, **kwargs):
    """Open a \*.csv file or a list of \*.csv files and set data/metadata
    in the current dataset

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to store the data and the metadata read from the spg file
    filename: str
        filename of the file to load
    directory: str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the current directory then it the test directory.

    Returns
    -------
    dataset: |NDDataset|

    Examples
    --------
    >>> A = NDDataset.read_csv('agirdata/A350/TGA/tg.csv')
    >>> print(A)
    <BLANKLINE>
      name/id: ...

    Notes
    -----
    This is limited to 1D array - csv file must have two columns [index, data]
    without header

    """
    #TODO: to allow header and nd-data

    log.debug("reading csv files")

    # filename will be given by a keyword parameter except the first parameters
    # is already the filename
    filename = kwargs.pop('filename', None)

    nd = _read(dataset, filename=filename, filter='csv file (*.csv)', **kwargs)

    return nd


# =============================================================================
# private functions
# =============================================================================

# .............................................................................
def _read(dataset, filename, **kwargs):
    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, str) and dataset != '':
            filename = dataset

        dataset = NDDataset()  # create a NDDataset

    # check if directory was specified
    directory = kwargs.get("directory", None)

    # returns a list of files to read
    filter = kwargs.get('filter', ['All files (*)'])
    files = readfilename(filename,
                         directory=directory,
                         filter=filter)

    if not files:
        # there is no files, return nothing
        return None

    datasets = []
    for extension in files.keys():

        for filename in files[extension]:
            if extension == '.zip':
                # zip returns a list, so we extend the list of datasets
                datasets.extend(_read_zip(dataset, filename, **kwargs))

            elif extension == '.csv':
                csv = _read_csv(dataset, filename, **kwargs)
                # check is it is a list of datasets or a single
                if isinstance(csv, NDDataset):
                    datasets.append(csv)
                elif is_sequence(csv):
                    datasets.extend(csv)
            else:
                # try another format!
                dat = dataset.read(filename, protocol=extension[1:],
                                   sortbydate=True, **kwargs)
                if isinstance(dat, NDDataset):
                    datasets.append(dat)
                elif is_sequence(dat):
                    datasets.extend(dat)

    # and concatenate them into a single file - this assume they are compatibles
    if len(datasets) > 1:
        new = concatenate(datasets, axis=0)
    else:
        new = datasets[0]

    # now we return the results
    return new


# .............................................................................
def _read_zip(dataset, filename, **kwargs):
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

        log.debug('reading %s ...' % (f))

        datasets.append(_read_csv(dataset, filename=f, fid=obj[f], **kwargs))
        if len(datasets) + 1 > only:
            break

    try:
        shutil.rmtree(temp)
    except:
        pass

    return datasets


# .............................................................................
def _read_csv(dataset, filename='', fid=None, **kwargs):
    # this is limited to 1D array (two columns reading!)
    # TODO: improve this for 2D with header

    if not isinstance(fid, bytes) and not os.path.exists(filename):
        raise IOError("{} file doesn't exists!".format(filename))

    new = dataset.copy()  # important
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

    # First row should now be the coordinates, and data the rest of the array
    coord0 = d[0]
    data = d[1]

    # create the dataset
    new.data = data
    name = os.path.splitext(os.path.basename(filename))[0]
    new.name = kwargs.get('name', name)
    new.title = kwargs.get('title', None)
    new.units = kwargs.get('units', None)
    new.description = kwargs.get('description',
                                 '"name" ' + 'read from .csv file')
    new.coordset = [coord0]
    new.history = str(datetime.now()) + ':read from .csv file \n'
    new._date = datetime.now()
    new._modified = new.date

    # here we can check some particular format
    origin = kwargs.get('origin', '')
    if 'omnic' in origin:
        # this will be treated as csv export from omnic (IR data)
        new._data = new.data[np.newaxis]  # add a dimension
        new = _add_omnic_info(new, **kwargs)

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
    xaxis = dataset.coordset[-1]
    dataset.coordset = [np.array([timestamp]), xaxis]
    dataset.coordset.titles = ('Acquisition timestamp (GMT)', 'Wavenumbers')
    dataset.coordset[1].units = 'cm^-1'
    dataset.coordset[0].labels = np.array([[acqdate], [name]])
    dataset.coordset[0].units = 's'

    # Set description and history
    dataset.description = (
        'Dataset from .csv file : {}\n'.format(desc))

    dataset.history = str(datetime.now()) + ':read from spg file \n'

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    return dataset


# ===============================================================================
# tests
# ===============================================================================
if __name__ == '__main__':
    pass
