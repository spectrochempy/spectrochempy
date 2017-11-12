# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

import os
import shutil

import warnings
import csv
import numpy as np

from io import StringIO

from datetime import datetime, timezone
import locale
locale.setlocale(locale.LC_ALL, 'en_US')  # to avoid problems with date format

# -----------------------------------------------------------------------------
# Local imports
# -----------------------------------------------------------------------------
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoords import Coord
from spectrochempy.core.processors.concatenate import concatenate, stack
from spectrochempy.application import options, log
from spectrochempy.utils import (readfilename, unzip, is_sequence,
                                SpectroChemPyWarning)

__all__ = ['read_zip', 'read_csv']

# =============================================================================
# read_zip
# =============================================================================

def read_zip(source, filename='', **kwargs):
    """Open a zipped list of .csv files
    and set data/metadata in the current dataset

    Parameters
    ----------

    source : NDDataset

        The dataset to store the data and the metadata read from the spg file

    filename: str

        filename of the file to load

    directory: `str` [optional, default=""].
        From where to read the specified filename. If not sperfied, read i the current directory.

    Examples
    --------
    >>> from spectrochempy.api import NDDataset, scpdata # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
        SpectroChemPy's API ...
    >>> A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip', directory=data)
    >>> print(A) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
    --------------------------------------------------------------------------------
      name/id: FTIR ...

    """

    return _read(source, filename,
              filter='zip file (*.zip);', **kwargs)


# =============================================================================
# read_csv
# =============================================================================

def read_csv(source, filename='', **kwargs):
    """Open a .csv file and set data/metadata in the current dataset

    Notes
    -----
    This is limited to 1D array - csv file must have two columns [index, data]
    without header  #TODO: imporve this for 2D with header

    Parameters
    ----------

    source : NDDataset

        The dataset to store the data and the metadata read from the spg file

    filename: str

        filename of the file to load

    directory: `str` [optional, default=""].
        From where to read the specified filename. If not sperfied, read i the current directory.

    Examples
    --------
    >>> from spectrochempy.api import NDDataset, scpdata # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
        SpectroChemPy's API ...
    >>> A = NDDataset.read_csv('agirdata/A350/TGA/tg.csv', directory=data)
    >>> print(A) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <BLANKLINE>
    --------------------------------------------------------------------------------
      name/id: tg ...

    """

    return _read(source, filename,
                 filter='csv file (*.csv);', **kwargs)


# =============================================================================
# private functions
# =============================================================================

def _read(source, filename='',
          filter='zip file (*.zip);', **kwargs):

    # check if the first parameter is a dataset
    # because we allow not to pass it
    if not isinstance(source, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(source, str) and source!='':
            filename = source

        source = NDDataset()  # create a NDDataset

    directory = kwargs.get("directory", options.scpdata)
    if not os.path.exists(directory):
        raise IOError("directory doesn't exists!")

    if os.path.isdir(directory):
        filename = os.path.expanduser(os.path.join(directory, filename))
    else:
        warnings.warn('Provided directory is a file, '
                      'so we use its parent directory', SpectroChemPyWarning)
        filename = os.path.join(os.path.dirname(directory), filename)


    # open file dialog box if necessary
    files = readfilename(filename,
                         directory = directory,
                         filter=filter)

    if not files:
        return None

    sources = []

    for extension in files.keys():

        for filename in files[extension]:
            if extension == '.zip':
                # zip returns a list, so we extend the list of sources
                sources.extend(_read_zip(source, filename, **kwargs))

            elif extension == '.csv':
                csv = _read_csv(source, filename, **kwargs)
                # check is it is a list of sources or a single
                if isinstance(csv, NDDataset):
                    sources.append(csv)
                elif is_sequence(csv):
                    sources.extend(csv)
            else:
                # try another format!
                dat = source.read(filename, protocol=extension[1:], **kwargs)
                if isinstance(dat, NDDataset):
                    sources.append(dat)
                elif is_sequence(dat):
                    sources.extend(dat)

    # and stack them into a single file - this assume they are compatibles
    new = stack(sources)

    # now we return the results
    return new


def _read_zip(source, filename, **kwargs):

    if not os.path.exists(filename):
        print('Sorry but this filename (%s) does not exists!'%filename)
        return None

    temp = os.path.join(os.path.dirname(filename), '~temp')
    basename = os.path.splitext(os.path.basename(filename))[0]
    unzip(filename, temp)
    unzipfilename = os.path.join(temp, basename, basename)

    # get all .csv in directory
    filelist = os.listdir(unzipfilename)
    filelist.sort()

    # read all .csv files?
    only = kwargs.pop('only',None)
    if only is not None:
        filelist = filelist[:only+1]
    sources = []

    for i, f in enumerate(filelist):
        f = os.path.basename(f)

        if os.path.splitext(f)[-1] != '.csv':
            continue # bypass non-csv files

        pth = os.path.join(unzipfilename,f)
        log.debug('reading %s: %s' % (f,pth) )

        sources.append(_read_csv(source, pth, **kwargs))

    try:
        shutil.rmtree(temp)
    except:
        pass

    return sources

def _read_csv(source, filename='', **kwargs):

    # this is limited to 1D array (two columns reading!)

    if not os.path.exists(filename):
        raise IOError("{} file doesn't exists!".format(filename))

    new = source.copy() # important
    delimiter = kwargs.get("csv_delimiter", options.csv_delimiter)
    try:
        d = np.loadtxt(filename, delimiter=delimiter)
    except ValueError:
        # it might be that the delimiter is not correct (default is ','), but
        # french excel export for instance, use ";".
        delimiter =';'
        # in this case, in french, very often the decimal '.' is replaced by a
        # comma:  Let's try to correct this
        with open(filename, "r") as f:
            txt = f.read()
            txt = txt.replace(',','.')
            fil = StringIO(txt)
            try:
                d = np.loadtxt(fil, delimiter=delimiter)
            except:
                raise IOError(
                  '{} is not a .csv file or its structure cannot be recognized')

    # transpose d so the the rows becomes the last dimensions
    d = d.T

    # First row should now be the coordinates, and data the rest of the array
    coord1 = d[0]
    data = d[1]

    # create the dataset
    new.data = data
    name = os.path.splitext(os.path.basename(filename))[0]
    new.name = kwargs.get('name', name)
    new.title = kwargs.get('title', None)
    new.units = kwargs.get('units', None)
    new.description = kwargs.get('description',
                                    '"name" '+ 'read from .csv file')
    coord0 = Coord(labels=[new.name])
    new.coordset = [coord0, coord1] #[coord0, coord1]
    new.history = str(datetime.now()) + ':read from .csv file \n'
    new._date = datetime.now()
    new._modified = new.date

    # here we can check some particular format
    origin = kwargs.get('origin', '')
    if 'omnic' in origin:
        # this will be treated as csv export from omnic (IR data)
        new = _add_omnic_info(new, **kwargs)

    return new

def _add_omnic_info(source, **kwargs):

    # get the time and name
    name = desc = source.name
    name, dat =  name.split('_')

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
    source.units = 'absorbance'
    source.title = 'Absorbance'
    source.name = name
    xaxis = source.coordset[-1]
    source.coordset = [np.array([timestamp]), xaxis]
    source.coordset.titles = ('Acquisition timestamp (GMT)', 'Wavenumbers')
    source.coordset[1].units = 'cm^-1'
    source.coordset[0].labels = np.array([[acqdate],[name]])
    source.coordset[0].units = 's'

    # Set description and history
    source.description = (
        'Dataset from .csv file : {}\n'.format(desc))

    source.history = str(datetime.now()) + ':read from spg file \n'

    # Set the NDDataset date
    source._date = datetime.now()
    source._modified = source.date

    return source

#===============================================================================
# tests
#===============================================================================
if __name__ == '__main__':

    from spectrochempy.api import (NDDataset, data, options, DEBUG, INFO, ERROR,
                                   figure, show)


    options.log_level = ERROR

    # A = NDDataset.read_zip('agirdata/A350/FTIR/FTIR.zip',
    #                        directory=data,
    #                        origin='omnic_export')
    # print(A)
    # A.plot_stack()


    B = NDDataset.read_csv('agirdata/A350/TGA/tg.csv', directory=data)
    print(B)

    B = B[-0.5:60.0]

    B.x.units = 'hour'
    B.x.title = 'time on stream'
    B.units = 'weight_percent'
    B.title = 'loss of mass'

    B.plot()
    show()

    # to open the file dialog editor
    #C = NDDataset.read_csv(directory=data)
    #print(C)