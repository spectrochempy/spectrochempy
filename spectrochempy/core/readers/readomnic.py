# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""This module to extend NDDataset with the import methods method.

"""
__all__ = ['read_omnic', 'read_spg', 'read_spa']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import os
from datetime import datetime, timezone, timedelta

import numpy as np

from spectrochempy.core import debug_
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndio import NDIO
from spectrochempy.utils import readfilename


# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------------------------------------------------

# ======================================================================================================================
# Public functions
# ======================================================================================================================

# .............................................................................
def read_omnic(dataset=None, **kwargs):
    """Open a Thermo Nicolet with extension ``.spg`` or a list of ``.spa`` files
     and set data/metadata in the current dataset

    Parameters
    ----------
    filename : `None`, `str`, or list of `str`
        Filename of the file(s) to load. If `None` : opens a dialog box to select
        ``.spa`` or ``.spg`` files. If `str` : a single filename. It list of str :
        a list of filenames.
    directory : str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the defaults datadir.
    sortbydate : bool, optional, default=True.
        Sort spectra by acquisition date

    Returns
    -------
    dataset : |NDDataset|
        A dataset corresponding to the ``.spg`` file or the set of ``.spa``
        files. A list of datasets is returned if several ``.spg`` files are passed.

    Examples
    --------
    >>> A = NDDataset.read_omnic('irdata/nh4y-activation.spg')
    >>> print(A)
    <BLANKLINE>
      id : NH4Y-activation.SPG ...


    """
    debug_("reading omnic files")

    # filename will be given by a keyword parameter except if the first parameters is already the filename
    filename = kwargs.get('filename', None)

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, (str, list)) and dataset != '':
            filename = dataset

        dataset = NDDataset()  # create an instance of NDDataset

    # check if directory was specified
    directory = kwargs.get("directory", None)
    sortbydate = kwargs.get("sortbydate", True)

    # returns a list of files to read
    files = readfilename(filename,
                         directory=directory,
                         filetypes=['OMNIC files (*.spa, *.spg)',
                                    'all files (*)'])

    if not files:
        # there is no files, return nothing
        return None

    datasets = []
    for extension in files.keys():

        extension = extension.lower()

        if extension == '.spg':
            for filename in files[extension]:
                # debug_("reading omnic spg file")
                datasets.append(_read_spg(dataset, filename, sortbydate=sortbydate))

        elif extension == '.spa':
            # debug_("reading omnic spa files")
            datasets.append(_read_spa(dataset, files[extension], sortbydate=sortbydate))
        else:
            # try another format!
            datasets = dataset.read(filename, protocol=extension[1:], sortbydate=sortbydate, **kwargs)

    if len(datasets) == 1:
        return datasets[0]  # a single dataset is returned

    # several datasets returned (only if several .spg files have been passed)
    return datasets


# alias
read_spg = read_omnic
read_spa = read_omnic

# make also classmethod
NDIO.read_spg = read_omnic
NDIO.read_spa = read_omnic


# ======================================================================================================================
# private functions
# ======================================================================================================================

# .............................................................................
def _readbtext(f, pos):
    # Read some text in binary file, until b\0\ is encountered.
    # Returns utf-8 string
    f.seek(pos)  # read first byte, ensure entering the while loop
    btext = f.read(1)
    while not (btext[len(
            btext) - 1] == 0):  # while the last byte of btext differs from zero
        btext = btext + f.read(1)  # append 1 byte

    btext = btext[0:len(btext) - 1]  # cuts the last byte
    try:
        text = btext.decode(encoding='utf-8')  # decode btext to string
    except UnicodeDecodeError:
        try:
            text = btext.decode(encoding='latin_1')
        except UnicodeDecodeError:
            text = btext.decode(encoding='utf-8', errors='ignore')
    return text


# .............................................................................
def _read_spg(dataset, filename, **kwargs):
    # read spg file

    sortbydate = kwargs.get('sortbydate', True)
    with open(filename, 'rb') as f:

        # Read title:
        # The file title starts at position hex 1e = decimal 30.
        # Its max length is 256 bytes and it is followed by at least
        # one \0. It is the original filename under which the group has
        # been saved: it won't match with the actual filename if a subsequent
        # renaming has been done in e.g. Windows.

        spg_title = _readbtext(f, 30)

        # The acquisition date (GMT) of 1st spectrum at hex 128 = decimal 296.
        # The format is HFS+ 32 bit hex value, little endian

        f.seek(296)

        # Count the number of spectra
        # From hex 120 = decimal 304, individual spectra are described
        # by blocks of lines starting with "key values",
        # for instance hex[02 6a 6b 69 1b 03 82] -> dec[02 106  107 105 27 03 130]
        # Each of theses lines provides positions of data and metadata in the file:
        #
        #     key: hex 02, dec  02: position of spectral header (=> nx,
        #                                 firstx, lastx, nscans, nbkgscans)
        #     key: hex 03, dec  03: intensity position
        #     key: hex 04, dec  04: user text position
        #     key: hex 1B, dec  27: position of History text
        #     key: hex 69, dec 105: ?
        #     key: hex 6a, dec 106: ?
        #     key: hex 6b, dec 107: position of spectrum title, the acquisition
        #                                 date follows at +256(dec)
        #     key: hex 80, dec 128: ?
        #     key: hex 82, dec 130: ?
        #
        # the number of line per block may change from one omnic version to another,
        # but the total number of lines is given at hex 294, hence allowing counting
        # nummber of spectra:

        # np.nonzero((code == 2)) ; np.count_nonzero((a==2))

        # read total number of lines
        f.seek(294)
        nlines = np.fromfile(f, 'uint16', count=1)

        # read "key values"
        pos = 304
        keys = np.zeros(nlines)
        for i in range(nlines[0]):
            f.seek(pos)
            keys[i] = np.fromfile(f, dtype='uint8', count=1)[0]
            pos = pos + 16

        # count the number of occurences of the key '02' == number of spectra
        nspec = np.count_nonzero((keys == 2))

        if nspec == 0:
            raise IOError('Error : File format not recognized'
                          ' - information markers not found')

        # Get xaxis (e.g. wavenumbers)
        # container to hold values
        nx, firstx, lastx = np.zeros(nspec, 'int'), np.zeros(nspec, 'float'), np.zeros(nspec, 'float')

        # Extracts positions of '02' keys
        key_is_02 = (keys == 2)  # ex: [T F F F F T F (...) F T ....]'
        indices02 = np.nonzero(key_is_02)  # ex: [1 9 ...]
        position02 = 304 * np.ones(len(indices02[0]), dtype='int') + 16 * indices02[0]

        # ex: [304 432 ...]
        for i in range(nspec):
            f.seek(position02[i] + 2)  # go to line and skip 2 bytes
            info_pos = np.fromfile(f, dtype='uint32', count=1)[0]
            nx_pos = info_pos + 4
            firstx_pos = info_pos + 16
            lastx_pos = info_pos + 20
            # other positions of potential interest:
            #   nscan_pos = info_pos + 36;
            #   nbkgscan_pos = info_pos + 52;

            f.seek(nx_pos)
            nx[i] = np.fromfile(f, 'uint32', 1)
            f.seek(firstx_pos)
            firstx[i] = np.fromfile(f, 'float32', 1)
            f.seek(lastx_pos)
            lastx[i] = np.fromfile(f, 'float32', 1)

        # check the consistency of xaxis
        if np.ptp(nx) != 0:
            raise ValueError('Inconsistant data set'
                             ' - number of wavenumber per spectrum should be identical')

        elif np.ptp(firstx) != 0:
            raise ValueError('Inconsistant data set'
                             ' - the x axis should start at same value')

        elif np.ptp(lastx) != 0:
            raise ValueError('Inconsistant data set'
                             ' - the x axis should end at same value')

        xaxis = np.around(np.linspace(firstx[0], lastx[0], nx[0]), 3)

        # now the intensity data

        # container to hold values
        intensity_pos, intensity_size = np.zeros(nspec, 'int'), np.zeros(nspec, 'int')

        # Extracts positions of '02' keys
        key_is_03 = (keys == 3)
        indices03 = np.nonzero(key_is_03)
        position03 = 304 * np.ones(len(indices03[0]), dtype='int') + 16 * indices03[0]

        # Read number of spectral intensities
        for i in range(nspec):
            # determines the position of informatioon
            f.seek(position03[i] + 2)  # go to line and skip 2 bytes
            intensity_pos[i] = np.fromfile(f, 'uint32', 1)
            f.seek(position03[i] + 6)
            intensity_size[i] = np.fromfile(f, 'uint32', 1)

        # check the consistency of intensities
        # (probably redundent w/ xaxis check above)
        if np.ptp(intensity_size) != 0:
            raise ValueError('Inconsistent data set'
                             ' - number of data per spectrum should be identical')

        nintensities = int(intensity_size[0] / 4)  # 4 = size of uint32

        if nintensities != nx[0]:
            raise ValueError('Inconsistent file'
                             ' - number of wavenumber per spectrum should be equal to number of intensities')

        # Read spectral intensities
        # ..............................................................................................................
        data = np.zeros((nspec, nintensities), dtype='float32')
        for i in range(nspec):
            f.seek(intensity_pos[i])
            data[i, :] = np.fromfile(f, 'float32', int(nintensities))
        # ..............................................................................................................

        # Get spectra titles & acquisition dates & history text
        # container to hold values
        alltitles, allacquisitiondates = [], []
        alltimestamps, allhistories = [], []

        # extract positions of '6B' keys (spectra titles & acquisition dates)
        key_is_6B = (keys == 107)
        indices6B = np.nonzero(key_is_6B)
        position6B = 304 * np.ones(len(indices6B[0]), dtype='int') + 16 * indices6B[0]

        # read spectra titles and acquisition date
        for i in range(nspec):
            # determines the position of informatioon
            f.seek(position6B[i] + 2)  # go to line and skip 2 bytes
            spa_title_pos = np.fromfile(f, 'uint32', 1)

            # read filename
            spa_title = _readbtext(f, spa_title_pos[0])
            alltitles.append(spa_title)

            # and the acquisition date
            f.seek(spa_title_pos[0] + 256)
            timestamp = np.fromfile(f, dtype=np.uint32, count=1)[
                0]  # days since 31/12/1899, 00:00
            acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=int(timestamp))
            allacquisitiondates.append(acqdate)
            timestamp = acqdate.timestamp()
            # Transform back to timestamp for storage in the Coord object
            # use datetime.fromtimestamp(d, timezone.utc))
            # to transform back to datetime obkct

            alltimestamps.append(timestamp)

            # extract positions of '1B' codes (history text
            #  -- sometimes absent, e.g. peakresolve)
            key_is_1B = (keys == 27)
            indices1B = np.nonzero(key_is_1B)
            position1B = 304 * np.ones(len(indices1B[0]), dtype='int') + 16 * indices6B[0]

            if len(position1B) != 0:
                # read history texts
                for j in range(nspec):
                    # determine the position of information
                    f.seek(position1B[j] + 2)
                    history_pos = np.fromfile(f, 'uint32', 1)

                    # read history
                    history = _readbtext(f, history_pos[0])
                    allhistories.append(history)

    # Create Dataset Object of spectral content
    dataset.data = data
    dataset.units = 'absorbance'
    dataset.title = 'Absorbance'
    dataset.name = spg_title
    dataset.filename = os.path.basename(filename).split('.')[0]

    # now add coordinates
    _x = Coord(xaxis, title='Wavenumbers', units='cm^-1')
    _y = Coord(alltimestamps,
               title='Acquisition timestamp (GMT)',
               units='s',
               labels=(allacquisitiondates, alltitles))
    dataset.set_coords(y=_y, x=_x)

    # Set origin, description and history
    dataset.origin = "omnic"
    dataset.description = (
            'Dataset from spg file : ' + spg_title + ' \n'
            + 'History of the 1st spectrum : ' + allhistories[0])

    dataset.history = str(datetime.now()) + ':read from spg file \n'

    if sortbydate:
        dataset.sort(dim='y', inplace=True)
        dataset.history = 'Sorted'

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    # debug_("end of reading")

    return dataset


# .............................................................................
def _read_spa(dataset, filenames, **kwargs):
    nspec = len(filenames)

    # containers to hold values
    nx = np.zeros(nspec, 'int')
    firstx = np.zeros(nspec, 'float')
    lastx = np.zeros(nspec, 'float')
    allintensities = []
    alltitles = []
    allacquisitiondates = []
    alltimestamps = []
    allhistories = []

    for i, _filename in enumerate(filenames):

        with open(_filename, 'rb') as f:

            # Read title:
            # The file title  starts at position hex 1e = decimal 30.
            # Its max length is 256 bytes and it is followed by at least
            # one \0. It is the original filename under which the group has
            # been saved: it won't match with the actual filename if a subsequent
            # renaming has been done in e.g. Windows.

            alltitles.append(_readbtext(f, 30))

            # The acquisition date (GMT) is at hex 128 = decimal 296.
            # The format is HFS+ 32 bit hex value, little endian

            f.seek(296)

            # days since 31/12/1899, 00:00
            timestamp = np.fromfile(f, dtype=np.uint32, count=1)[0]
            acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + \
                      timedelta(seconds=int(timestamp))
            allacquisitiondates.append(acqdate)
            timestamp = acqdate.timestamp()
            # Transform back to timestamp for storage in the Coord object
            # use datetime.fromtimestamp(d, timezone.utc))
            # to transform back to datetime object

            alltimestamps.append(timestamp)

            # From hex 120 = decimal 304, the spectrum is described
            # by blocks of lines starting with "key values",
            # for instance hex[02 6a 6b 69 1b 03 82] -> dec[02 106  107 105 27 03 130]
            # Each of theses lines provides positions of data and metadata in the file:
            #
            #     key: hex 02, dec  02: position of spectral header (=> nx,
            #                                 firstx, lastx, nscans, nbkgscans)
            #     key: hex 03, dec  03: intensity position
            #     key: hex 04, dec  04: user text position
            #     key: hex 1B, dec  27: position of History text
            #     key: hex 69, dec 105: ?
            #     key: hex 6a, dec 106: ?
            #     key: hex 80, dec 128: ?
            #     key: hex 82, dec 130: ?
            #

            gotinfos = [False, False,
                        False]  # spectral header, intensity, history
            # scan "key values"
            pos = 304
            #        keys = np.zeros((nlines))
            #        for i in range(nlines):
            #            f.seek(pos)
            #            keys[i] = np.fromfile(f, dtype = 'uint8', count = 1)[0]
            #        pos = pos + 16

            while not (all(gotinfos)):
                f.seek(pos)
                key = np.fromfile(f, dtype='uint8', count=1)[0]
                if key == 2:
                    f.seek(pos + 2)  # skip 2 bytes
                    info_pos = np.fromfile(f, dtype='uint32', count=1)[0]
                    nx_pos = info_pos + 4
                    firstx_pos = info_pos + 16
                    lastx_pos = info_pos + 20
                    # other positions of potential interest:
                    #   nscan_pos = info_pos + 36;
                    #   nbkgscan_pos = info_pos + 52;

                    f.seek(nx_pos)
                    nx[i] = np.fromfile(f, 'uint32', 1)[0]
                    f.seek(firstx_pos)
                    firstx[i] = np.fromfile(f, 'float32', 1)[0]
                    f.seek(lastx_pos)
                    lastx[i] = np.fromfile(f, 'float32', 1)[0]

                    xaxis = np.around(
                        np.linspace(firstx[0], lastx[0], nx[0]), 3)
                    gotinfos[0] = True

                elif key == 3:
                    f.seek(pos + 2)  # skip 2 bytes
                    intensity_pos = np.fromfile(f, 'uint32', 1)[0]
                    f.seek(pos + 6)
                    intensity_size = np.fromfile(f, 'uint32', 1)[0]

                    nintensities = int(intensity_size / 4)
                    # Read spectral intensities
                    f.seek(intensity_pos)
                    allintensities.append(
                        np.fromfile(f, 'float32', int(nintensities)))
                    gotinfos[1] = True

                elif key == 27:
                    f.seek(pos + 2)
                    history_pos = np.fromfile(f, 'uint32', 1)[0]
                    # read history
                    history = _readbtext(f, history_pos)
                    allhistories.append(history)
                    gotinfos[2] = True

                elif not key:
                    break

                pos = pos + 16

    # check the consistency of xaxis
    if np.ptp(nx) != 0:
        raise ValueError(
            'Error : Inconsistent data set - number of wavenumber per spectrum should be identical')
    elif np.ptp(firstx) != 0:
        raise ValueError(
            'Error : Inconsistent data set - the x axis should start at same value')
    elif np.ptp(lastx) != 0:
        raise ValueError(
            'Error : Inconsistent data set - the x axis should end at same value')

    # load into the  NDDataset Object of spectral content
    dataset.data = np.array(allintensities)
    dataset.units = 'absorbance'
    dataset.title = 'Absorbance'
    dataset.name = ' ... '.join({alltitles[0], alltitles[-1]})
    dataset._date = dataset._modified = datetime.now()

    # Create Dataset Object of spectral content
    # now add coordinates
    _x = Coord(xaxis, title='Wavenumbers', units='cm^-1')
    _y = Coord(alltimestamps, title='Acquisition timestamp (GMT)', units='s', labels=(allacquisitiondates, alltitles))
    dataset.set_coords(y=_y, x=_x)

    # Set origin, description and history
    dataset.origin = "omnic"
    dataset.description = "Dataset from {0} spa files : '{1}'\nHistory of the {2}spectrum : {3}".format(
        nspec, ' ... '.join({filenames[0], filenames[-1]}), '1st ' if nspec > 1 else '', allhistories[0])

    dataset.history = str(datetime.now()) + ':read from spa files \n'

    if kwargs.get('sortbydate', True):
        dataset.sort(dim=0, inplace=True)
        dataset.history = 'Sorted'

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    # debug_("end of reading")

    # return the dataset
    return dataset


if __name__ == '__main__':
    pass
