# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
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
from spectrochempy.utils import readfilename, pathclean


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
     and set data/metadata in the current dataset. The collected metatdata are:
     - names of spectra
     - acquisition dates (UTC)
     - units of spectra (absorbance, transmittance, reflectance, Log(1/R), Kubelka-Munk,
     Raman intensity, photoacoustics, volts)
     - units of xaxis (wavenumbers in cm-1, wavelengths in nm or micrometer, Raman shift in cm-1)
     - spectra history (but only incorporated in the NDDataset if single spa is read)

     An error is generated if attempt is made to inconsistent datasets: units of spectra and
     xaxis, limits and number of points of the xaxis.

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
    description: string, default=None
        Custom description

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
    filename = pathclean(kwargs.get('filename', None))

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, (str, list)) and dataset != '':
            filename = dataset

        dataset = NDDataset()  # create an instance of NDDataset

    sortbydate = kwargs.pop("sortbydate", True)

    # returns a list of files to read
    directory = pathclean(kwargs.get("directory", None))
    files = readfilename(filename,
                         directory=directory,
                         filetypes=['OMNIC files (*.spa, *.spg)',
                                    'OMNIC series (*.srs)',
                                    'all files (*)'])

    if not files:
        # there is no files, return nothing
        return None

    datasets = []
    for extension in files.keys():

        extension = extension.lower()

        if extension == '.spg':
            for filename in files[extension]:
                datasets.append(_read_spg(dataset, filename, **kwargs))

        elif extension == '.spa':
            datasets.append(_read_spa(dataset, files[extension], **kwargs))

        elif extension == '.srs':
            for filename in files[extension]:
                datasets.append(_read_srs(dataset, filename, **kwargs))
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


def _readheader02(f, pos):
    # read spectrum header, pos is the position of the 02 key
    # returns a dict
    f.seek(pos + 2)  # go to line and skip 2 bytes
    info_pos = np.fromfile(f, dtype='uint32', count=1)[0]

    # other positions:
    #   nx_pos = info_pos + 4
    #   xaxis unit code = info_pos + 8
    #   data unit code = info_pos + 12
    #   fistx_pos = info_pos + 16
    #   lastx_pos = info_pos + 20
    #   nscan_pos = info_pos + 36;
    #   nbkgscan_pos = info_pos + 52;

    f.seek(info_pos + 4)
    out = {'nx': np.fromfile(f, 'uint32', 1)}

    # read xaxis unit
    f.seek(info_pos + 8)
    key = np.fromfile(f, dtype='uint8', count=1)[0]
    if key == 1:
        out['xunits'] = 'cm ^ -1'
        out['xtitle'] = 'Wavenumbers'
    elif key == 2:
        out['xunits'] = None
        out['xtitle'] = 'Data points'
    elif key == 3:
        out['xunits'] = 'nm'
        out['xtitle'] = 'Wavelengths'
    elif key == 4:
        out['xunits'] = 'um'
        out['xtitle'] = 'Wavelengths'
    elif key == 32:
        out['xunits'] = 'cm^-1'
        out['xtitle'] = 'Raman Shift'
    else:
        out['xunits'] = None
        out['xtitle'] = 'xaxis'
        # warning: 'The nature of data is not recognized, xtitle set to \'xaxis\')

    # read data unit
    f.seek(info_pos + 12)
    key = np.fromfile(f, dtype='uint8', count=1)[0]
    if key == 17:
        out['units'] = 'absorbance'
        out['title'] = 'Absorbance'
    elif key == 16:
        out['units'] = 'percent'
        out['title'] = 'Transmittance'
    elif key == 11:
        out['units'] = 'percent'
        out['title'] = 'Reflectance'
    elif key == 12:
        out['units'] = None
        out['title'] = 'Log(1/R)'
    elif key == 20:
        out['units'] = 'Kubelka_Munk'
        out['title'] = 'Kubelka-Munk'
    elif key == 22:
        out['units'] = 'V'
        out['title'] = 'Volts'
    elif key == 26:
        out['units'] = None
        out['title'] = 'Photoacoustic'
    elif key == 31:
        out['units'] = None
        out['title'] = 'Raman Intensity'
    else:
        out['title'] = None
        out['title'] = 'Intensity'
        # warning: 'The nature of data is not recognized, title set to \'Intensity\')

    f.seek(info_pos + 16)
    out['firstx'] = np.fromfile(f, 'float32', 1)
    f.seek(info_pos + 20)
    out['lastx'] = np.fromfile(f, 'float32', 1)
    f.seek(info_pos + 36)
    out['nscan'] = np.fromfile(f, 'uint32', 1)
    f.seek(info_pos + 52)
    out['nbkgscan'] = np.fromfile(f, 'uint32', 1)

    return out


def _getintensities(f, pos):
    # get intensities from the 03 key
    # returns a ndarray

    f.seek(pos + 2)  # skip 2 bytes
    intensity_pos = np.fromfile(f, 'uint32', 1)[0]
    f.seek(pos + 6)
    intensity_size = np.fromfile(f, 'uint32', 1)[0]
    nintensities = int(intensity_size / 4)

    # Read and return spectral intensities
    f.seek(intensity_pos)
    return np.fromfile(f, 'float32', int(nintensities))


# .............................................................................


def _read_spg(dataset, filename, **kwargs):
    # read spg file

    sortbydate = kwargs.get('sortbydate', True)
    with open(filename, 'rb') as f:

        # Read title:
        # The file title starts at position hex 1e = decimal 30. Its max length is 256 bytes.
        #  It is the original filename under which the group has been saved: it won't match with
        #  the actual filename if a subsequent renaming has been done in the OS.

        spg_title = _readbtext(f, 30)

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
        # number of spectra:

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

        # the number of occurences of the key '02' is number of spectra
        nspec = np.count_nonzero((keys == 2))

        if nspec == 0:
            raise IOError('Error : File format not recognized'
                          ' - information markers not found')

        # container to hold values
        nx, firstx, lastx = np.zeros(nspec, 'int'), np.zeros(nspec, 'float'), np.zeros(nspec, 'float')
        xunits = []
        xtitles = []
        units = []
        titles = []

        # Extracts positions of '02' keys
        key_is_02 = (keys == 2)  # ex: [T F F F F T F (...) F T ....]'
        indices02 = np.nonzero(key_is_02)  # ex: [1 9 ...]
        position02 = 304 * np.ones(len(indices02[0]), dtype='int') + 16 * indices02[0]  # ex: [304 432 ...]

        for i in range(nspec):
            info02 = _readheader02(f, position02[i])
            nx[i] = info02['nx']
            firstx[i] = info02['firstx']
            lastx[i] = info02['lastx']
            xunits.append(info02['xunits'])
            xtitles.append(info02['xtitle'])
            units.append(info02['units'])
            titles.append(info02['title'])

        # check the consistency of xaxis and data units
        if np.ptp(nx) != 0:
            raise ValueError('Error : Inconsistent data set -'
                             ' number of wavenumber per spectrum should be identical')
        elif np.ptp(firstx) != 0:
            raise ValueError('Error : Inconsistent data set - '
                             'the x axis should start at same value')
        elif np.ptp(lastx) != 0:
            raise ValueError('Error : Inconsistent data set -'
                             ' the x axis should end at same value')
        elif len(set(xunits)) != 1:
            raise ValueError('Error : Inconsistent data set - '
                             'data units should be identical')
        elif len(set(units)) != 1:
            raise ValueError('Error : Inconsistent data set - '
                             'x axis units should be identical')

        data = np.ndarray((nspec, nx[0]), dtype='float32')
        # now the intensity data
        # Extracts positions of '03' keys
        key_is_03 = (keys == 3)
        indices03 = np.nonzero(key_is_03)
        position03 = 304 * np.ones(len(indices03[0]), dtype='int') + 16 * indices03[0]

        # Read number of spectral intensities
        for i in range(nspec):
            data[i, :] = _getintensities(f, position03[i])
        # ..............................................................................................................

        # Get spectra titles & acquisition dates:
        # container to hold values
        spectitles, acquisitiondates, timestamps = [], [], []

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
            spectitles.append(spa_title)

            # and the acquisition date
            f.seek(spa_title_pos[0] + 256)
            timestamp = np.fromfile(f, dtype=np.uint32, count=1)[0]  # days since 31/12/1899, 00:00
            acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=int(timestamp))
            acquisitiondates.append(acqdate)
            timestamp = acqdate.timestamp()
            # Transform back to timestamp for storage in the Coord object
            # use datetime.fromtimestamp(d, timezone.utc))
            # to transform back to datetime obkct

            timestamps.append(timestamp)

            # Not used at present
            # -------------------
            # extract positions of '1B' codes (history text
            #  -- sometimes absent, e.g. peakresolve)
            # key_is_1B = (keys == 27)
            # indices1B = np.nonzero(key_is_1B)
            # position1B = 304 * np.ones(len(indices1B[0]), dtype='int') + 16 * indices6B[0]
            # if len(position1B) != 0:
            #    # read history texts
            #    for j in range(nspec):
            #        # determine the position of information
            #        f.seek(position1B[j] + 2)
            #        history_pos = np.fromfile(f, 'uint32', 1)
            #        # read history
            #        history = _readbtext(f, history_pos[0])
            #        allhistories.append(history)

    # Create Dataset Object of spectral content
    dataset.data = data
    dataset.units = units[0]
    dataset.title = titles[0]
    dataset.name = spg_title
    dataset.filename = os.path.basename(filename).split('.')[0]

    # now add coordinates
    _x = Coord(np.around(np.linspace(firstx[0], lastx[0], nx[0]), 3),
               title=xtitles[0],
               units=xunits[0])

    _y = Coord(timestamps,
               title='Acquisition timestamp (GMT)',
               units='s',
               labels=(acquisitiondates, spectitles))

    dataset.set_coords(y=_y, x=_x)

    # Set origin, description and history
    dataset.origin = "omnic"
    dataset.description = kwargs.get('description',
                                     'Dataset from spg file {} : '.format(filename))

    dataset.history = str(datetime.now()) + ':imported from spg file {} ; '.format(filename)

    if sortbydate:
        dataset.sort(dim='y', inplace=True)
        dataset.history = str(datetime.now()) + ':sorted by date'

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    # debug_("end of reading")

    return dataset


def _read_spa(dataset, filenames, **kwargs):
    nspec = len(filenames)

    # containers to hold values
    nx = np.zeros(nspec, 'int')
    firstx = np.zeros(nspec, 'float')
    lastx = np.zeros(nspec, 'float')
    allintensities = []
    xunits = []
    xtitles = []
    units = []
    titles = []
    spectitles = []
    allacquisitiondates = []
    alltimestamps = []
    allhistories = []

    for i, _filename in enumerate(filenames):

        with open(_filename, 'rb') as f:

            # Read title:
            # The file title  starts at position hex 1e = decimal 30. Its max length is 256 bytes. It is the original
            # filename under which the group has  been saved: it won't match with the actual filename if a subsequent
            # renaming has been done in the OS.

            spectitles.append(_readbtext(f, 30))

            # The acquisition date (GMT) is at hex 128 = decimal 296.
            # The format is HFS+ 32 bit hex value, little endian

            f.seek(296)

            # days since 31/12/1899, 00:00
            timestamp = np.fromfile(f, dtype=np.uint32, count=1)[0]
            acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=int(timestamp))
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

            gotinfos = [False, False, False]  # spectral header, intensity, history
            # scan "key values"
            pos = 304
            while not (all(gotinfos)):
                f.seek(pos)
                key = np.fromfile(f, dtype='uint8', count=1)[0]
                if key == 2:
                    info02 = _readheader02(f, pos)
                    nx[i] = info02['nx']
                    firstx[i] = info02['firstx']
                    lastx[i] = info02['lastx']
                    xunits.append(info02['xunits'])
                    xtitles.append(info02['xtitle'])
                    units.append(info02['units'])
                    titles.append(info02['title'])
                    gotinfos[0] = True

                elif key == 3:
                    allintensities.append(_getintensities(f, pos))
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

                pos += 16

    # check the consistency of xaxis and data units
    if np.ptp(nx) != 0:
        raise ValueError(
            'Error : Inconsistent data set - number of wavenumber per spectrum should be identical')
    elif np.ptp(firstx) != 0:
        raise ValueError(
            'Error : Inconsistent data set - the x axis should start at same value')
    elif np.ptp(lastx) != 0:
        raise ValueError(
            'Error : Inconsistent data set - the x axis should end at same value')
    elif len(set(xunits)) != 1:
        raise ValueError(
            'Error : Inconsistent data set - data units should be identical')
    elif len(set(units)) != 1:
        raise ValueError(
            'Error : Inconsistent data set - x axis units should be identical')

    # load spectral content into the  NDDataset
    dataset.data = np.array(allintensities, dtype='float32')
    dataset.units = units[0]
    dataset.title = titles[0]
    dataset.name = ' ... '.join({spectitles[0], spectitles[-1]})
    dataset._date = dataset._modified = datetime.now()

    # now add coordinates
    _x = Coord(np.around(np.linspace(firstx[0], lastx[0], nx[0]), 3),
               title=xtitles[0],
               units=xunits[0])

    _y = Coord(alltimestamps, title='Acquisition timestamp (GMT)', units='s', labels=(allacquisitiondates, titles))
    dataset.set_coords(y=_y, x=_x)

    # Set origin, description, history, date
    dataset.origin = "omnic"
    dataset.description = kwargs.get('description',
                                     "Dataset from {0} spa files : {1}".format(
                                         nspec, ' ... '.join({filenames[0], filenames[-1]}))
                                     )

    dataset.history = str(datetime.now()) + ':read from spa files ; '

    if kwargs.get('sortbydate', True) and nspec > 1:
        dataset.sort(dim=0, inplace=True)
        dataset.history = 'Sorted'

    dataset._date = datetime.now()
    dataset._modified = dataset.date

    return dataset


def _read_srs(dataset, filenames, **kwargs):
    return dataset


if __name__ == '__main__':
    pass
