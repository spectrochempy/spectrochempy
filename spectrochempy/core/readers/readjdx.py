# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module to extend NDDataset with the import methods method.

"""

__all__ = ['read_jdx']

__dataset_methods__ = __all__

# ----------------------------------------------------------------------------------------------------------------------
# standard imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import re
from datetime import datetime

# ----------------------------------------------------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.utils import readfilename


# ............................................................................
def read_jdx(dataset, filename=None, directory=None, sortbydate=True):
    """
    Open Infrared JCAMP-DX files with extension ``.jdx`` or ``.dx``.
    Limited to AFFN encoding (see R. S. McDonald and Paul A. Wilks,
    JCAMP-DX: A Standard Form for Exchange of Infrared Spectra in Computer Readable Form,
    Appl. Spec., 1988, 1, 151–162. doi:10.1366/0003702884428734.)
    
    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to store the data and metadata read from a JCAMP-DX file.
        If None, a |NDDataset| is created.
    filename : None, str, or list of str
        Filename of the file(s) to load. If None : opens a dialog box to select
        filename.
    directory : str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the defaults datadir.
    sortbydate : bool, optional, default=True.
        Sort spectra by acquisition date
    
    Returns
    -------
    dataset : |NDDataset|
        A dataset corresponding to the ``.jdx`` file.

    """

    # filename will be given by a keyword parameter except if the first parameters is already the filename

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, (str, list)) and dataset != '':
            filename = dataset

        dataset = NDDataset()  # create an instance of NDDataset

    # returns a list of filenames
    filenames = readfilename(filename,
                         directory=directory,
                         filetypes=['JCAMP-DX files (*.jdx, *.dx)',
                                    'All files (*)'], dictionary=False)

    if not filenames:
        return None

    for filename in filenames:
        f = open(filename, 'r')
        # Read header of outer Block **********************************************
        keyword = ''

        while keyword != '##TITLE':
            keyword, text = _readl(f)
        if keyword != 'EOF':
            jdx_title = text
        else:
            print('Error : no ##TITLE LR in outer block header')
            return

        while (keyword != '##DATA TYPE') and (keyword != '##DATATYPE'):
            keyword, text = _readl(f)
        if keyword != 'EOF':
            jdx_data_type = text
        else:
            print('Error : no ##DATA TYPE LR in outer block header')
            return

        if jdx_data_type == 'LINK':
            while keyword != '##BLOCKS':
                keyword, text = _readl(f)
            nspec = int(text)
        elif jdx_data_type.replace(' ', '') == 'INFRAREDSPECTRUM':
            nspec = 1
        else:
            print('Error : DATA TYPE must be LINK or INFRARED SPECTRUM')
            return

        # Create variables ********************************************************
        xaxis = np.array([])
        data = np.array([])
        alltitles, alltimestamps, alldates, xunits, yunits = [], [], [], [], []
        nx, firstx, lastx = np.zeros(nspec, 'int'), np.zeros(nspec,
                                                             'float'), np.zeros(
            nspec, 'float')

        # Read the spectra ********************************************************
        for i in range(nspec):

            # Reset variables
            keyword = ''
            [year, month, day, hour, minute, second] = '', '', '', '', '', ''
            # (year, month,...) must be reset at each spectrum because labels "time" and "longdate" are not required and JDX file

            # Read JDX file for spectrum n° i
            while keyword != '##END':
                keyword, text = _readl(f)
                if keyword == '##TITLE':
                    # Add the title of the spectrum in the list alltitles
                    alltitles.append(text)
                elif keyword == '##LONGDATE':
                    [year, month, day] = text.split('/')
                elif keyword == '##TIME':
                    [hour, minute, second] = re.split(':|\.', text)
                elif keyword == '##XUNITS':
                    xunits.append(text)
                elif keyword == '##YUNITS':
                    yunits.append(text)
                elif keyword == '##FIRSTX':
                    firstx[i] = float(text)
                elif keyword == '##LASTX':
                    lastx[i] = float(text)
                elif keyword == '##XFACTOR':
                    xfactor = float(text)
                elif keyword == '##YFACTOR':
                    yfactor = float(text)
                elif keyword == '##NPOINTS':
                    nx[i] = float(text)
                elif keyword == '##XYDATA':
                    # Read all the intensities
                    allintensities = []
                    while keyword != '##END':
                        keyword, text = _readl(f)
                        # for each line, get all the values exept the first one (first value = wavenumber)
                        intensities = list(filter(None, text.split(' ')[1:]))

                        allintensities = allintensities + intensities
                    spectra = np.array([allintensities])  # convert allintensities into an array
                    spectra[spectra == '?'] = 'nan'  # deals with missing or out of range intensity values
                    spectra = spectra.astype(float)
                    spectra = spectra * yfactor
                    # add spectra in "data" matrix
                    if not data.size:
                        data = spectra
                    else:
                        data = np.concatenate((data, spectra), 0)

            # Check "firstx", "lastx" and "nx"
            if firstx[i] != 0 and lastx[i] != 0 and nx[i] != 0:
                # Creation of xaxis if it doesn't exist yet
                if not xaxis.size:
                    xaxis = np.linspace(firstx[0], lastx[0], nx[0])
                    xaxis = np.around((xaxis * xfactor), 3)
                else:  # Check the consistency of xaxis
                    if nx[i] - nx[i - 1] != 0:
                        raise ValueError(
                            'Error : Inconsistent data set - number of wavenumber per spectrum should be identical')
                    elif firstx[i] - firstx[i - 1] != 0:
                        raise ValueError(
                            'Error : Inconsistent data set - the x axis should start at same value')
                    elif lastx[i] - lastx[i - 1] != 0:
                        raise ValueError(
                            'Error : Inconsistent data set - the x axis should end at same value')
            else:
                raise ValueError(
                    'Error : ##FIRST, ##LASTX or ##NPOINTS are unusuable in the spectrum n°',
                    i + 1)

                # Creation of the acquisition date
            if (year != '' and month != '' and day != '' and hour != '' and minute != '' and second != ''):
                date = datetime(int(year), int(month), int(day),
                                int(hour), int(minute), int(second))
                timestamp = date.timestamp()
                # Transform back to timestamp for storage in the Coord object
                # use datetime.fromtimestamp(d, timezone.utc))
                # to transform back to datetime object
            else:
                timestamp = date = None
                # Todo: cases where incomplete date and/or time info
            alltimestamps.append(timestamp)
            alldates.append(date)

            # Check the consistency of xunits and yunits
            if i > 0:
                if yunits[i] != yunits[i - 1]:
                    print((
                        'Error : ##YUNITS sould be the same for all spectra (check spectrum n°',
                        i + 1, ')'))
                    return
                elif xunits[i] != xunits[i - 1]:
                    print((
                        'Error : ##XUNITS sould be the same for all spectra (check spectrum n°',
                        i + 1, ')'))
                    return


        # Determine xaxis name ****************************************************
        if xunits[0].strip() == '1/CM':
            axisname = 'Wavenumbers'
            axisunit = 'cm^-1'
        elif xunits[0].strip() == 'MICROMETERS':
            axisname = 'Wavelength'
            axisunit = 'um'
        elif xunits[0].strip() == 'NANOMETERS':
            axisname = 'Wavelength'
            axisunit = 'nm'
        elif xunits[0].strip() == 'SECONDS':
            axisname = 'Time'
            axisunit = 's'
        elif xunits[0].strip() == 'ARBITRARY UNITS':
            axisname = 'Arbitrary unit'
            axisunit = '-'
        else:
            axisname = ''
            axisunit = ''
        f.close()

        dataset = NDDataset(data)
        dataset.name = jdx_title
        if yunits[0].strip()=='ABSORBANCE':
            dataset.units = 'absorbance'
            dataset.title = 'Absorbance'
        elif yunits[0].strip()=='TRANSMITTANCE':
            dataset.title = 'Transmittance'
        dataset.name = jdx_title
        dataset._date = dataset._modified =datetime.now()

        # now add coordinates
        _x = Coord(xaxis, title=axisname, units=axisunit)
        if jdx_data_type == 'LINK':
            _y = Coord(alltimestamps, title='Timestamp', units='s',
                   labels=(alldates, alltitles))
            dataset.set_coords(y=_y, x=_x)
        else:
            _y = Coord()
        dataset.set_coords(y=_y, x=_x)


        # Set origin, description and history
        dataset.origin = "JCAMP-DX"
        dataset.description = "Dataset from jdx: '{0}'".format(jdx_title)

        dataset.history = str(datetime.now()) + ':imported from jdx file \n'

        if sortbydate:
            dataset.sort(dim='x', inplace=True)
            dataset.history = str(datetime.now()) + ':sorted by date\n'
        # Todo: make sure that the lowest index correspond to the largest wavenumber
        #  for compatibility with dataset created by spgload:

        # Set the NDDataset date
        dataset._date = datetime.now()
        dataset._modified = dataset.date

        return dataset

# ======================================================================================================================
# private functions
# ======================================================================================================================

def _readl(f):
    line = f.readline()
    if not line:
        return 'EOF', ''
    line = line.strip(' \n')  # remove newline character
    if line[0:2] == '##':  # if line starts with "##"
        if line[0:5] == '##END':  # END KEYWORD, no text
            keyword = '##END'
            text = ''
        else:  # keyword + text
            keyword = line.split('=')[0]
            text = line.split('=')[1]
    else:
        keyword = ''
        text = line.strip()
    return keyword, text
