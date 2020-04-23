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

import os as os
import numpy as np
from datetime import datetime, timezone, timedelta

# ----------------------------------------------------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.utils import readfilename
from spectrochempy.core import info_, debug_, error_, warning_

# ............................................................................
def read_jdx(dataset=None, **kwargs):
    """Open a JCAMP-DX file with extension ``.jdx``

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to store the data and metadata read from a JCAMP-DX file.
        If None, a |NDDataset| is created.
    filename : `None`, `str`, or list of `str`
        Filename of the file(s) to load. If `None` : opens a dialog box to select
        filename.
    directory : str, optional, default="".
        From where to read the specified filename. If not specified, read in
        the defaults datadir.
    sortbydate : bool, optional, default=True.
        Sort spectra by acquisition date

    Returns
    -------
    dataset : |NDDataset|
        A dataset corresponding to the ``.jdx``file.

    Example
    --------
    #todo: add example
    """

    #debug_("reading jdx file")
    sortbydate = kwargs.get('sortbydate', True)
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

    # returns a list of files to read
    files = readfilename(filename,
                             directory=directory,
                             filetypes=['JCAMP-DX files (*.jdx)',
                                        'All files (*)'])

    if not files:
        # there is no files, return nothing
        return None


    datasets = []
    for filename in files['.jdx']:
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
    # Unuse for the moment...
    #    while keyword !='##JCAMP-DX':
    #        keyword, text = readl(f)
    #    if keyword != 'EOF':
    #        jdx_jcamp_dx = text
    #    else:
    #        print('Error: no ##JCAMP-DX LR in outer block header')
    #        return

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
        elif jdx_data_type == 'INFRARED SPECTRUM':
            nspec = 1
        else:
            print('Error : DATA TYPE must be LINK or INFRARED SPECTRUM')
            return

        # Create variables ********************************************************
        xaxis  = np.array([])
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
                    alltitles.append(
                        text)  # Add the title of the spectrum in the liste alltitles
                if keyword == '##LONGDATE':
                    [year, month, day] = text.split('/')
                if keyword == '##TIME':
                    [hour, minute, second] = text.split(':')
                if keyword == '##XUNITS':
                    xunits.append(text)
                if keyword == '##YUNITS':
                    yunits.append(text)
                if keyword == '##FIRSTX':
                    firstx[i] = float(text)
                if keyword == '##LASTX':
                    lastx[i] = float(text)
                # Unuse for the moment...
                #                if keyword =='##FIRSTY':
                #                firsty = float(text)

                if keyword == '##XFACTOR':
                    xfactor = float(text)
                if keyword == '##YFACTOR':
                    yfactor = float(text)
                if keyword == '##NPOINTS':
                    nx[i] = float(text)
                if keyword == '##XYDATA':
                    # Read all the intensities
                    allintensities = []
                    while keyword != '##END':
                        keyword, text = _readl(f)
                        intensities = text.split(' ')[
                                      1:]  # for each line, get all the values exept the first one (first value = wavenumber)
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
                            'Error : Inconsistant data set - number of wavenumber per spectrum should be identical')
                        return
                    elif firstx[i] - firstx[i - 1] != 0:
                        raise ValueError(
                            'Error : Inconsistant data set - the x axis should start at same value')
                        return
                    elif lastx[i] - lastx[i - 1] != 0:
                        raise ValueError(
                            'Error : Inconsistant data set - the x axis should end at same value')
                        return
            else:
                raise ValueError(
                    'Error : ##FIRST, ##LASTX or ##NPOINTS are unusuable in the spectrum n°',
                    i + 1)
                return

                # Creation of the acquisition date
            if (year != '' and month != '' and day != '' and hour != '' and minute != '' and second != ''):
                date = datetime(int(year), int(month), int(day),
                                            int(hour), int(minute), int(second))
                timestamp = date.timestamp()
                # Transform back to timestamp for storage in the Coord object
                # use datetime.fromtimestamp(d, timezone.utc))
                # to transform back to datetime object
            else:
                timestamp = None
                #Todo: cases where incomplete date and/or time info
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
        if xunits[0] == '1/CM':
            axisname = 'Wavenumbers'
            axisunit = 'cm^-1'
        elif xunits[0] == 'MICROMETERS':
            axisname = 'Wavelength'
            axisunit = 'um'
        elif xunits[0] == 'NANOMETERS':
            axisname = 'Wavelength'
            axisunit = 'nm'
        elif xunits[0] == 'SECONDS':
            axisname = 'Time'
            axisunit = 's'
        elif xunits[0] == 'ARBITRARY UNITS':
            axisname = 'Arbitrary unit'
            axisunit = '-'
        f.close()

        dataset = NDDataset(data)
        dataset.name = jdx_title
        dataset.author = (os.environ['USERNAME'] + '@' + os.environ[
            'COMPUTERNAME'])  # dataset author string

        dataset.units = 'absorbance'
        dataset.title = 'Absorbance'
        dataset.name = ' ... '.join(set([alltitles[0], alltitles[-1]]))
        dataset._date = datetime.now()
        dataset._modified = dataset._date

        # now add coordinates
        _x = Coord(xaxis, title=axisname, units=axisunit)
        _y = Coord(alltimestamps, title='Timestamp', units='s',
                   labels=(alldates, alltitles))
        dataset.set_coords(y=_y, x=_x)

        # Set origin, description and history
        dataset.origin = "JCAMP-DX"
        dataset.description = "Dataset from jdx: '{0}'"\
            .format( ' ... '.join(set([alltitles[0], alltitles[0]])))

        dataset.history = str(datetime.now()) + ':imported from jdx files \n'

        if sortbydate:
            dataset.sort(dim=0, inplace=True)
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
            text=''
        else: # keyword + text
            keyword = line.split('=')[0]
            text = line.split('=')[1]
    else:
        keyword = ''
        text = line
    return keyword, text

