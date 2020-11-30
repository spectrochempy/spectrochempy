# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

"""
This module to extend NDDataset with the import methods.

"""

__all__ = ['read_jcamp', 'read_jdx']
__dataset_methods__ = __all__

import io
import re
from datetime import datetime, timezone
import numpy as np

from spectrochempy.core import debug_
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.core.readers.importer import docstrings, Importer, importermethod

# ======================================================================================================================
# Public functions
# ======================================================================================================================

# ............................................................................
@docstrings.dedent
def read_jcamp(*args, **kwargs):
    """
    Open Infrared JCAMP-DX files with extension ``.jdx`` or ``.dx``.
    Limited to AFFN encoding (see R. S. McDonald and Paul A. Wilks,
    JCAMP-DX: A Standard Form for Exchange of Infrared Spectra in Computer Readable Form,
    Appl. Spec., 1988, 1, 151–162. doi:10.1366/0003702884428734.)

    Parameters
    ----------
    %(read_method.parameters.no_origin|csv_delimiter)s

    Other Parameters
    ----------------
    %(read_method.other_parameters)s

    Returns
    -------
    out : NDDataset| or list of |NDDataset|
        The dataset or a list of dataset corresponding to a (set of) .jdx file(s).

    See Also
    ---------
    read : Generic read method
    read_csv, read_zip, read_matlab, read_omnic, read_opus, read_topspin

    """
    kwargs['filetypes'] = ['JCAMP-DX files (*.jdx *.dx)']
    kwargs['protocol'] = ['.jcamp', '.jdx', '.dx']
    importer = Importer()
    return importer(*args, **kwargs)

read_jdx = read_jcamp
read_jdx.__doc__ = 'This method is an alias of `read_jcamp` '
read_dx = read_jcamp
read_dx.__doc__ = 'This method is an alias of `read_jcamp` '

# ======================================================================================================================
# private functions
# ======================================================================================================================

@importermethod
def _read_jdx(*args, **kwargs):

    debug_("reading a json file")

    # read jdx file
    dataset , filename = args
    content = kwargs.get('content', None)
    sortbydate = kwargs.pop("sortbydate", True)

    if content is not None:
        fid = io.StringIO(content.decode("utf-8"))
    else:
        fid = open(filename, 'r')

    # Read header of outer Block
    # ..................................................................................................................
    keyword = ''

    while keyword != '##TITLE':
        keyword, text = _readl(fid)
    if keyword != 'EOF':
        jdx_title = text
    else:
        raise ValueError('No ##TITLE LR in outer block header')

    while (keyword != '##DATA TYPE') and (keyword != '##DATATYPE'):
        keyword, text = _readl(fid)
    if keyword != 'EOF':
        jdx_data_type = text
    else:
        raise ValueError('No ##DATA TYPE LR in outer block header')

    if jdx_data_type == 'LINK':
        while keyword != '##BLOCKS':
            keyword, text = _readl(fid)
        nspec = int(text)
    elif jdx_data_type.replace(' ', '') == 'INFRAREDSPECTRUM':
        nspec = 1
    else:
        raise ValueError('DATA TYPE must be LINK or INFRARED SPECTRUM')


    # Create variables
    # ..................................................................................................................
    xaxis = np.array([])
    data = np.array([])
    alltitles, alltimestamps, alldates, xunits, yunits = [], [], [], [], []
    nx, firstx, lastx = np.zeros(nspec, 'int'), np.zeros(nspec, 'float'), np.zeros(nspec, 'float')

    # Read the spectra
    # ..................................................................................................................
    for i in range(nspec):

        # Reset variables
        keyword = ''

        # (year, month,...) must be reset at each spectrum because labels "time"
        # and "longdate" are not required in JDX file
        [year, month, day, hour, minute, second] = '', '', '', '', '', ''

        # Read JDX file for spectrum n° i
        while keyword != '##END':
            keyword, text = _readl(fid)
            if keyword in ['##ORIGIN', '##OWNER', '##JCAMP-DX']:
                continue
            elif keyword == '##TITLE':
                # Add the title of the spectrum in the list alltitles
                alltitles.append(text)
            elif keyword == '##LONGDATE':
                [year, month, day] = text.split('/')
            elif keyword == '##TIME':
                [hour, minute, second] = re.split(r':|\.', text)
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
                # Read the intensities
                allintensities = []
                while keyword != '##END':
                    keyword, text = _readl(fid)
                    # for each line, get all the values exept the first one (first value = wavenumber)
                    intensities = list(filter(None, text.split(' ')[1:]))
                    if len(intensities) > 0:
                        allintensities += intensities
                spectra = np.array([allintensities])  # convert allintensities into an array
                spectra[spectra == '?'] = 'nan'  # deals with missing or out of range intensity values
                spectra = spectra.astype(np.float32)
                spectra *= yfactor
                # add spectra in "data" matrix
                if not data.size:
                    data = spectra
                else:
                    data = np.concatenate((data, spectra), 0)

        # Check "firstx", "lastx" and "nx"
        if firstx[i] != 0 and lastx[i] != 0 and nx[i] != 0:
            if not xaxis.size:
                # Creation of xaxis if it doesn't exist yet
                xaxis = np.linspace(firstx[0], lastx[0], nx[0])
                xaxis = np.around((xaxis * xfactor), 3)
            else:
                # Check the consistency of xaxis
                if nx[i] - nx[i - 1] != 0:
                    raise ValueError('Inconsistent data set: number of wavenumber per spectrum should be identical')
                elif firstx[i] - firstx[i - 1] != 0:
                    raise ValueError('Inconsistent data set: the x axis should start at same value')
                elif lastx[i] - lastx[i - 1] != 0:
                    raise ValueError('Inconsistent data set: the x axis should end at same value')
        else:
            raise ValueError('##FIRST, ##LASTX or ##NPOINTS are unusuable in the spectrum n°', i + 1)

        # Creation of the acquisition date
        if (year != '' and month != '' and day != '' and hour != '' and minute != '' and second != ''):
            date = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
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
                raise ValueError(f'##YUNITS should be the same for all spectra (check spectrum n°{i + 1}')
            elif xunits[i] != xunits[i - 1]:
                raise ValueError(f'##XUNITS should be the same for all spectra (check spectrum n°{i + 1}')

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
        axisunit = None
    else:
        axisname = ''
        axisunit = ''
    fid.close()

    dataset.data = data
    dataset.name = jdx_title
    if yunits[0].strip() == 'ABSORBANCE':
        dataset.units = 'absorbance'
        dataset.title = 'Absorbance'
    elif yunits[0].strip() == 'TRANSMITTANCE':
        # TODO: This units not in pint. Add this
        dataset.title = 'Transmittance'

    # now add coordinates
    _x = Coord(xaxis, title=axisname, units=axisunit)
    if jdx_data_type == 'LINK':
        _y = Coord(alltimestamps, title='Acquisition timestamp (GMT)', units='s', labels=(alldates, alltitles))
        dataset.set_coords(y=_y, x=_x)
    else:
        _y = Coord()
    dataset.set_coords(y=_y, x=_x)

    # Set origin, description and history
    dataset.origin = "omnic"
    dataset.description = "Dataset from jdx: '{0}'".format(jdx_title)

    dataset.history = str(datetime.now()) + ':imported from jdx file \n'

    if sortbydate:
        dataset.sort(dim='x', inplace=True)
        dataset.history = str(datetime.now()) + ':sorted by date\n'
    # Todo: make sure that the lowest index correspond to the largest wavenumber
    #  for compatibility with dataset created by read_omnic:

    # Set the NDDataset date
    dataset._date = datetime.now()
    dataset._modified = dataset.date

    return dataset

# ......................................................................................................................
@importermethod
def _read_dx(*args, **kwargs):
    return _read_jdx(*args, **kwargs)


# ......................................................................................................................
def _readl(fid):
    line = fid.readline()
    if not line:
        return 'EOF', ''
    line = line.strip(' \n')  # remove newline character
    if line[0:2] == '##':  # if line starts with "##"
        if line[0:5] == '##END':  # END KEYWORD, no text
            keyword = '##END'
            text = ''
        else:  # keyword + text
            keyword, text = line.split('=')
    else:
        keyword = ''
        text = line.strip()
    return keyword, text

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
