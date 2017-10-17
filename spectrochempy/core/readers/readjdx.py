# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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

"""Plugin module to extend NDDataset with the import methods method.

"""
import os as os
import numpy as np
from datetime import datetime, timezone, timedelta

from traitlets import HasTraits, Unicode, List

__all__ = ['read_jdx']

def read_jdx(filename='', sortbydate=True):
    """Open a .jdx file and return the correspondant dataset

    :param filename: filename of file to load
    :type filename: str
    :return: a  dataset object with spectra and metadata
    :rtype: sappy.Dataset

    Examples
    --------

    >>> import spectrochempy as sa
    >>> A = sa.loadjdx('C:\Spectra\Ex_spectra.jdx')
    >>> A.print()
       name: Ex_spectra.jdx
     author: Username
       date: Wed., 26-Nov-14, 08:30:35
       data: 10x1350  [float64]
       (...)

    .. seealso:: loadspg(), spectrochempy.Dataset()

    """

    # open file dialog box ****************************************************
    if filename == '':
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry('0x0+0+0')
        root.deiconify()
        root.lift()
        root.focus_force()
        filename = filedialog.askopenfilename(parent=root, \
                                              filetypes=[('jdx files', '.jdx'), ('all files', '.*')],
                                              title='Open .jdx file')
        root.destroy()

    # Open the file ***********************************************************
    f = open(filename, 'r')

    def readl(f):
        line = f.readline()
        if not line:
            return 'EOF', ''
        line = line.strip(' \n')  # remove newline character
        if line[0:2] == '##':  # if line starts with "##"
            if line[-1] != '=':  # line does not end by "="
                keyword = line.split('=')[0]
                text = line.split('=')[1]
            else:
                keyword = line.split('=')[0]
                text = ''
        else:
            keyword = ''
            text = line
        return keyword, text

        # Read header of outer Block **********************************************

    keyword = ''

    while keyword != '##TITLE':
        keyword, text = readl(f)
    if keyword != 'EOF':
        jdx_title = text
    else:
        print('Error: no ##TITLE LR in outer block header')
        return
    # Unuse for the moment...
    #    while keyword !='##JCAMP-DX':
    #        keyword, text = readl(f)
    #    if keyword != 'EOF':
    #        jdx_jcamp_dx = text
    #    else:
    #        print('Error: no ##JCAMP-DX LR in outer block header')
    #        return

    while ((keyword != '##DATA TYPE') and (keyword != '##DATATYPE')):
        keyword, text = readl(f)
    if keyword != 'EOF':
        jdx_data_type = text
    else:
        print('Error: no ##DATA TYPE LR in outer block header')
        return

    if jdx_data_type == 'LINK':
        while keyword != '##BLOCKS':
            keyword, text = readl(f)
        nspec = int(text)
    elif jdx_data_type == 'INFRARED SPECTRUM':
        nspec = 1
    else:
        print('Error: DATA TYPE must be LINK or INFRARED SPECTRUM')
        return

    # Create variables ********************************************************
    xaxis, data = [], []
    alltitles, allacquisitiondates, xunits, yunits = [], [], [], []
    nx, firstx, lastx = np.zeros(nspec, 'int'), np.zeros(nspec, 'float'), np.zeros(nspec, 'float')

    # Read the spectra ********************************************************
    for i in range(nspec):

        # Reset variables
        keyword = ''
        [year, month, day, hour, minute, second] = '', '', '', '', '', ''
        # (year, month,...) must be reseted at each spectrum because labels "time" and "longdate" are not required and JDX file

        # Read JDX file for spectrum n° i
        while keyword != '##END':
            keyword, text = readl(f)
            if keyword == '##TITLE':
                alltitles.append(text)  # Add the title of the spectrum in the liste alltitles
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
                    keyword, text = readl(f)
                    intensities = text.split(' ')[
                                  1:]  # for each line, get all the values exept the first one (first value = wavenumber)
                    allintensities = allintensities + intensities
                spectra = np.array([allintensities])  # convert allintensities into an array
                spectra[spectra == '?'] = 'nan'  # deals with missing or out of range intensity values
                spectra = spectra.astype(float)
                spectra = spectra * yfactor
                # add spectra in "data" matrix
                if not data:
                    data = spectra
                else:
                    data = np.concatenate((data, spectra), 0)

        # Check "firstx", "lastx" and "nx"
        if (firstx[i] != 0 and lastx[i] != 0 and nx[i] != 0):
            # Creation of xaxis if it doesn't exist yet
            if not xaxis:
                xaxis = np.linspace(firstx[0], lastx[0], nx[0])
                xaxis = np.around((xaxis * xfactor), 3)
            else:  # Check the consistency of xaxis
                if nx[i] - nx[i - 1] != 0:
                    print('Error: Inconsistant data set - number of wavenumber per spectrum should be identical')
                    return
                elif firstx[i] - firstx[i - 1] != 0:
                    print('Error: Inconsistant data set - the x axis should start at same value')
                    return
                elif lastx[i] - lastx[i - 1] != 0:
                    print('Error: Inconsistant data set - the x axis should end at same value')
                    return
        else:
            print('Error: ##FIRST, ##LASTX or ##NPOINTS are unusuable in the spectrum n°', i + 1)
            return

            # Creation of the acquisition date
        if (year != '' and month != '' and day != '' and hour != '' and minute != '' and second != ''):
            acqdate = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        else:
            acqdate = ''
        allacquisitiondates.append(acqdate)

        # Check the consistency of xunits and yunits
        if i > 0:
            if yunits[i] != yunits[i - 1]:
                print('Error: ##YUNITS sould be the same for all spectra (check spectrum n°', i + 1, ')')
                return
            elif xunits[i] != xunits[i - 1]:
                print('Error: ##XUNITS sould be the same for all spectra (check spectrum n°', i + 1, ')')
                return

    # Determine xaxis name ****************************************************
    if xunits[0] == '1/CM':
        axisname = 'Wavenumber (cm-1)'
    elif xunits[0] == 'MICROMETERS':
        axisname = 'Wavelength (µm)'
    elif xunits[0] == 'NANOMETERS':
        axisname = 'Wavelength (nm)'
    elif xunits[0] == 'SECONDS':
        axisname = 'Time (s)'  # <--------- Je ne sais pas quelle grandeur physique ici

    f.close()

    out = Dataset(data)
    out.name = jdx_title
    out.author = (os.environ['USERNAME'] + '@' + os.environ['COMPUTERNAME'])  # dataset author string
    out.date = datetime.datetime.now()
    out.moddate = out.date
    out.datalabel = yunits[0]
    out.appendlabels(Labels(alltitles, 'Title'))
    out.appendlabels(Labels(allacquisitiondates, 'Acquisition date (GMT)'))
    out.appendaxis(Axis(xaxis, 'Wavenumbers (cm-1)'), dim=1)
    if sortbydate:
        out.addtimeaxis()
        out.sort(0, 0)
        out.dims[0].deleteaxis(0)
    out.description = ('dataset "' + out.name + '" : imported from jdx file. \n')
    out.history = (str(out.date) + " : Created by jdxload('" + filename + "') \n")

    # make sure that the lowest( index correspond to th largest wavenember*
    # for compatibility with dataset creacted by spgload:

    if out.dims[1].axisset.axes[0][0] < out.dims[1].axisset.axes[0][-1]:
        out = out[:, ::-1]

    return out

