# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================




__all__ = ['readfilename', 'readXlCellRange']

import xlrd
import os
import zipfile
import numpy as np

def readXlCellRange(xlFileName, cellRange, sheetNumber=0):
    """ reads data in a cellrange: A23:AD23 ''"""

    def colNameToColNumber(L):
        """converts the column alphabetical character designator
        to number, e.g. A -> 0; AD -> 29

        L: str, alphabetical character designator"""

        number = 0
        for i, l in enumerate(L[::-1]):
            number = number + (26 ** i) * (ord(l.lower()) - 96)
        number = number - 1
        return number

    start, stop = cellRange.split(':')

    colstart = colNameToColNumber(''.join([l for l in start if l.isalpha()]))
    colstop = colNameToColNumber(''.join([l for l in stop if l.isalpha()]))
    linstart = int(''.join([l for l in start if not l.isalpha()])) - 1
    linstop = int(''.join([l for l in stop if not l.isalpha()])) - 1

    xlfile = xlrd.open_workbook(xlFileName)
    sheetnames = xlfile.sheet_names()
    sheet = xlfile.sheet_by_name(sheetnames[0])

    out = []
    if colstart == colstop:
        for line in np.arange(linstart, linstop + 1):
            out.append(sheet.cell_value(line, colstart))
    elif linstart == linstop:
        for column in np.arange(colstart, colstop + 1):
            out.append(sheet.cell_value(linstart, column))
    else:
        raise ValueError('Cell range must be within a single column or line...')

    return out

# =============================================================================
# Utility function
# =============================================================================


def readfilename(filename, **kwargs):
    """
    returns a list of the filenames of existing files, filtered by extensions
    :param filename: Filename of file(s). If `None`: opens a dialog box to select
    files. If `str`: a single filename. If list of str: a list of filenames.
    :param directory [optional, default=""]: the directory where to look at. If not specified, read in
       current directory
    :param filetypes [optional, default=['all files, '.*)']]
    :return: a list of the filenames
    """

    directory = kwargs.get("directory", "")

    filetypes = kwargs.get("filetypes", [('all files', '.*')])
    if not os.path.exists(directory):
        raise IOError("directory doesn't exists!")

    if isinstance(filename, str) and os.path.isdir(filename):
        raise IOError('a directory has been provided instead of a filename!')

    if not filename:
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry('0x0+0+0')
        root.deiconify()
        root.lift()
        root.focus_force()
        filenamestring = filedialog.askopenfilenames(parent=root, \
                                                     filetypes=filetypes,
                                                     title='Open omnic file')

        root.quit()
        filename = [_filename for _filename in filenamestring]

    if isinstance(filename, list):
        if not all(isinstance(elem, str) for elem in filename):
            raise IOError('one of the list elements is not a filename!')
        else:
            filenames = [os.path.join(directory, elem) for elem in filename]

    if isinstance(filename, str):
        filenames = [filename]

    # filenames passed
    files = {}
    for filename in filenames:
        _, extension = os.path.splitext(filename)
        extension = extension.lower()
        if extension in files.keys():
            files[extension].append(filename)
        else:
            files[extension] = [filename]
    return files


if __name__ == '__main__':

    pass