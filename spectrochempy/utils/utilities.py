# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================




__all__ = ['unzip', 'readfilename', 'readXlCellRange']

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

def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        for member in zf.infolist():
            # Path traversal defense copied from
            # http://hg.python.org/cpython/file/tip/Lib/http/server.py#l789
            words = member.filename.split('/')
            path = dest_dir
            for word in words[:-1]:
                drive, word = os.path.splitdrive(word)
                head, word = os.path.split(word)
                if word in (os.curdir, os.pardir, ''): continue
                path = os.path.join(path, word)
            zf.extract(member, path)

def readfilename(filename, directory='', filter=''):

    if os.path.isdir(filename):
        directory = filename
        filename = None

    if not filename:
        raise IOError('no filename provided!')

    else:
        filenames = [filename]

    # filenames passed
    files = {}
    for filename in filenames:
        _, extension = os.path.splitext(filename)
        extension = extension.lower()
        if extension in files.keys():
            files[extension].append(filename)
        else:
            files[extension]=[filename]

    return files


if __name__ == '__main__':

    res = readfilename(None, filter='OMNIC file (*.spg);;'
                                         'OMNIC file (*.spa)')
    print(res)