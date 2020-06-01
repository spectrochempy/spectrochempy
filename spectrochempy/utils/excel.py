# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

# TODO: 0.1.21 - Do we need this?

__all__ = ['readXlCellRange']

import xlrd
import numpy as np


def readXlCellRange(xlFileName, cellRange, sheetNumber=0):
    """ reads data in a cellrange : A23:AD23 ''"""

    def colNameToColNumber(L):
        """converts the column alphabetical character designator
        to number, e.g. A -> 0; AD -> 29

        L : str, alphabetical character designator"""

        number = 0
        for i, l in enumerate(L[::-1]):
            number = number + (26 ** i) * (ord(l.lower()) - 96)
        number = number - 1
        return number

    start, stop = cellRange.split(':')

    colstart = colNameToColNumber(''.join([item for item in start if item.isalpha()]))
    colstop = colNameToColNumber(''.join([item for item in stop if item.isalpha()]))
    linstart = int(''.join([item for item in start if not item.isalpha()])) - 1
    linstop = int(''.join([item for item in stop if not item.isalpha()])) - 1

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


if __name__ == '__main__':
    pass
