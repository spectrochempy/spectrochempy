# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================




"""Plugin module to extend NDDataset with the import methods method.

"""
import os as os
import numpy as np
from datetime import datetime, timezone, timedelta

from traitlets import HasTraits, Unicode, List

from spectrochempy.utils import SpectroChemPyWarning

__all__ = ['read_dso']


def read_dso(filename='', **kwargs):
    """Open an eigevector DSO object (.mat file) and return the corresponding dataset

    Parameters
    ===========
    filename : str
        filename of file to load

    Returns
    ========
    dataset : : a  dataset object with spectra and metadata


    Examples
    =========
    >>> import spectrochempy as sa
    >>> A = sa.loaddso('C:\Spectra\Ex_spectra.mat')
    >>> A.print()
       name: Ex_spectra.mat
     author: Username
       date: Wed., 26-Nov-14, 08:30:35
       data: 10x1350  [float64]
       (...)

    See Also
    =========



    """
    # open file dialog box
    if filename == '':
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry('0x0+0+0')
        root.deiconify()
        root.lift()
        root.focus_force()
        filename = filedialog.askopenfilename(parent=root,
                                              filetypes=[('mat files', '.mat'),
                                                         ('all files', '.*')],
                                              title='Open .mat file')
        root.destroy()

    content = sio.whosmat(filename)

    if len(content) > 1:
        raise ValueError('too many elements in the .mat file')

    if content[0][2] != 'object':
        raise TypeError('not a DSO object')

    f = sio.loadmat(filename)
    dso = content[0][0]

    name_mat = f[dso]['name'][0][0]
    if len(name_mat) == 0:
        name = '*unknown*'
    else:
        name = name_mat[0]

    typedata_mat = f[dso]['type'][0][0]
    if len(name_mat) == 0:
        typedata = ''
    else:
        typedata = typedata_mat[0]
    if typedata != 'data':
        raise TypeError('only \'data\' DSO object can be imported')

    author_mat = f[dso]['author'][0][0]
    if len(author_mat) == 0:
        author = '*unknown*'
    else:
        author = author_mat[0]

    date_mat = f[dso]['date'][0][0]
    if len(date_mat) == 0:
        date = datetime.datetime(1, 1, 1, 0, 0)
    else:
        date = datetime.datetime(int(date_mat[0][0]), int(date_mat[0][1]),
                                 int(date_mat[0][2]), int(date_mat[0][3]),
                                 int(date_mat[0][4]),
                                 int(date_mat[0][5]))

    data = f[dso]['data'][0][0]

    out = Dataset(data)
    out.author = author
    out.name = name
    out.date = date

    # add labels
    for i in range(out.ndim):
        for nj, j in enumerate(f[dso]['label'][0][0][i][0]):
            if len(j):
                if len(f[dso]['label'][0][0][i][1][nj]):
                    out.dims[i].appendlabels(Labels(j,
                                                    f[dso]['label'][0][0][i][1][
                                                        nj][0]))  # or [nj][0]
                else:
                    out.dims[i].appendlabels(Labels(j, '*unammed*'))

    # add axes
    for i in range(out.ndim):
        for nj, j in enumerate(f[dso]['axisscale'][0][0][i][0]):
            if len(j[0]):
                if len(f[dso]['axisscale'][0][0][i][1]):
                    out.dims[i].appendaxis(
                        Coords(j[0], f[dso]['axisscale'][0][0][i][1][nj][
                            0]))  # sometimes: Coord(j, f[dso]... , i.e. no j[0]    and  [nj] i.e. no [nj][0][0]
                else:
                    out.dims[i].appendaxis(Coords(j[0], '*unammed*'))

    for i in f[dso]['description'][0][0]:
        out.description += i + ' \n'

    for i in f[dso]['history'][0][0][0]:
        out.history += i[0] + ' \n'

    out.history += (
    str(datetime.datetime.now()) + ': imported by spectrochempy.loaddso() \n')

    return out
