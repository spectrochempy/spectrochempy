# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


# Author(s): Arnaud Travert (LCS)
# Contributor(s): Christian Fernandez (LCS)

"""Plugin module to extend NDDataset with the import methods method.

"""
import os as os
import numpy as np
from datetime import datetime, timezone, timedelta

from traitlets import HasTraits, Unicode, List

__all__ = []


def write_jdx(X, filename=''):
    """ Exports dataset to jcampdx format"""

    # if no filename is provided, open a dialog box to create jdx file
    if filename == '':
        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry('0x0+0+0')
        root.deiconify()
        root.lift()
        root.focus_force()
        f = filedialog.asksaveasfile(mode='w', initialfile='dataset',
                                     defaultextension=".jdx",
                                     filetypes=(("JCAMPDX", "*.jdx"),
                                                ("All Files", "*.*")))
        if f is None:  # asksaveasile return `None` if dialog closed with "cancel".
            return
        root.destroy()
    else:
        f = open(filename,
                 'w')  # if filename is provided,directly create jdx file

    # writes first lines
    f.write('##TITLE=' + X.name + '\n')
    f.write('##JCAMP-DX=5.01' + '\n')
    # if several spectra => Data Type = LINK
    if X.shape[0] > 1:
        f.write('##DATA TYPE=LINK' + '\n')
        f.write('##BLOCKS=' + str(X.shape[
                                      0]) + '\n')  # number of spectra (size of 1st dimension )
    else:
        f.write('##DATA TYPE=INFRARED SPECTRUM' + '\n')

    for i in range(X.shape[0]):
        if X.shape[0] > 1:
            if len(X.dims[0].labels):
                f.write('##TITLE=' + X.dims[0].labels[0][i] + '\n')
            else:
                f.write('##TITLE= spectrum #' + str(i) + '\n')
            f.write('##JCAMP-DX=5.01' + '\n')
        f.write('##ORIGIN=' + X.author + '\n')
        f.write('##OWNER=LCS' + '\n')
        if len(X.dims[0].labels):
            f.write('##LONGDATE=' + X.dims[0].labels[1][i].strftime(
                "%Y/%m/%d") + '\n')
            f.write(
                '##TIME=' + X.dims[0].labels[1][i].strftime("%H:%M:%S") + '\n')
        f.write('##XUNITS=1/CM' + '\n')
        f.write('##YUNITS=' + 'ABSORBANCE' + '\n')
        nx = X.shape[1]
        firstx, lastx = X.dims[1].axes[0][0], X.dims[1].axes[0][nx - 1]
        firsty, lasty = X.data[0, 0], X.data[0, nx - 1]
        if firstx > lastx:
            maxx, minx = firstx, lastx
        else:
            maxx, minx = lastx, firstx

        f.write('##FIRSTX=' + str(firstx) + '\n')
        f.write('##LASTX=' + str(lastx) + '\n')
        f.write('##FIRSTY=' + str(firsty) + '\n')
        f.write('##LASTY=' + str(lasty) + '\n')
        if firstx > lastx:
            maxx, minx = firstx, lastx
        else:
            maxx, minx = lastx, firstx
        nx = X.shape[1]
        f.write('##MAXX=' + str(maxx) + '\n')
        f.write('##MINX=' + str(minx) + '\n')
        maxy, miny = np.nanmax(X.data), np.nanmin(X.data)
        f.write('##MAXY=' + str(maxy) + '\n')
        f.write('##MINY=' + str(miny) + '\n')
        f.write('##XFACTOR=1.000000' + '\n')
        f.write('##YFACTOR=1.000000E-08' + '\n')
        f.write('##NPOINTS=' + str(nx) + '\n')
        yfactor = 1e-8
        f.write('##XYDATA=(X++(Y..Y))' + '\n')
        x = str(X.dims[1].axes[0][0])  # first x
        f.write(x + ' ')  # Write the first x
        y = str(int(X.data[i, 0] / yfactor))  # first y
        f.write(y + ' ')  # write first y
        llen = len(x) + len(y) + 2  # length of current line
        for j in np.arange(1, nx):
            if np.isnan(X.data[i, j]):
                y = '?'
            else:
                y = str(int(X.data[i, j] / yfactor))
            f.write(y + ' ')
            llen = llen + len(y) + 1
            if llen > 75:
                x = str(X.dims[1].axes[0][j])
                f.write('\n' + x + ' ')
                llen = len(x) + 1
        f.write('\n' + '##END' + '\n')

    f.write('##END=' + '\n')
    f.close()
