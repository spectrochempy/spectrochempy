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
