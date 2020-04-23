# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


# Author(s): Arnaud Travert (LCS)
# Contributor(s): Christian Fernandez (LCS)

"""Plugin module to extend NDDataset with export methods method.

"""
import os as os
import numpy as np
from datetime import datetime, timezone, timedelta

from traitlets import HasTraits, Unicode, List

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoord import Coord
from ...core import info_, debug_, error_, warning_

__all__ = ['write_jdx']


def write_jdx(dataset=None, **kwargs):
    """Writes a the dataset X in jdx format

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset to store the data and metadata read from the OMNIC file(s).
        If None, a |NDDataset| is created.
    filename : `None`, `str`
        Filename of the file to write. If `None`: opens a dialog box to safe files.
    directory: str, optional, default="".
        Where to save the file. If not specified, write in
        the current directory.

    Returns
    -------
    None

    Examples
    --------
    >>> A = NDDataset.write_jdx('myfile.jdx')


    """
    debug_("writing jdx file")

    # filename will be given by a keyword parameter except if the first parameters is already the filename
    filename = kwargs.get('filename', None)

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(dataset, NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(dataset, (str, list)) and dataset != '':
            filename = dataset

    # check if directory was specified
    directory = kwargs.get("directory", None)
    if not directory:
        directory = os.getcwd()
    # check if a valid filename is given



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
