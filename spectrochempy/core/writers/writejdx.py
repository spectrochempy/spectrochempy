# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================


"""Plugin module to extend NDDataset with export methods method.

"""
import os as os
import numpy as np
from datetime import datetime, timezone, timedelta

from traitlets import HasTraits, Unicode, List

from spectrochempy.core.dataset.nddataset import NDDataset
from ...utils import savefilename
from ...core import info_, debug_, error_, warning_

__all__ = ['write_jdx']

__dataset_methods__ = __all__


def write_jdx(*args, **kwargs):
    """Writes a dataset in JCAMP-DX format

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset
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
    >>> X.write_jdx('myfile.jdx')
    """
    debug_("writing jdx file")

    # filename will be given by a keyword parameter except if the first parameters is already the filename
    filename = kwargs.get('filename', None)

    # check if the first parameter is a dataset because we allow not to pass it
    if not isinstance(args[0], NDDataset):
        # probably did not specify a dataset
        # so the first parameters must be the filename
        if isinstance(args[0], str) and args[0] != '':
            filename = args[0]
    else: #then the dataset is the first and the filename might be the second parameter:
        dataset = args[0]
        if isinstance(args[1], str) and args[0] != '':
            filename = args[1]

    directory = kwargs.get('directory', None)

    filename = savefilename(filename=filename,
                            directory=directory,
                            filters="JCAMP-DX (*.JDX) ;; All files (*)")
    if filename is None:
        # no filename from the dialogbox
        return

    f = open(filename, 'w')  # if filename is provided,directly create jdx file

    # writes first lines
    f.write('##TITLE=' + dataset.name + '\n')
    f.write('##JCAMP-DX=5.01' + '\n')
    # if several spectra => Data Type = LINK
    if dataset.shape[0] > 1:
        f.write('##DATA TYPE=LINK' + '\n')
        f.write('##BLOCKS=' + str(dataset.shape[
                                      0]) + '\n')  # number of spectra (size of 1st dimension )
    else:
        f.write('##DATA TYPE=INFRARED SPECTRUM' + '\n')

    # determine whether the spectra have a title and a datetime field in the labels,
    # by default, the title if any will be is the first string; the timestamp will
    # be the fist datetime.datetime
    title_index = None
    timestamp_index  = None
    if dataset.y.labels is not None:
        for i, label in enumerate(dataset.y.labels[0]):
            if not title_index and type(label) is str:
                title_index = i
            if not timestamp_index and type(label) is datetime:
                timestamp_index = i

    if timestamp_index is None:
        timestamp = datetime.now()

    for i in range(dataset.shape[0]):
        if dataset.shape[0] > 1:
            if title_index:
                f.write('##TITLE=' + dataset.y.labels[i][title_index] + '\n')
            else:
                f.write('##TITLE= spectrum #' + str(i) + '\n')
            f.write('##JCAMP-DX=5.01' + '\n')
        f.write('##ORIGIN=' + dataset.origin  + '\n')
        f.write('##OWNER=' + dataset.author + '\n')

        if timestamp_index is not None:
            timestamp = dataset.y.labels[i][timestamp_index]
        f.write('##LONGDATE=' +
                    timestamp.strftime("%Y/%m/%d") + '\n')
        f.write('##TIME=' +
                    timestamp.strftime("%H:%M:%S") + '\n')
        f.write('##XUNITS=1/CM' + '\n')
        f.write('##YUNITS=' + 'ABSORBANCE' + '\n')

        firstx, lastx = dataset.x.data[0], dataset.x.data[-1]
        firsty, lasty = dataset.data[0,0], dataset.data[0,-1]

        f.write('##FIRSTX=' + str(firstx) + '\n')
        f.write('##LASTX=' + str(lastx) + '\n')
        f.write('##FIRSTY=' + str(firsty) + '\n')
        f.write('##LASTY=' + str(lasty) + '\n')
        if firstx > lastx:
            maxx, minx = firstx, lastx
        else:
            maxx, minx = lastx, firstx
        nx = dataset.shape[1]
        f.write('##MAXX=' + str(maxx) + '\n')
        f.write('##MINX=' + str(minx) + '\n')
        maxy, miny = np.nanmax(dataset.data), np.nanmin(dataset.data)
        f.write('##MAXY=' + str(maxy) + '\n')
        f.write('##MINY=' + str(miny) + '\n')
        f.write('##XFACTOR=1.000000' + '\n')
        f.write('##YFACTOR=1.000000E-08' + '\n')
        f.write('##NPOINTS=' + str(nx) + '\n')
        yfactor = 1e-8
        f.write('##XYDATA=(X++(Y..Y))' + '\n')
        x = str(firstx)  # first x
        f.write(x + ' ')  # Write the first x
        y = str(int(firsty / yfactor))  # first y
        f.write(y + ' ')  # write first y
        llen = len(x) + len(y) + 2  # length of current line
        for j in np.arange(1, nx):
            if np.isnan(dataset.data[i, j]):
                y = '?'
            else:
                y = str(int(dataset.data[i, j] / yfactor))
            f.write(y + ' ')
            llen = llen + len(y) + 1
            if llen > 75:
                x = str(dataset.x.data[j])
                f.write('\n' + x + ' ')
                llen = len(x) + 1
        f.write('\n' + '##END' + '\n')

    f.write('##END=' + '\n')
    f.close()
