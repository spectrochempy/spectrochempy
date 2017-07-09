# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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

"""Plugin module to perform view of spectra along columns

"""

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Global preferences
# ==============================================================================
from spectrochempy.application import SCP

_DO_NOT_BLOCK = SCP.plotoptions.DO_NOT_BLOCK


__all__ = ['plotc']

def plotc(source,
          nfig = None,
          invertxaxis=False,
          formatstring ='-',
          **kwargs):

    """Plots the columns of a dataset contents.
    
    :param nfig: figure number. If None is provided, a new figure is created  
    :type nfig: int
    :param invertaxis: whether the x axis is inverted for plotting  
    :type invertaxis: bool
    :param fomatstring: indicates the color and line type of the plot
    :type invertaxis: str

     """                     
    #Check the dimensions:         
    if source.ndim > 2:
        print('Warning: nD>2D data only the first slab will be plotted')
    
    yaxis = source.axes[0]

    ydata = yaxis.data
    zdata = source.data

    if nfig == None:
        plt.figure()
    else:
        plt.figure(nfig)

    if len(yaxis) == 0:
        plt.plot(zdata, formatstring)
        plt.xlabel('row(X)')
    else:
        plt.plot(ydata, zdata, formatstring)
        plt.xlabel("{} [{:~P}]".format(yaxis.title, yaxis.units))

    if invertxaxis:
        [xleft, xright] = plt.gca().get_xlim()
        if xleft < xright:
            plt.gca().invert_xaxis()

    plt.ylabel("{} [{:~P}]".format(source.title, source.units))
    plt.title(source.name)

    if not _DO_NOT_BLOCK:
        plt.show()

#------------------------------------------------------------------------------
from ..dataset import NDDataset
setattr(NDDataset, 'plotc', plotc)