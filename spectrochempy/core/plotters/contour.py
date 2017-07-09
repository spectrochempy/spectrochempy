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



"""Plugin module to perform automatic subtraction of ref on a dataset.

"""

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Global preferences
# ==============================================================================
from spectrochempy.application import SCP

_DO_NOT_BLOCK = SCP.plotoptions.DO_NOT_BLOCK


__all__ = ['contour']

def contour(ds, nfig = None, invertxaxis=True,
            formatstring = '-'):
    """Plots dataset contour
    
    :param nfig: figure number. If None is provided, a new figure is created  
    :type nfig: int
    :param invertaxis: whether the x axis is inverted for plotting  
    :type invertaxis: bool
    :param fomatstring: indicates the color and line type of the plot
    :type invertaxis: str

     """                     
    #Check the dimensions:         
    if ds.data.ndim > 2:
        print('Warning: n-way data only the first slab will be plot')
    
    xaxis = ds.dims[1].axes[0].values   
    yaxis = ds.dims[0].axes[0].values
    if nfig == None:
            plt.figure()
    else:
        plt.figure(nfig)      
    if xaxis.size == 0 and yaxis.size == 0:
        plt.contourf(ds.data)
    else:
        plt.contourf(xaxis, yaxis, ds.data)
        plt.xlabel(ds.dims[1].axes[0].name)
    if invertxaxis:
        [xleft, xright] = plt.gca().get_xlim()
        if xleft < xright:
            plt.gca().invert_xaxis()
    if yaxis.size == 0:
        plt.ylabel('')
    else:
        plt.ylabel(ds.dims[0].axes[0].name)
    plt.title(ds.name)

    if not _DO_NOT_BLOCK:
        plt.show()

#------------------------------------------------------------------------------
from ..dataset import NDDataset
setattr(NDDataset, 'contour', contour)