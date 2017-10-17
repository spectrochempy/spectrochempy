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

__all__ = ['interleave', 'interleaved2complex']
import numpy as np

def interleave(data):
    """
    This function make an array where real and imaginary part are interleaved

    Parameters
    ==========
    data : complex ndarray
        If the array is not complex, then data are
        returned inchanged

    Returns
    =======
    data : ndarray with interleaved complex data

    iscomplex : is the data are really complex it is set to true

    """
    if np.any(np.iscomplex(data)) or data.dtype == np.complex:
        # unpack (we must double the last dimension)
        newshape = list(data.shape)
        newshape[-1] *= 2
        new = np.empty(newshape)
        new[..., ::2] = data.real
        new[..., 1::2] = data.imag
        return new, True
    else:
        return data, False

def interleaved2complex(data):
    """
    Make a complex array from interleaved data

    """
    return data[..., ::2] + 1j * data[..., 1::2]