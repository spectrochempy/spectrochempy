# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['trapz']

__dataset_methods__  = __all__

import scipy.signal

"""wrapper of scipy.signal integration function"""

def trapz(dataset, dim='x'):
    '''
     Wrapper of scpy.signal.find_peaks(). Integrate along the given dim using the composite trapezoidal rule.

    Parameters
    ----------
    X : |NDDataset|
        A NDDataset to integrate

    dim: the dim along which to integrate. If the coord is reversed (e.g. IR data), positive peaks will
    have positive integral.

    Returns
    -------
    res : float or array of float
            Definite integral as approximated by trapezoidal rule'''

    if dim == 'x':
        axis = -1
    if dim == 'y':
        axis = -2
    if dim == 'z':
        axis = -3

    res = scipy.integrate.trapz(dataset.data, x=dataset.coord([dim]).data, axis=axis)
    if dataset.coord([dim]).reversed:
        res *= -1
    return res


