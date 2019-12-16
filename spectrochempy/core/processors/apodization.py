# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = []

__dataset_methods__ = []

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from ...units import ur, Quantity
from ...utils import epsilon, docstrings
from .. import general_preferences, error_

# ======================================================================================================================
# generic apodization function
# ======================================================================================================================
@docstrings.get_sectionsf('apodize', sections=['Parameters','Returns'])
@docstrings.dedent
def apodize(dataset, method, apod, inv=False, rev=False, inplace=False, dim=-1, **kwargs):
    """
    Calculate an apodization window function

    In principle, this function should not be called directly: its purpose is to be used by `em`, `gm`, `sinm`, etc...
    kind of apodization functions.

    Parameters
    ----------
    dataset : |NDDataset| or |NDPanel|.
        Dataset we want to apodize
    method : Callable.
        Apodization function
    apod : tuple.
        Apodization window parameter(s) depending on the actual apodization `method` used.
    inv : bool, optional, default=False.
        True for inverse apodization.
    rev : bool, optional, default=False.
        True to reverse the apodization before applying it to the data.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new dataset
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    
    Returns
    -------
    object : |NDDataset| or |NDPanel|.
        The apodized dataset if apply is True, the apodization array if not True.

    """

    # output dataset inplace (by default) or not
    if not inplace:
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset
        
    # On which axis do we want to apodize? (get axis from arguments)
    axis, dim = dataset.get_axis(dim, negative_axis=True)
    
    # The last dimension is always the dimension on which we apply the apodization window.
    # If needed, we swap the dimensions to be sure to be in this situation

    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    lastcoord = new.coords[dim]
    
    # compute the apodization function
    x = lastcoord
    apod_arr = method(x, *apod)

    if rev:
        apod_arr = apod_arr[::-1] # reverse apodization

    if inv:
        apod_arr = 1. / apod_arr  # invert apodization

    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    # TODO: CHECK IF THIS WORK WITH 2D DATA - IMPORTANT - CHECK IN PARTICULAR IF SWAPING ALSO SWAP METADATA (NOT SURE FOR NOW)
    iscomplex = new.is_complex
    isquaternion = new.is_quaternion
    encoding = new.meta.encoding[-1]
    # TODO: handle this eventual complexity

    new._data *= apod_arr

    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace

    name = method.__module__.split('.')[-1]
    new.history = f'{name} apodization performed on dimension {dim} with parameters:' + str(apod)
    # TODO: improve display of apod parameters

    return new, apod_arr

    # shifted = args.shifted  # float(kargs.get('top', 0.0))
    # k_shifted = args.k_shifted
    #
    # if k_shifted and (axis == -1 or axis == 1) and dataset.is_2d:
    #     # in k shifted method for 2D spectra, the top of
    #     # the broadening function follow the top of the echoes
    #     # parameters p and q should be defined in parameters
    #     command = 'k_shifted_em'
    #
    #     p = k_shifted[0]  # coherence
    #     q = k_shifted[1]  # satellite order
    #
    #     if p == 1 and q == 0:
    #         ratio = 1.0
    #     else:
    #         ratio = k_ratio(dataset, p, q)
    #     shifted = np.abs(ratio)
    #     if args.verbose:
    #         print('k_ratio: %.3f' % ratio)
    #
    # par.LB = lb
    #
    # kargs['lb'] = lb / dataset.get_multiplicator(axis)
    # kargs['shifted'] = shifted
    # kargs['states'] = True if 'STATES' in par.encoding else False
    #

    #
    # if axis == 0:
    #     # transpose temporarily the data for indirect dimension ft
    #     datv = datv.T
    #
    # dataset.data = datv
    # dataset.history.append('Exponential apodization lb:%.2f' % par.LB)
