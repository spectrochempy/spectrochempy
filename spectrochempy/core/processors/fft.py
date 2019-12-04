# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["fft", "ifft"]

__dataset_methods__  = __all__

#======================================================================================================================
# Standard python imports
# ======================================================================================================================

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from spectrochempy.utils import closer_power_of_two
from spectrochempy.units import set_nmr_context
from spectrochempy.core import project_preferences, general_preferences, error_, warning_


epsilon = np.finfo(float).eps


# ======================================================================================================================
# interface for the processing class
# ======================================================================================================================

# ======================================================================================================================
# generic transform function
# ======================================================================================================================

def ifft(dataset, size=None, inplace=True, **kwargs):
    r"""
    Apply inverse fast fourier transform.
    (see `fft` documentation.)
    """
    return fft(dataset, size=size, inv=True, inplace=inplace, **kwargs)


def fft(dataset, size=None, sizeff=None, inv=False, inplace=True, dim=-1, **kwargs):
    r"""
    Apply a complex fast fourier transform.

    For multidimensional NDDataset or NDPanels,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be in time-domain (or without dimension)
    or an error is raised.

    To make reverse Fourier transform, i.e., from frequency to time domain, use the `ifft` transform (or equivalently, the `ìnv`=True parameters.

    Parameters
    ----------
    size : int, optional
        size of the transformed dataset dimension - a shorter parameter is `si`. by default, the size is the closest
        power of two greater than the data size
    sizeff : int, optional
        The number of effective data point to take into account for the transformation. By default it is equal to the
        data size, but may be smaller.
    inv : bool, optional, default=False
        if True, an inverse Fourier transform is performed
    inplace : bool, optional, default=True.
        True if we make the transform inplace.  If False, the function return a new dataset
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    **kwargs : other parameters (see other parameters)

    Other Parameters
    ----------------
    tdeff : int, optional
        alias of sizeff (specific to NMR). If both sizeff and tdeff are passed, sizeff has the priority.
        
    Returns
    -------
    object : nd-dataset or nd-array
        transformed dataset

    """
    # datatype
    is_nmr = dataset.origin in ["bruker",]
    
    # On which axis do we want to apodize? (get axis from arguments)
    axis, dim = dataset.get_axis(dim, negative_axis=True)
    
    # The last dimension is always the dimension on which we apply the fourier transform.
    # If needed, we swap the dimensions to be sure to be in this situation
    
    swaped = False
    if axis != -1:
        dataset.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True
    
    # select the last coordinates
    lastcoord = dataset.coords[dim]
    
    if (not inv and not lastcoord.unitless and not lastcoord.dimensionless and lastcoord.units.dimensionality != '[time]'):
        error_('fft apply only to dimensions with [time] dimensionality, unitless or dimensionless data\n'
               'fft processing was thus cancelled')
        return dataset
    
    elif (inv and not lastcoord.unitless and not lastcoord.dimensionless and lastcoord.units.dimensionality != '[frequency]'):
        error_('ifft apply only to dimensions with [frequency] dimensionality, unitless or dimensionless data\n'
               'ifft processing was thus cancelled')
        return dataset
    
    elif dataset.is_masked :
        error_('current fft or ifft processing does not support masked data as input.\n processing was thus cancelled')
        
    # TODO: other tests data spacing and so on.

    # output dataset inplace (by default) or not
    if not inplace:
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset

    # Can we use some metadata as for NMR spectra
    if is_nmr:
        td = dataset.meta.td[-1]
    else:
        td = lastcoord.size
        
    # if no size (or si) parameter then use twice the size of the data
    if size is None:
        size = kwargs.get('si', td * 2)

    # we default to the closest power of two larger than the data size
    size = closer_power_of_two(size)
        
    # do we have an effective td to apply
    tdeff = sizeff
    if tdeff is None:
        tdeff = kwargs.get("tdeff", td)
    
    if tdeff is None or tdeff<5 or tdeff>size:
        tdeff =  size
        
    # Eventually apply the effective size
    new[...,tdeff:] = 0.
    
    # should we work on complex data
    iscomplex = new.is_complex
    
    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    encoding = None
    if is_nmr:
        encoding = new.meta.encoding[-1]
    
    # perform the fft
    if iscomplex and encoding in ['QSIM', 'DQD']:
        data = np.fft.fft(new.data, size)
        data = np.fft.fftshift(data, -1)
    else:
        raise NotImplementedError(encoding)
    
    # We need here to create a new dataset with new shape and axis
    new._data = data
    new.mask = False # TODO: make a test on mask - should be none before fft!
    
    # create new coordinates for the transformed data
    
    if is_nmr:
        sfo1 = new.meta.sfo1[-1]
        bf1 = new.meta.bf1[-1]
        sf = new.meta.sf[-1]
        sw = new.meta.sw_h[-1]
    
    sizem = max(size - 1, 1)
    delta = -sw / sizem
    first = sfo1 - sf - delta * sizem / 2.
    
    newcoord = type(lastcoord)(np.arange(size)*delta + first)
    newcoord.name = lastcoord.name
    newcoord.title = 'frequency'
    newcoord.units = "Hz"
    
    if is_nmr:
        newcoord.meta.larmor = bf1  # needed for ppm transformation
        newcoord.origin = 'bruker'
        
    new.coords[-1]=newcoord
    
    
    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace
    
    new.history = f'fft applied on dimension {dim}'
    
    return new


# ======================================================================================================================
if __name__ == '__main__':  # pragma: no cover
    from spectrochempy import *
    
    dataset1D = NDDataset()
    path = os.path.join('nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
    dataset1D.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
    
    dataset1D /= dataset1D.real.data.max()  # normalize
    
    p = dataset1D.plot()
    
    new = dataset1D.fft(tdeff=8192)

    # change frequency axis to ppm
    new.frequency.ito('ppm')
    
    new.plot()
    
    show()
