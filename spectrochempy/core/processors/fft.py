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
import re

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np

# ======================================================================================================================
# Local imports
# ======================================================================================================================
from ...extern.nmrglue.process.proc_base import largest_power_of_2, zf_size
from .. import error_
from ...units import ur

def ifft(dataset, size=None, inplace=False, **kwargs):
    r"""
    Apply inverse fast fourier transform.
    (see `fft` documentation.)
    """
    return fft(dataset, size=size, inv=True, inplace=inplace, **kwargs)


def fft(dataset, size=None, sizeff=None, inv=False, inplace=False, dim=-1, ppm=True, **kwargs):
    r"""
    Apply a complex fast fourier transform.
    
    For multidimensional NDDataset or NDPanels,
    the apodization is by default performed on the last dimension.
    
    The data in the last dimension MUST be in time-domain (or without dimension)
    or an error is raised.
    
    To make reverse Fourier transform, i.e., from frequency to time domain, use the `ifft` transform
    (or equivalently, the `inv=True` parameters.
    
    Parameters
    ----------
    size : int, optional
        size of the transformed dataset dimension - a shorter parameter is `si`. by default, the size is the closest
        power of two greater than the data size
    sizeff : int, optional
        The number of effective data point to take into account for the transformation. By default it is equal to the
        data size, but may be smaller.
    inv : bool, optional, default=False
        if True, an inverse Fourier transform is performed - size parameter is not taken into account
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new dataset
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    ppm : bool, optional, default=True
        If True, and data are from NMR, then a ppm scale is calculated instead of frequency.
    **kwargs :
        other parameters (see other parameters)
    
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
    
    if (not inv and lastcoord.units.dimensionality != '[time]'):
        error_('fft apply only to dimensions with [time] dimensionality\n'
               'fft processing was thus cancelled')
        return dataset
    
    elif (inv and lastcoord.units.dimensionality != '1/[time]' and lastcoord.units != 'ppm'):
        error_('ifft apply only to dimensions with [frequency] dimensionality or with ppm units\n'
               ' ifft processing was thus cancelled')
        return dataset
    
    elif dataset.is_masked :
        error_('current fft or ifft processing does not support masked data as input.\n processing was thus cancelled')
        
    # TODO: other tests data spacing and so on.

    # output dataset inplace or not
    if not inplace:           # default
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset         #

    # Can we use some metadata as for NMR spectra
    if is_nmr and not inv:
        td = dataset.meta.td[-1]
    else:
        td = lastcoord.size
        
    # if no size (or si) parameter then use the size of the data (size not used for inverse transform
    if size is None or inv:
        size = kwargs.get('si', td)

    # we default to the closest power of two larger of the data size
    size = largest_power_of_2(size)
        
    # do we have an effective td to apply
    tdeff = sizeff
    if tdeff is None:
        tdeff = kwargs.get("tdeff", td)
    
    if tdeff is None or tdeff<5 or tdeff>size:
        tdeff = size
        
    # Eventually apply the effective size
    new[...,tdeff:] = 0.
    
    # should we work on complex data
    iscomplex = new.is_complex
    
    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    encoding = None
    if is_nmr and not inv:
        encoding = new.meta.encoding[-1]
    
    # perform the fft
    if iscomplex and encoding in ['QSIM', 'DQD']:
        data = zf_size(new.data, size)
        data = np.fft.fft(data)
        data = np.fft.fftshift(data, -1)
    elif inv:
        # we assume no special encoding for inverse fft transform
        data = np.fft.ifftshift(new.data, -1)
        data = np.fft.ifft(data)
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
        
    else:
        sfo1 = 1.0 * ur.Hz
        bf1 = sfo1
        sf = 0 * ur.Hz
        dw = lastcoord.spacing
        sw = 1. / dw
        
    if not inv:
        # time to frequency
        sizem = max(size - 1, 1)
        deltaf = -sw / sizem
        first = sfo1 - sf - deltaf * sizem / 2.
        
        newcoord = type(lastcoord)(np.arange(size)*deltaf + first)
        newcoord.name = lastcoord.name
        newcoord.title = 'frequency'
        newcoord.ito("Hz")
        
    else:
        # frequency or ppm to time
        sw = abs(lastcoord.data[-1]-lastcoord.data[0])
        if lastcoord.units == 'ppm':
            sw = bf1.to("Hz") * sw / 1.0e6
        deltat = 1. /sw
        
        newcoord = type(lastcoord)(np.arange(size)*deltat)
        newcoord.name = lastcoord.name
        newcoord.title = 'time'
        newcoord.ito("s")
        
    if is_nmr and not inv:
        newcoord.meta.larmor = bf1  # needed for ppm transformation
        newcoord.origin = 'bruker'
        if ppm:
            newcoord.ito('ppm')
            if new.meta.nuc1 is not None:
                nuc1 = new.meta.nuc1[-1]
                regex = r"([^a-zA-Z]+)([a-zA-Z]+)"
                m = re.match(regex, nuc1)
                mass = m[1]
                name = m[2]
                nucleus = '^{' + mass + '}' + name
            else:
                nucleus = ""
            newcoord.title = fr"$\delta\ {nucleus}$"
    
    new.coords[-1]=newcoord

    # if some phase related metadata do not exist yet, initialize them
    new.meta.readonly = False
    
    if not new.meta.phased:
        new.meta.phased = [False] * new.ndim
        
    if not new.meta.pivot:
        new.meta.pivot = [float(abs(new).max().coords[i].data) * new.coords[i].units for i in range(new.ndim) ]  # create pivot metadata
        
    # applied the stored phases
    new.pk(inplace=True)
    
    new.meta.phased[-1] = True
    
    new.meta.readonly = True

    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace
        
    new.history = f'fft applied on dimension {dim}'
    
    return new


# ======================================================================================================================
if __name__ == '__main__':  # pragma: no cover
    pass