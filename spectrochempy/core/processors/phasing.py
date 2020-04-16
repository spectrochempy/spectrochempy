# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["pk", "apk", "autophase", "interact_pk", "pc", "mc"]
__dataset_methods__  = __all__

# imports
import numpy as np

from ipywidgets import FloatSlider, interactive
from IPython.display import display
import matplotlib.pyplot as pl

#from masai.core.utils.misc import position2index, getsection

from ..fitting.parameters import FitParameters
from ..fitting.optimization import optimize

from .baseline import BaselineCorrection
#from ..analysis.picking import picking

from .. import error_, warning_, print_, debug_
from ...units import ur

# ======================================================================================================================
# pk function
# ======================================================================================================================
def pk(dataset, phc0=None, phc1=None, pivot='auto',  dim=-1, inplace=False, interactive=False, **kwargs):
    """
    Manual phasing of 1D or 2D NMR spectra

    Parameters
    ----------
    phc0 : float or quantity with deg units, optional
        Zero order (constant) phase correction
    phc1 : float or quantity with deg units, optional
        First order (linear) phase correction
    pivot : float or quantity with hz or ppm units or 'auto', optional, default='auto'
        Pivot position in spectral units, except if set to 'auto': in this case it is the absolute max position.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new dataset
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.

    """
    debug_('MANUAL PHASING MODE')
    
    # output dataset inplace or not
    if not inplace:           # default
        new = dataset.copy()
    else:
        new = dataset

    # On which axis do we want to phase? (get axis from arguments)
    # The last dimension is always the dimension on which we apply the phases.
    # If needed, we swap the dimensions to be sure to be in this situation

    axis, dim = new.get_axis(dim, negative_axis=True)
    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    # select the last coordinates and check the unit validity
    lastcoord = new.coords[dim]
    if (lastcoord.units.dimensionality != '1/[time]' and lastcoord.units != 'ppm'):
        error_('Phasing apply only to dimensions with [frequency] dimensionality or with ppm units\n'
               'Phasing processing was thus cancelled')
        return new

    # get initial absolute phase and pivot
    # get the initial phase setting
    debug_(f'Current phases : {new.meta.phc0[-1]}, {new.meta.phc1[-1]}')

    # make metadata updatable
    readonly = new.meta.readonly
    if readonly:
        new.meta.readonly = False # allow updating phases

    # read the phases and pivot arguments

    if phc0 is None:
        # phase were not given, read stored phase
        phc0 = new.meta.phc0[-1].copy()
        if not new.meta.phased[-1]:
            new.meta.phc0[-1] = 0. * ur.deg
    
    if not hasattr(phc0, 'units'):
        phc0 = phc0 * ur.deg
    rphc0 = phc0.copy() - new.meta.phc0[-1]
    
    if phc1 is None:
        # first order phase was not given, read stored phase
        phc1 = new.meta.phc1[-1].copy()
        if not new.meta.phased[-1]:
            new.meta.phc1[-1] = 0. * ur.deg
    
    if not hasattr(phc1, 'units'):
        phc1 = phc1 * ur.deg
    rphc1 = phc1.copy() - new.meta.phc1[-1]

    oldpivot = new.meta.pivot[-1]
    poldpivot = lastcoord.loc2index(oldpivot.m) - lastcoord.size/2
    
    if pivot == 'auto':
        pivot = float(abs(new).max().coords[-1].data)
    
    if not hasattr(pivot, 'units'):
        pivot = pivot * lastcoord.units
    ppivot = lastcoord.loc2index(pivot.m) - lastcoord.size/2

    pivot_phc0 = phc1 * (poldpivot - ppivot)/lastcoord.size
    
    debug_(f'Phases demanded for axis {dim} : {phc0}, {phc1} with pivot: {pivot}' )
    debug_(f'Actual phases to apply for axis {dim} : {rphc0 + pivot_phc0}, {rphc1}' )

    
    # do processing
    data = _ps(new.data, rphc0, rphc1)
    
    new.data = data
    new.meta.phc0[-1] = phc0
    new.meta.phc1[-1] = phc1
    new.meta.pivot[-1] = pivot
    
    new.history = f'Manual phasing phc0:{phc0}, phc1:{phc1}, pivot:{pivot}'

    if readonly :
        new.meta.readonly = True # reset readonly status

    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace
        
    return new


def timeshift(dataset, timeshift):
    """
    
    Parameters
    ----------
    dataset
    timeshift : float of quantity with time units, optional, default=0
        Shift of the time origin. This allow calculation of of linear phase correction corresponding
        to the given time shift

    Returns
    -------

    """
    #TODO
    if timeshift is None:
        # if timeshift is not given read stored data
        rtimeshift = timeshift = new.meta.timeshift[-1]
    else:
        #store the readed timeshift
        rtimeshift = timeshift - new.meta.timeshift[-1]

    # Calculate linear phase correction from a timeshift
    sw = new.meta.sw_h[-1]
    p_shift = (360. * timeshift * sw).to_base_units() * ur.deg
    rp_shift = (360. * rtimeshift * sw).to_base_units() * ur.deg


#
# autophase
#

# boundary checking function
def _checkin(ph, bp):
    if ph < -bp:
        ph = -bp
    if ph > bp:
        ph = bp
    return ph


def _neg_peak_score(data):
    """
    Phase correction using simple minima-minimisation around highest peak

    This is a naive approach but is quick and often achieves reasonable
    results.  The optimisation is performed by finding the highest peak in the
    spectra (e.g. TMSP) and then attempting to reduce minima surrounding it.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.

    Returns
    -------
    err : float
        Value of the objective function (phase score)

    """

    data = np.real(data)
    
    i = np.argmax(data)
    mina = np.min(data[i-100:i])
    minb = np.min(data[i:i+100])
    
    return np.abs(mina - minb)


def _neg_area_score(data):
    fm = data[data.real <= 0].real
    fm = fm - np.min(fm)
    err = np.sum(fm ** 2)

    # normalisation
    fp = data.real
    fp = fp - np.min(fm)
    err = err * 100. / (np.sum(fp ** 2) + 1.e-30)
    
    return err


def _acme_score(data):
    """
    Phase correction using ACME (Automated phase Correction based on Minimization of Entropy) algorithm by Chen Li et al.
    Journal of Magnetic Resonance 158 (2002) 164-168
    Copied NMRGLUE (see function autophase)
    
    Parameters
    ----------
    pd : tuple
        Current p0 and p1 values
    data : ndarray
        Array of NMR data.

    Returns
    -------
    score : float
        Value of the objective function (phase score)

    """
    stepsize = 1
    
    data = data.real
    
    # Calculation of first derivatives
    ds1 = np.abs((data[1:]-data[:-1]) / (stepsize*2))
    p1 = ds1 / np.sum(ds1)
    
    # Calculation of entropy
    p1[p1 == 0] = 1
    
    h1 = -p1 * np.log(p1)
    h1s = np.sum(h1)
    
    # Calculation of penalty
    pfun = 0.0
    as_ = data - np.abs(data)
    sumas = np.sum(as_)
    
    if sumas < 0:
        pfun = pfun + np.sum((as_/2) ** 2)
    
    p = 1000 * pfun
    
    return h1s + p

# ----------------------------------------------------------------------------------------------------------------------
def apk(dataset, dim=-1, **kwargs):
    """
    Automatic phasing of 1D or 2D spectra
    
    Parameters:
    -----------
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new object
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    algorithm : str or function, optional, default='negmin'
        Algorithm to use for phase scoring. Built in functions can be
        specified by one of the following strings : "peak_minima", "negmin", "acme",...
    optmode: str, optional, default='simplex'
        Method of optimisation, choices=['simplex', 'hopping']
    
    ediff: int, optional, default=2
        order of derivation for method entropy
    gamma: float, optional, default=1.0
        relative weight for method entropy error with respect to negmin
    fit_phc1: bool, optional, default=False
        also optimize first order phase if True
    bound_phc0 : float, optional, default=360. deg
        phc0 boundary in degree
    bound_phc1 : float, optiopnal, default=10. deg
        phc1 boundary in degree



    # parser.add_argument('--pos', default=(0,), nargs='*', type=float,
    #                     help='row or column position where to check phase')
    # parser.add_argument('--shifted', default=0.0, type=float,
    #                     help="position of the top in units of time")
    # parser.add_argument('--fit_phc1', '-u1', action='store_true', help='use phc1 in automatic phasing', )
    # parser.add_argument('--gamma', '-ga', default=1.0, type=float, help='weight', )
    # parser.add_argument('--select', '-st', default='standard', choices=['standard', 'max', 'cols','pos'],
    #                     help='selection mode in automatic phasing', )
    # parser.add_argument('--threshold', '-th', default=50.0, type=int, help='default threshold for columns selection', )

    # parser.add_argument('--byrow', '-br', action='store_true', help='to phase each row separately for series or 2D')
    # #parser.add_argument('--absolute', action='store_true', help='absolute flag: take the absolute value of phases')
    #
    # args = parser.parse_args(options.split())

    """
    debug_(f'AUTOMATIC PHASE MODE')

    # output dataset inplace or not
    if not kwargs.pop('inplace', False):
        # default
        new = dataset.copy()
    else:
        new = dataset

    # On which axis do we want to phase? (get axis from arguments)
    # The last dimension is always the dimension on which we apply the phases.
    # If needed, we swap the dimensions to be sure to be in this situation
    axis, dim = new.get_axis(dim, negative_axis=True)
    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    # select the last coordinates and check the unit validity
    lastcoord = new.coords[dim]
    if (lastcoord.units.dimensionality != '1/[time]' and lastcoord.units != 'ppm'):
        error_('Phasing apply only to dimensions with [frequency] dimensionality or with ppm units\n'
               'Phasing processing was thus cancelled')
        return new

    # get initial absolute phase and pivot
    # get the initial phase setting
    debug_(f'Current phases : {new.meta.phc0[-1]}, {new.meta.phc1[-1]}')
        
    #if args.select == 'cols' and source.is_2d and axis == 0:
    #    ar = picking(source, args.threshold, index=True)
    #    args.cols = zip(*ar)[1]

    #if args.byrow and source.is_2d:
    #    rows = []
    #    for index in range(data.index.size):
    #        row = data.iloc[index:index + 1].values
    #        row, phc0, phc1 = autophase(row, args)
    #        row = row - basecorr(row)
    #        rows.append(row)
    #merge all rows to recreate data
    #    data = np.vstack(rows)
    #else:

    # 2D Case only

    # select = args.select
    #
    # if select == 'standard' or data.shape[0] == 1:
    #     # select the first row
    #     dat = dat[0]
    #     select = 'standard'
    #
    # if select == 'max' and data.ndim > 1:
    #     i, j = np.unravel_index(data.real.argmax(), data.shape)
    #     dat = data[i]
    #
    # if select == 'cols' and data.ndim > 1:
    #     l = []
    #     cols = args.cols
    #     if verbose:
    #         print('columns selected:', cols)
    #     for col in cols:
    #         i = int(col)
    #         l.append(data[i])
    #     dat = np.vstack(l)

    # apkmode
    # --------
    #if DEBUG:
    #    print('\nselect: ', select)

    
    readonly = new.meta.readonly
    if readonly:
        new.meta.readonly = False # allow updating phases
        
    phc0 = new.meta.phc0[-1]
    phc1 = new.meta.phc1[-1]
    fit_phc1 = kwargs.pop('fit_phc1', True)
    pivot = float(abs(new).max().coords[-1].data)
    ppivot = lastcoord.loc2index(pivot)
    
    # prepare the phase to be optimized with bound
    fp = FitParameters()
    
    bp0 = kwargs.get('bound_phc0',360. * ur.deg)
    phc0 = _checkin(phc0, bp0)
    fp['phc0'] = phc0, -bp0, bp0, False

    bp1 = kwargs.get('bound_phc1',20. * ur.deg)
    phc1 = _checkin(phc1, bp1)
    fixed = not fit_phc1
    fp['phc1'] = phc1, -bp1, bp1, fixed

    # algorithm to compute the cost function
    algorithm= kwargs.pop('algorithm','neg_area')
    alg = {
        'neg_peak': _neg_peak_score, # Phase correction using simple minima-minimisation around highest peak
        'neg_area': _neg_area_score, # negative area minimization
        'acme': _acme_score,         # entropy minimization
    }
    if not callable(algorithm):
        if algorithm not in alg.keys():
            error_("algorithm must be a callable or a string among 'neg_area', 'neg_peak" or 'acme')
            return new
        algorithm = alg[algorithm]
        
    # cost function
    def _cost_function(p, s):

        p0 = p['phc0']
        p1 = p['phc1']
        sc = s.copy()
        sc = _ps(sc, p0, p1)

        # baseline correction #TODO
        scp = sc #- basecorr(sc)

        # score computation
        return algorithm(scp)

    # convergence is not insured depending on the starting values
    optmode= kwargs.pop('optmode','simplex')
    fp, err = optimize(_cost_function, fp,
                        args=[new.data, ],
                        method=optmode,
                        callback=None)

    # extract results
    phc0 = fp['phc0']
    phc1 = fp['phc1']

    debug_(f'Calculated phases (cost: {err}):')

    if fit_phc1:
        debug_(f'phc0: {phc0}')
        debug_(f'phc1: {phc1}')
    else:
        debug_(f'phc0: {phc0}')
        debug_(f'phc1 (not optimized): {phc1}')

    # apply to the original data and return
    data = _ps(new.data, phc0, phc1)


    # atxt = '(not optimized)' if not args.fit_phc1 else ''
    #
    # sbyrow = 'byrow' if args.byrow else ''
    # source.history.append('Auto-phasing %s:  phc0 = %.3f, phc1%s = %.3f, pivot:%.2f' % (sbyrow, phc0, atxt, phc1, pivot))
    # # store the new phases
    # par.PHC0, par.PHC1 = phc0 , phc1

    new.data = data
    new.meta.phc0[-1] = phc0
    new.meta.phc1[-1] = phc1
    
    new.history = f'Automatic phasing phc0:{phc0}, phc1:{phc1}'
    
    if readonly :
        new.meta.readonly = True # reset readonly status

    # restore original data order if it was swaped
    if swaped:
        new.swapaxes(axis, -1, inplace=True)  # must be done inplace
    
    return new

autophase = apk


# ----------------------------------------------------------------------------------------------------------------------
def interact_pk(dataset, dim=-1, **kwargs):
    
    debug_('INTERACTIVE PHASING MODE')
    
    new = dataset.copy()
    
    # On which axis do we want to phase? (get axis from arguments)
    # The last dimension is always the dimension on which we apply the phases.
    # If needed, we swap the dimensions to be sure to be in this situation
    axis, dim = new.get_axis(dim, negative_axis=True)
    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True
    
    # select the last coordinates and check the unit validity
    lastcoord = new.coords[dim]
    if (lastcoord.units.dimensionality != '1/[time]' and lastcoord.units != 'ppm'):
        error_('Phasing apply only to dimensions with [frequency] dimensionality or with ppm units\n'
               'Phasing processing was thus cancelled')
        return new
    
    # get initial absolute phase and pivot
    # get the initial phase setting
    debug_(f'Current phases : {new.meta.phc0[-1]}, {new.meta.phc1[-1]}')

    phc0 = new.meta.phc0[-1]
    phc1 = new.meta.phc1[-1]
    pivot = float(abs(new).max().coords[-1].data) * lastcoord.units
    
    def phasing(ph0, ph1, pivot):
        
        global interact
        
        rphc0 = (ph0 - phc0)
        rphc1 = (ph1 - phc1)
        ppivot = lastcoord.loc2index(pivot.m)
        
        # do processing
        data = _ps(new.data, rphc0, rphc1)

        new.data = data
        new.meta.phc0[-1] = phc0
        new.meta.phc1[-1] = phc1
        
        new.plot()
    
    w = interactive(phasing,
                    ph0=FloatSlider(min=phc0-45, max=phc0+45, step=0.001,
                                    value=phc0, continuous_update=False),
                    ph1=FloatSlider(min=phc1-180, max=phc1+180, step=0.01,
                                    value=phc1, continuous_update=False),
                    pivot=FloatSlider(min=lastcoord.min(),
                                      max=lastcoord.max(), step=0.001,
                                      value=pivot,
                                      continuous_update=False))
    
    output = w.children[-1]
    display(w)
    
    return w

# ======================================================================================================================
# pc : power calculation
# ======================================================================================================================
def pc(dataset, dim=-1, **kwargs):
    """
    Power calculation

    Parameters
    ----------
    dataset : |NDDataset| or |NDPanel|.
        Input dataset or panel
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
        
    Returns
    -------
    out : same type as input dataset
        the modified object
        
    """

    # output dataset inplace or not
    if not kwargs.pop('inplace', False):
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset

    # On which axis do we want to apply the calculation ?
    axis, dim = dataset.get_axis(dim, negative_axis=True)

    # The last dimension is always the dimension on which we apply the apodization window.
    # If needed, we swap the dimensions to be sure to be in this situation

    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    # select the last coordinates and check the unit validity
    lastcoord = new.coords[dim]
    if (lastcoord.units.dimensionality != '1/[time]' and lastcoord.units != 'ppm'):
        error_('`ab` apply only to dimensions with [frequency] dimensionality or with ppm units\n'
               'Baseline correction processing was thus cancelled')
        return new


    # data on which we apply pc
    
    datv = source.data.values
    if axis == 0:
        # transpose temporarily the data for indirect dimension ft
        datv = datv.T
    
    npc = args.npc
    datv = np.abs(datv) ** npc
    
    # un-transpose the data if needed
    if axis == 0:
        datv = datv.T
    
    # change the current data with the transformed data
    source.data = datv  # scale will be calculated here
    
    new.history = 'Power spectrum calculation'
    
    return new

# ======================================================================================================================
# mc : magnitude calculation
# ======================================================================================================================
def mc(dataset, dim=-1, **kwargs):
    """
    Magnitude calculation

    Parameters
    ----------
    dataset : |NDDataset| or |NDPanel|.
        Input dataset or panel
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
        
    Returns
    -------
    out : same type as input dataset
        the modified object
        
    """

    return np.sqrt(pc(dataset, dim=dim, **kwargs))


# ======================================================================================================================
# utilities functions
# ======================================================================================================================
def _ps(data, p0=0.0, p1=0.0):
    """
    Linear phase correction

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    p0 : float
        Zero order phase in degree
    p1 : float
        First order phase in degree

    Returns
    -------
    out : ndarray
        Phased NMR data.

    """
    p0r = p0.copy().to('rad').m
    p1r = p1.copy().to('rad').m
    size = data.shape[-1]
    out = data * np.exp(1.0j * (p0r + p1r * np.arange(size)/ size)).astype(data.dtype)
    
    return out

if __name__ == '__main__':
    pass