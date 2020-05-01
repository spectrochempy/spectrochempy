# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Description
# Utility functions to deal with Cantera input/output
#
# ---------------------------------------------------------------------------------

__all__ = ['coverages_vs_time', 'concentrations_vs_time', 'modify_rate',
           'modify_reactive_phase', 'fit_to_concentrations']

import cantera as ct
import datetime
import numpy as np
from scipy.optimize import minimize

from spectrochempy.core.dataset.nddataset import NDDataset, Coord


def coverages_vs_time(surface, t, returnNDDataset=False):
    ''' Returns the surface coverages at time(s) t
    params:
    ------
    surface: instance of cantera.composite.Interface
    t: iterable or spectrochempy.Coord, times at which the coverages must be computed
    return_NDDataset: boolean, if True returns the concentration matrix as a NDDataset, else as a np.ndarray
    default: False
    '''
    init_coverages = surface.coverages
    coverages = np.zeros((len(t), surface.coverages.shape[0]))

    if type(t) is Coord:
        t = t.data

    for i, ti in enumerate(t):
        surface.coverages = init_coverages
        surface.advance_coverages(ti)
        coverages[i, :] = surface.coverages
        surface.coverages = init_coverages
    if returnNDDataset:
        coverages = NDDataset(coverages)
        coverages.y = Coord(t, title='time')
        coverages.x.title = 'coverage / -'
        coverages.x.labels = surface.species_names
    return coverages

def concentrations_vs_time(reactive_phase, t, reactorNet = None, returnNDDataset=False):
    ''' Returns the  concentrations at time(s) t
    params:
    ------
    surface: instance of cantera.composite.Interface or
    t: iterable or spectrochempy.Coord, times at which the concentrations must be computed
    return_NDDataset: boolean, if True returns the concentration matrix as a NDDataset, else as a np.ndarray
    default: False
    '''

    if type(reactive_phase) is ct.composite.Interface:
        concentrations = coverages_vs_time(reactive_phase, t, returnNDDataset) * reactive_phase.site_density
        if returnNDDataset:
            concentrations.x.title = 'concentration'
        return concentrations

    else:
        raise NotImplementedError('not implmented for reactive_phase={}'.format(str(type(reactive_phase))))
        ## code for reactorNet
        # if type(t) is Coord:
        #     t = t.data
        #
        # for i, ti in enumerate(t):
        #     reactorNet.advance(ti)
        #     concentrations[i, :] = reactive_phase.concentrations
        #     reactive_phase.concentrations = init_concentrations
        #
        #
        # if returnNDDataset:
        #     concentrations = NDDataset(concentrations)
        #     concentrations.y = Coord(t, title='time')
        #     concentrations.x.title = 'concentrations'
        #     concentrations.x.labels = reactive_phase.species_names



def modify_rate(reactive_phase, i_reaction, rate):
    """ Changes one of the rates  of a cantera reactive phase.
        """
    rxn = reactive_phase.reaction(i_reaction)
    rxn.rate = rate
    reactive_phase.modify_reaction(i_reaction, rxn)
    return reactive_phase


def modify_reactive_phase(reactive_phase, param_to_change, param_value):
    """changes a set of numerical parameters of a cantera reactive phase.
    only implemented for cantera.composite.Interface and the following parameters:
    site_density, coverages, concentrations,
    pre-exponential factor, temperature_exponent, activation_energy
    """
    # check some parameters
    if type(reactive_phase) is not ct.composite.Interface:
        raise ValueError('only implemented of ct.composite.Interface')
    if len(param_to_change) != len(param_value):
        raise ValueError('number of parameters to change and values should be equal')

    for i, param in enumerate(param_to_change):

        # check that  param_to_change exists
        try:
            eval('reactive_phase.' + param)
        except:
            ValueError('class {} has no \'{}\' attribute'.format(type(reactive_phase), param))

        # if exists => sets its new value
        # if the attribute is writable:
        if param in ('site_density', 'coverages', 'concentrations'):
            exec('reactive_phase.' + param + '=' + str(param_value[i]))
        # else use Cantera methods (or derived from cantera)
        elif param.split('.')[-1] == 'pre_exponential_factor':
            str_rate = 'reactive_phase.' + '.'.join(param.split('.')[-3:-1])
            b, E = eval(str_rate + '.temperature_exponent,' + str_rate + '.activation_energy ' )
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(reactive_phase, rxn, ct.Arrhenius(param_value[i], b, E))

        elif param.split('.')[-1] == 'temperature_exponent':
            str_rate = 'reactive_phase.' + '.'.join(param.split('.')[-3:-1])
            A, E = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.activation_energy ')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(reactive_phase, rxn, ct.Arrhenius(A, param_value[i], E))

        elif param.split('.')[-1] == 'activation_energy':
            str_rate = 'reactive_phase.' + '.'.join(param.split('.')[-3:-1])
            A, b = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.temperature_exponent')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(reactive_phase, rxn, ct.Arrhenius(A, b, param_value[i]))
    return


def fit_to_concentrations(C, externalConc, external_to_C_idx, reactive_phase, param_to_optimize, guess_param, **kwargs):
    r"""
    Function fitting rate parameters and concentrations to a given concentration profile.
    
    Parameters
    ------------
    
    C: NDDataset
        experimental concentration profiles on which to fit the model. C can contain more concentration
        profiles than those to fit.
    externalConc:
        indexes of experimental concentration profiles on which the model will be fitted
    external_to_C_idx:
        correspondence between optimized (external) concentration profile and experimental
        concentration profile
    reactivePhase:
        cantera active phase. Currently implemented for surface only
    param_to_optimize: list
        list of reactive phase parameters to optomize
    guess_param:
        initial guess for the parameters to fit
    **kwargs:
        parameters for the optimization (see scipy.optimize.minimize)
    
    Returns
    ----------
    a dictionary
    
    """

    def objective(param_value, param_to_optimize, C, externalConc, external_to_C_idx, surface):
        modify_reactive_phase(surface, param_to_optimize, param_value)
        Chat = concentrations_vs_time(surface, C.y)
        return np.sum(np.square(C.data[:,externalConc] - Chat[:,external_to_C_idx]))

    method = kwargs.get("method", "Nelder-Mead")
    bounds = kwargs.get("bounds", None)
    tol = kwargs.get("tol", None)
    options = kwargs.get("options", {'disp': True})
    if options['disp']:
        print('Optimization of the parameters.')
        print('         Initial parameters: {}'.format(guess_param))
        print('         Initial function value: {}'.format(objective(guess_param, param_to_optimize, C, externalConc,
                                                                     external_to_C_idx,reactive_phase)))
    tic = datetime.datetime.now()
    res = minimize(objective, guess_param, args=(param_to_optimize, C,  externalConc, external_to_C_idx, reactive_phase),
                   method=method,  bounds=bounds, tol=tol, options=options)
    toc = datetime.datetime.now()
    guess_param = res.x
    if options['disp']:
        print('         Optimization time: {}'.format((toc - tic)))
        print('         Final parameters: {}'.format(guess_param))
    Ckin = concentrations_vs_time(reactive_phase, C.y, returnNDDataset=True)
    newargs = (reactive_phase, param_to_optimize, guess_param)
    return {'concentrations': Ckin,
            'results': res,
            'new_args': newargs}