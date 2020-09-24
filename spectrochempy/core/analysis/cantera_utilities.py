# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Description
# Utility functions to deal with Cantera input/output
#
# ---------------------------------------------------------------------------------

__all__ = ['surface_coverages_vs_time', 'surface_concentrations_vs_time', 'modify_rate',
           'reactors_composition_vs_time', 'modify_reactive_phase', 'fit_to_concentrations',
           'PFR']

import cantera as ct
import datetime
import numpy as np
from scipy.optimize import minimize
from collections.abc import Iterable

from spectrochempy.core.dataset.nddataset import NDDataset, Coord

def surface_coverages_vs_time(surface, t, returnNDDataset=False):
    ''' Returns the surface coverages at time(s) t of a reactive surface
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


def surface_concentrations_vs_time(reactive_phase, t, reactorNet=None, returnNDDataset=False):
    ''' Returns the  concentrations at time(s) t
    params:
    ------
    surface: instance of cantera.composite.Interface or
    t: iterable or spectrochempy.Coord, times at which the concentrations must be computed
    return_NDDataset: boolean, if True returns the concentration matrix as a NDDataset, else as a np.ndarray
    default: False
    '''

    if type(reactive_phase) is ct.composite.Interface:
        concentrations = surface_coverages_vs_time(reactive_phase, t, returnNDDataset) * reactive_phase.site_density
        if returnNDDataset:
            concentrations.x.title = 'concentration'
        return concentrations

    else:
        raise NotImplementedError('not implmented for reactive_phase={}'.format(str(type(reactive_phase))))

def reactors_composition_vs_time(reactors, t, surfaces=None, returnNDDataset=False):
    ''' Returns the gas molar fractions - and optionally the surfaces coverages vs. time in a single reactor
    or a list of connected reactors

    parameters:
    -----------
    gasreactors: cantera._cantera.IdealGasReactor or list of cantera._cantera.IdealGasReactor
        reactor(s) in which the mole fractions are computed
    t: iterable or spectrochempy.Coord
        times at which the coverages must be computed
    surfaces: instance of cantera._cantera.ReactorSurface or list a of such instances, optional
        reacting surface(s) in contact with the reactors. If not None, all gas reactors are expected to be
        in contact with a reacting surface of the same type.
    return_NDDataset: boolean, optional
        if True returns the mole fraction matrix (and coverage matrix if surfaces are includes) as NDDataset
        else as np.ndarray
        default: False (returns ndarray(s)

    returns:
    --------
    gas_compositions: ndarray or NDDataset
        mole fractions in reactor(s) at times t
    coverages: ndarray or NDDataset
        surface coverages in reactor(s) at times t
    '''

    # check input
    if isinstance(reactors, ct._cantera.IdealGasReactor):
        reactors = [reactors]
    elif isinstance(reactors, Iterable):
        for r in reactors:
            if not isinstance(r, ct._cantera.IdealGasReactor):
                raise TypeError("reactors should contain only IdealGasReactor elements")
    else:
        raise TypeError("reactors should be either a single IdealGasReactor or a list of IdealGasReactor(s)")

    if surfaces is not None:
        if isinstance(surfaces, ct._cantera.ReactorSurface):
            surfaces = [surfaces]
        elif isinstance(surfaces, Iterable):
            for s in surfaces:
                if not isinstance(s, ct._cantera.ReactorSurface):
                    raise TypeError("surfaces should contain only ReactorSurface elements")
        else:
            raise TypeError("surfaces should be either a single ReactorSurface or a list of ReactorSurface(s)")

        if len(surfaces) != len(reactors):
            raise IOError("all reactors should ")

    if type(t) is Coord:
        t = t.data

    if len(reactors)==1:
        gas_compositions  = np.zeros((len(t), reactors[0].Y.size))
        if surfaces is not None:
            surf_coverages = np.zeros((len(t), surfaces[0].coverages.size))
    else:
        gas_compositions = np.zeros((len(reactors), len(t), reactors[0].Y.size))
        if surfaces is not None:
            surf_coverages = np.zeros((len(reactors), len(t), surfaces[0].coverages.size))

    rnet = ct.ReactorNet(reactors)

    for i, ti in enumerate(t):
        rnet.advance(ti)
        if len(reactors) > 1:
            for j, r in enumerate(reactors):
                gas_compositions[j, i, :] = reactors[j].thermo.X[:]
                if surfaces is not None:
                    surf_coverages[j, i, :] = surfaces[j].coverages[:]
        else:
            gas_compositions[i, :] = reactors[0].thermo.X[:]
            if surfaces is not None:
                surf_coverages[i, :] = surfaces[0].coverages[:]


    if returnNDDataset:
        gas_compositions = NDDataset(gas_compositions)
        gas_compositions.title = 'mol fraction'
        gas_compositions.y = Coord(t, title='time')
        gas_compositions.x.title = 'species'
        gas_compositions.x.labels = reactors[0].kinetics.species_names
        if len(gas_compositions.dims) == 3:
            gas_compositions.z.labels = [r.name for r in reactors]

        if surfaces is not None:
            surf_coverages = NDDataset(surf_coverages)
            surf_coverages.title = 'coverage'
            surf_coverages.y = Coord(t, title='time')
            surf_coverages.x.title = 'species'
            surf_coverages.x.labels = surfaces[0].kinetics.species_names
            if len(surf_coverages.dims) == 3:
                surf_coverages.z.labels = [r.name for r in reactors]

    if surfaces is None:
        return gas_compositions
    else:
        return gas_compositions, surf_coverages


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
        except ValueError:
            print('class {} has no \'{}\' attribute'.format(type(reactive_phase), param))
            raise

        # if exists => sets its new value
        # if the attribute is writable:
        if param in ('site_density', 'coverages', 'concentrations'):
            exec('reactive_phase.' + param + '=' + str(param_value[i]))
        # else use Cantera methods (or derived from cantera)
        elif param.split('.')[-1] == 'pre_exponential_factor':
            str_rate = 'reactive_phase.' + '.'.join(param.split('.')[-3:-1])
            b, E = eval(str_rate + '.temperature_exponent,' + str_rate + '.activation_energy ')
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
    Fits rate parameters and concentrations to a given concentration profile.

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
        Chat = surface_concentrations_vs_time(surface, C.y)
        return np.sum(np.square(C.data[:, externalConc] - Chat[:, external_to_C_idx]))

    method = kwargs.get("method", "Nelder-Mead")
    bounds = kwargs.get("bounds", None)
    tol = kwargs.get("tol", None)
    options = kwargs.get("options", {'disp': True})
    if options['disp']:
        print('Optimization of the parameters.')
        print('         Initial parameters: {}'.format(guess_param))
        print('         Initial function value: {}'.format(objective(guess_param, param_to_optimize, C, externalConc,
                                                                     external_to_C_idx, reactive_phase)))
    tic = datetime.datetime.now()
    res = minimize(objective, guess_param, args=(param_to_optimize, C, externalConc, external_to_C_idx, reactive_phase),
                   method=method, bounds=bounds, tol=tol, options=options)
    toc = datetime.datetime.now()
    guess_param = res.x
    if options['disp']:
        print('         Optimization time: {}'.format((toc - tic)))
        print('         Final parameters: {}'.format(guess_param))
    Ckin = surface_concentrations_vs_time(reactive_phase, C.y, returnNDDataset=True)
    newargs = (reactive_phase, param_to_optimize, guess_param)
    return {'concentrations': Ckin,
            'results': res,
            'new_args': newargs}

def PFR(Vcstr, initial_atmosphere, inlet_gas, inlet_flow, surface=None, areaCstr=None):
    r"""creates a PFR reactor as a network of CSTRs in series

    parameters
    ----------
    Vcstr: iterable
        contains the volumes of the CSTRs


    """


    # create the inlet gas reservoir and set the entry profile, to be connected to the first reactor with a Mass Flow Controller.

    inlet = ct.Reservoir(contents=inlet_gas, name='inlet')
    Ncstr = len(Vcstr)
    cstr = []
    if surface is not None:
        cstrSurf = []

    for i in range(Ncstr):
        # create reactor i
        cstr.append(ct.IdealGasReactor(initial_atmosphere, name="R_{}".format(i), energy='off'))
        # sets the volume of the cstr reactor
        cstr[i].volume = Vcstr[i]
        if surface is not None:
            # add reactive surface
            cstrSurf.append(ct.ReactorSurface(kin=surface, r=cstr[i], A=areaCstr[i]))
        # make the connection with the previous element:
        if i == 0:  # this is the first reactor,
            # connect it to the upstream reservoir by a MFC:
            if callable(inlet_flow):
                mfc = ct.MassFlowController(upstream=inlet, downstream=cstr[i],
                                        mdot= lambda t: inlet_flow(t) * inlet_gas.density)
            else:
                mfc = ct.MassFlowController(upstream=inlet, downstream=cstr[i],
                                            mdot= inlet_flow * inlet_gas.density)

        else:  # this reactor i is connected the previous one (i-1) through a pressure controller:
            ct.PressureController(upstream=cstr[i - 1], downstream=cstr[i], master=mfc, K=1e-5)

    # create a final reservoir for the exhaust of the last reactor. The composition of this reservoir is irrelevant.
    event = ct.Reservoir(initial_atmosphere, name='event')
    ct.PressureController(upstream=cstr[-1], downstream=event, master=mfc, K=1e-5)

    if surface is None:
        return cstr
    else:
        return cstr, cstrSurf