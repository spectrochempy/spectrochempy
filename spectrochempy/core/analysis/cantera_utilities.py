# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Description
# Utility functions to deal with Cantera input/output
#
# ---------------------------------------------------------------------------------

__all__ = ['surface_coverages_vs_time', 'surface_concentrations_vs_time', 'modify_rate',
           'reactors_composition_vs_time', 'modify_surface_kinetics', 'fit_to_surface_concentrations',
           'fit_to_reactor_concentrations', 'PFR']

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


def surface_concentrations_vs_time(surface, t, returnNDDataset=False):
    ''' Returns the  concentrations at time(s) t on a surface
    params:
    ------
    surface: instance of cantera.composite.Interface
    t: iterable or spectrochempy.Coord, times at which the concentrations must be computed
    return_NDDataset: boolean, if True returns the concentration matrix as a NDDataset, else as a np.ndarray
    default: False
    '''

    if type(surface) is ct.composite.Interface:
        concentrations = surface_coverages_vs_time(surface, t, returnNDDataset) * surface.site_density
        if returnNDDataset:
            concentrations.x.title = 'concentration'
        return concentrations

    else:
        raise NotImplementedError('not implemented for reactive_phase={}'.format(str(type(surface))))


def reactors_composition_vs_time(reactors, t, surfaces=None, returnNDDataset=False):
    ''' Returns the gas molar fractions - and optionally the surfaces coverages vs. time in a single reactor
    or a list of connected reactors

    parameters:
    -----------
    reactors: cantera._cantera.IdealGasReactor or list of cantera._cantera.IdealGasReactor
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
            raise IOError("the number of ReactorSurfaces should equal the number of reactors")

    if type(t) is Coord:
        t = t.data

    # set containers for gas composition and surface coverages of the reactors vs time
    if len(reactors) == 1:
        gas_compositions = np.zeros((len(t), reactors[0].Y.size))
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


def modify_surface_kinetics(surface, param_to_set, param_value):
    """changes a set of numerical parameters of a an Interface among following:
    site_density, coverages, concentrations,
    pre-exponential factor, temperature_exponent, activation_energy
    """
    # check some parameters
    if type(surface) is not ct.composite.Interface:
        raise ValueError('only implemented of ct.composite.Interface')
    if len(param_to_set) != len(param_value):
        raise ValueError('number of parameters to set and values should be equal')

    for i, param in enumerate(param_to_set):

        # check that  param_to_change exists
        try:
            eval('surface.' + param)
        except ValueError:
            print('class {} has no \'{}\' attribute'.format(type(surface), param))
            raise

        # if exists => sets its new value
        # if the attribute is writable:
        if param in ('site_density', 'coverages', 'concentrations'):
            init_coverages = surface.coverages
            exec('surface.' + param + '=' + str(param_value[i]))
            if param == 'site_density':
                # coverages must be reset
                surface.coverages = init_coverages

        # else use Cantera methods (or derived from cantera)
        elif param.split('.')[-1] == 'pre_exponential_factor':
            str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
            b, E = eval(str_rate + '.temperature_exponent,' + str_rate + '.activation_energy ')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(surface, rxn, ct.Arrhenius(param_value[i], b, E))

        elif param.split('.')[-1] == 'temperature_exponent':
            str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
            A, E = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.activation_energy ')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(surface, rxn, ct.Arrhenius(A, param_value[i], E))

        elif param.split('.')[-1] == 'activation_energy':
            str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
            A, b = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.temperature_exponent')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(surface, rxn, ct.Arrhenius(A, b, param_value[i]))
    return


def fit_to_surface_concentrations(exp_conc, exp_idx, fit_to_exp_idx, surface, param_to_optimize, guess_value, **kwargs):
    r"""
    Fits rate parameters to a given concentration profile.

    Parameters
    ------------

    exp_conc: NDDataset
        experimental concentration profiles on which to fit the model. C can contain more concentration
        profiles than those to fit.
    exp_idx:
        indexes of experimental concentration profiles on which the model will be fitted
    fit_to_exp_idx:
        correspondence between optimized (external) concentration profile and experimental
        concentration profile
    reactivePhase:
        cantera active phase. Currently implemented for surface only
    param_to_optimize: list
        list of reactive phase parameters to optomize
    guess_value:
        initial guess for the parameters to fit
    **kwargs:
        parameters for the optimization (see scipy.optimize.minimize)

    Returns
    ----------
    a dictionary

    """

    def objective(guess_value, param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, surface):
        modify_surface_kinetics(surface, param_to_optimize, guess_value)
        Chat = surface_concentrations_vs_time(surface, exp_conc.y)
        return np.sum(np.square(exp_conc.data[:, exp_idx] - Chat[:, fit_to_exp_idx]))

    method = kwargs.get("method", "Nelder-Mead")
    bounds = kwargs.get("bounds", None)
    tol = kwargs.get("tol", None)
    options = kwargs.get("options", {'disp': True})
    if options['disp']:
        print('Optimization of the parameters.')
        print('         Initial parameters: {}'.format(guess_value))
        print('         Initial function value: {}'.format(objective(guess_value, param_to_optimize, exp_conc, exp_idx,
                                                                     fit_to_exp_idx, surface)))
    tic = datetime.datetime.now()
    res = minimize(objective, guess_value, args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, surface),
                   method=method, bounds=bounds, tol=tol, options=options)
    toc = datetime.datetime.now()
    guess_value = res.x
    if options['disp']:
        print('         Optimization time: {}'.format((toc - tic)))
        print('         Final parameters: {}'.format(guess_value))
    Ckin = surface_concentrations_vs_time(surface, exp_conc.y, returnNDDataset=True)
    newargs = (surface, param_to_optimize, guess_value)
    return {'concentrations': Ckin,
            'results': res,
            'new_args': newargs}


def fit_to_reactor_concentrations(reactor, exp_conc, exp_idx, fit_to_exp_idx, param_to_optimize, guess_value,
                                  other_param=None, other_value=None, **kwargs):
    r"""
    Function fitting rate parameters and concentrations to a given concentration profile.

    Parameters
    ------------
    reactor: dict of reactor parameters
    exp_conc: NDDataset
        experimental concentration profiles on which to fit the model. Can contain more concentration
        profiles than those to fit. the y Coord should be time
    exp_idx:
        indexes of experimental concentration profiles on which the model will be fitted
    fit_to_exp_idx:
        correspondence between optimized concentration profile and experimental
        concentration profile
    param_to_optimize: list
        list of reactive phase parameters to optomize
    guess_value:
        initial guess for the parameters to fit
    param_to_set: list
        names of kinetic parameters differing from the cti file but fixed during optimization
    param_value: list
        values of the kinetic parameters listed in param_to_set

    **kwargs:
        parameters for the optimization (see scipy.optimize.minimize)

    Returns
    ----------
    a dictionary

    """

    global it

    def objective(guess_value, param_to_optimize,
                  exp_conc, exp_idx, fit_to_exp_idx,
                  reactor,
                  other_param, other_value,
                  **kwargs):

        global it
        it = it + 1

        # create reactor

        if other_param is not None:
            all_param_to_set = param_to_optimize + other_param
            all_param_value = list(guess_value) + list(other_value)
        else:
            all_param_to_set = param_to_optimize
            all_param_value = guess_value

        cstr, cstrsurfaces = PFR(reactor, kin_param_to_set=all_param_to_set, kin_param_value=all_param_value)

        fitted_concentrations = reactors_composition_vs_time(cstr, exp_conc.y, surfaces=None,
                                                             returnNDDataset=False)[-1, :, :]
        sse = np.sum(np.square(exp_conc.data[:, exp_idx] - fitted_concentrations[:, fit_to_exp_idx]))
        if options['disp']:
            print(f'         Evaluation # {it} | Current function value: {sse} \r', end="")
        return sse

    method = kwargs.get("method", "Nelder-Mead")
    bounds = kwargs.get("bounds", None)
    tol = kwargs.get("tol", None)
    options = kwargs.get("options", {'disp': True})
    it = -1
    init_function_value = objective(guess_value, param_to_optimize,
                                    exp_conc, exp_idx, fit_to_exp_idx,
                                    reactor, other_param, other_value)
    if options['disp']:
        print('Optimization of the parameters.')
        print(f'         Initial parameters: {guess_value}')
        print(f'         Initial function value: {init_function_value}')

    tic = datetime.datetime.now()
    # make PFR

    res = minimize(objective, guess_value,
                   args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, reactor, other_param, other_value),
                   method=method, bounds=bounds, tol=tol, options=options)
    toc = datetime.datetime.now()
    optim_param = res.x
    if options['disp']:
        print('         Optimization time: {}'.format((toc - tic)))
        print('         Final parameters: {}'.format(optim_param))

    if other_param is not None:
        all_param_to_set = param_to_optimize + other_param
        all_param_value = list(guess_value) + list(other_value)
    else:
        all_param_to_set = param_to_optimize
        all_param_value = optim_param

    cstr, cstrsurfaces = PFR(reactor,
                             kin_param_to_set=all_param_to_set,
                             kin_param_value=all_param_value)

    fitted_concentrations = reactors_composition_vs_time(cstr, exp_conc.y, surfaces=None, returnNDDataset=True)

    newargs = (reactor, all_param_to_set, all_param_value)

    return {'fitted_concentrations': fitted_concentrations,
            'results': res,
            'newargs': newargs}


def PFR(reactor, kin_param_to_set=None, kin_param_value=None):
    r"""creates a PFR reactor as a network of CSTRs in series

    parameters
    ----------
    reactor: dict
        minimal settings to create a PFR as a set cstr in series with:
        'cstr_volumes': iterable
                        the volumes of the cstr (the number of cstr is thus fixed by len(cstr_volumes))
        'cstr_surface_areas': iterable or number, optional
                        if not None, surface areas of reactive surfaces (the PFR is then catalytic reactor)
        'cti': cti filename (including path).
                         This file must contain a gas phase named 'gas' and - in the case of a catalytic reactor -
                         an interface named "surface"

        'initial_composition': array or dictionary,
                               initial composition of the cstrs, identical for all of them
        'inlet_composition':  array or dictionary,
                              composition of the inlet flow
        'inlet_flow': callable (lambda) or number
                    inlet flow rate
        'kin_param_to_set': list, optional
                            names of surface kinetic parameters which value should be changed
                            with respect to those of the cti file.
        'kin_param_value': list, optional
                            values af the corresponding parameters
         ''
    """
    initial_gas = ct.Solution(reactor['cti'], 'gas')
    initial_gas.X = reactor['initial_composition']
    inlet_gas = ct.Solution(reactor['cti'], 'gas')
    inlet_gas.X = reactor['inlet_composition']
    inlet_flow = reactor['inlet_flow']
    # gas for the event reservoir, its composition is arbitrarily set
    event_gas = ct.Solution(reactor['cti'], 'gas')
    event_gas.X = reactor['initial_composition']
    surface = ct.Interface(reactor['cti'], phaseid='surface', phases=[initial_gas])
    if kin_param_to_set is not None:
        modify_surface_kinetics(surface, kin_param_to_set, kin_param_value)

    cstr_volumes = reactor['cstr_volumes']
    cstr_areas = reactor['cstr_surface_areas']

    Ncstr = len(cstr_volumes)
    cstr = []
    if surface is not None:
        cstr_surfaces = []

    inlet_reservoir = ct.Reservoir(contents=inlet_gas, name='inlet')
    for i in range(Ncstr):
        # create reactor i
        cstr.append(ct.IdealGasReactor(initial_gas, name="R_{}".format(i), energy='off'))
        # sets the volume of the cstr reactor
        cstr[i].volume = cstr_volumes[i]
        if cstr_areas is not None:
            # add reactive surface
            cstr_surfaces.append(ct.ReactorSurface(kin=surface, r=cstr[i], A=cstr_areas[i]))
        # make the connection with the previous element:
        if i == 0:  # this is the first reactor,
            # connect it to the upstream reservoir by a MFC:
            if callable(inlet_flow):
                mfc = ct.MassFlowController(upstream=inlet_reservoir, downstream=cstr[i],
                                            mdot=lambda t: inlet_flow(t) * inlet_gas.density)
            else:
                mfc = ct.MassFlowController(upstream=inlet_reservoir, downstream=cstr[i],
                                            mdot=inlet_flow * inlet_gas.density)

        else:  # this reactor i is connected the previous one (i-1) through a pressure controller:
            ct.PressureController(upstream=cstr[i - 1], downstream=cstr[i], master=mfc, K=1e-5)

    # create a final reservoir for the exhaust of the last reactor. The composition of this reservoir is irrelevant.
    event = ct.Reservoir(event_gas, name='event')
    ct.PressureController(upstream=cstr[-1], downstream=event, master=mfc, K=1e-5)

    if surface is None:
        return cstr
    else:
        return cstr, cstr_surfaces
