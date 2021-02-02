# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# Description
# Utility functions to deal with Cantera input/output
#
# ---------------------------------------------------------------------------------

__all__ = ['PFR', 'fit_to_gas_concentrations',
           'surface_coverages_vs_time', 'fit_to_surface_concentrations', 'modify_rate',
           'modify_surface_kinetics']

import cantera as ct
import datetime
import numpy as np
from scipy.optimize import minimize, least_squares
from collections.abc import Iterable


from spectrochempy.core.dataset.nddataset import NDDataset, Coord




class PFR():
    '''implements a PFR reactor with inlets, a set CSTR in series and evebnt'''

    def __init__(self, cti_file, init_X, inlet_X, inlet_F, volume, n_cstr=0, area=None, P=ct.one_atm, T=298,
                  K=1e-5, kin_param_to_set=None):
        '''
        parameters:
        cti_file: str,
            the  file must contain a gas phase named 'gas' and optionally a reactive surface named 'surface'
        init_X: dict, array or list of them
            initial composition of the reactors
        inlet_X: dict, array or list of them
            composition of the inlet reservoirs
        inlet_F: number, callable or list iof them
            inlet flow rates from the reservoirs to the reactor
        volume: float or iterable (e.g. list or array) of floats
            if float: the total volume of the PFR, requires defining n_cstr, the number of cstrs (see below). The volume of
            each cstr will the be given by volume/cstr
            if iterable: the volumes of the cstrs.
        n_cstr: int,; optional
            must be set when only the total volume of the PFR is given
        area: otpional, None, float or iterable (e.g. list or array) of floats. Default: None
            if None: no reactive surface is added to the reactor
            if float: the total area of the reactive surface. The surface in each cstr will the be given by area/n_cstr
            if iterable: the surfrace areas of the reactove surfaces in each of the cstrs.
        P: float, optional, default: one atmosphere
            initial pressure of the PFR, the inlet reservoirs and event
        T: float, otpional, default: 298 K
            initial temperature of the PFR, the inlet reservoirs and event
        K: float, default 1e-5
           Pressure coefficient, set the constant proportionality constant between pressure drop and mass flow rate
           through Pressure controllers (see Cantera documentation)
    '''

        def modify_rate(reactive_phase, i_reaction, rate):
            """ Changes one of the rates  of a cantera reactive phase
                """
            rxn = reactive_phase.reaction(i_reaction)
            rxn.rate = rate
            reactive_phase.modify_reaction(i_reaction, rxn)
            return reactive_phase

        def modify_surface_kinetics(surface, param_to_set):
            """changes a set of numerical parameters of a an Interface among following:
            site_density, coverages, concentrations,
            pre-exponential factor, temperature_exponent, activation_energy
            """
            # check some parameters
            if type(surface) is not ct.composite.Interface:
                raise ValueError('only implemented of ct.composite.Interface')

            for param in param_to_set:
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
                    exec('surface.' + param + '=' + str(param_to_set[param]))
                    if param == 'site_density':
                        # coverages must be reset
                        surface.coverages = init_coverages

                # else use Cantera methods (or derived from cantera)
                elif param.split('.')[-1] == 'pre_exponential_factor':
                    str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
                    b, E = eval(str_rate + '.temperature_exponent,' + str_rate + '.activation_energy ')
                    rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
                    modify_rate(surface, rxn, ct.Arrhenius(param_to_set[param], b, E))

                elif param.split('.')[-1] == 'temperature_exponent':
                    str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
                    A, E = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.activation_energy ')
                    rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
                    modify_rate(surface, rxn, ct.Arrhenius(A, param_to_set[param], E))

                elif param.split('.')[-1] == 'activation_energy':
                    str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
                    A, b = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.temperature_exponent')
                    rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
                    modify_rate(surface, rxn, ct.Arrhenius(A, b, param_to_set[param]))
            return

        if area is None:
            add_surface = False
        else:
            add_surface = True
        # copy inlet parameters (for copy)
        self._cti = cti_file
        self._init_X = init_X
        self._inlet_X = inlet_X
        self._inlet_F = inlet_F
        self._volume = volume
        self.T = T
        self.P = P
        self._area = area
        self._K = K
        self._kin_param_to_set = kin_param_to_set

        self.cstr = []      #  list of cstrs
        self.surface = []   # reactor surfaces of cstr's
        self._mfc = [] # mass flow
        self.inlet = []  # reservoirs
        self.event = None
        self._pc = [] # pressure controllers

        if isinstance(self._volume, (float, int)):
            self._volume = self._volume * np.ones((n_cstr))/n_cstr
        if add_surface and isinstance(area, (float, int)):
            self._area = self.area * np.ones((n_cstr))/n_cstr
        self.n_cstr = len(volume)
        # first cstr
        initial_gas = ct.Solution(self._cti, 'gas')
        initial_gas.TPX = self.T, self.P, init_X
        self.n_gas_species = len(initial_gas.X)
        self.cstr.append(ct.IdealGasReactor(initial_gas, name="R_0", energy='off'))
        self.cstr[0].volume = volume[0]
        if add_surface:
            surface = ct.Interface(self._cti, phaseid='surface', phases=[initial_gas])
            if kin_param_to_set is not None:
                modify_surface_kinetics(surface, kin_param_to_set)
            self.n_surface_species = len(surface.X)
            self.surface.append(ct.ReactorSurface(kin=surface, r=self.cstr[0], A=area[0]))

        # create and connect inlets to R_0
        if not isinstance(inlet_X, Iterable):
            inlet_X = [inlet_X]
            inlet_F = [inlet_F]

        self._inlet_F = inlet_F

        for i, (X, F) in enumerate(zip(inlet_X, self._inlet_F)):
            inlet_gas = ct.Solution(self._cti, 'gas')
            inlet_gas.TPX = self.T, self.P, X
            self.inlet.append(ct.Reservoir(contents=inlet_gas, name=f'inlet_{i}'))
            self._mfc.append(ct.MassFlowController(self.inlet[-1], self.cstr[0], name=f'MFC_{i}'))
            if not callable(F):
                self._mfc[-1].set_mass_flow_rate(F * inlet_gas.density)
            else:
                # it is tricky to pass non explicit lambda functions to MassFlowControllers
                # the following works while use of self._inlet_F[i](t) generate an error
                # when using reactorNet.advance()
                if i == 0:
                    self._mfc[-1].set_mass_flow_rate(lambda t: self._inlet_F[0](t) * inlet_gas.density)
                elif i == 1:
                    self._mfc[-1].set_mass_flow_rate(lambda t: self._inlet_F[1](t) * inlet_gas.density)
                elif i == 2:
                    self._mfc[-1].set_mass_flow_rate(lambda t: self._inlet_F[2](t) * inlet_gas.density)
                elif i == 3:
                    self._mfc[-1].set_mass_flow_rate(lambda t: self._inlet_F[3](t) * inlet_gas.density)
                elif i == 4:
                    self._mfc[-1].set_mass_flow_rate(lambda t: self._inlet_F[4](t) * inlet_gas.density)
                else:
                    raise ValueError("variable flow rate(s) must be associated within the first"
                                     "five MFC(s)")
        # create other cstrs and link them with the previous one throupd a pressure controller
        for i in range(1,len(volume)):
            initial_gas = ct.Solution(self._cti, 'gas')
            initial_gas.TPX = self.T, self.P, init_X
            self.cstr.append(ct.IdealGasReactor(initial_gas, name="R_0", energy='off'))
            self.cstr[i].volume = volume[i]
            if add_surface:
                surface = ct.Interface(self._cti, phaseid='surface', phases=[initial_gas])
                self.n_surface_species = len(surface.X)
                if kin_param_to_set is not None:
                    modify_surface_kinetics(surface, kin_param_to_set)
                self.surface.append(ct.ReactorSurface(kin=surface, r=self.cstr[i], A=area[i]))

            self._pc.append(ct.PressureController(self.cstr[i-1], self.cstr[i],
                                                  master=self._mfc[-1], K=K))

        # create the event
        event_gas = ct.Solution(self._cti, 'gas')
        event_gas.TPX = self.T, self.P, init_X
        self.event = ct.Reservoir(contents=event_gas, name=f'event')
        self._pc.append(ct.PressureController(self.cstr[-1], self.event,
                                              master=self._mfc[-1], K=K))
        self.X = np.ones((self.n_cstr, self.n_gas_species))
        self.coverages = np.ones((self.n_cstr, self.n_surface_species))
        for i, (r, s) in enumerate(zip(self.cstr, self.surface)):
            self.X[i,:] = r.thermo.X
            self.coverages[i, :] = s.coverages

        self.net = ct.ReactorNet(self.cstr)


    # properties
    @property
    def time(self):
        return self.net.time
    # private methods


    # public methods

    def advance(self, time):
        self.net.advance(time)
        for i, (r, s) in enumerate(zip(self.cstr, self.surface)):
            self.X[i, :] = r.thermo.X
            self.coverages[i, :] = s.coverages

    def composition_vs_time(self, time, returnNDDataset=True):
        if isinstance(time, Coord):
            time = time.data
        X = np.zeros((len(time), self.n_cstr,  self.n_gas_species))
        coverages = np.zeros((len(time), self.n_cstr,  self.n_surface_species))

        for i, t in enumerate(time):
            self.advance(t)
            X[i] = self.X
            coverages[i] = self.coverages

        if returnNDDataset:
            X = NDDataset(X)
            X.title = 'mol fraction'
            X.z = Coord(time, title='time')
            X.y.labels = [r.name for r in self.cstr]
            X.x.title = 'species'
            X.x.labels = self.cstr[0].kinetics.species_names


            coverages = NDDataset(coverages)
            coverages.title = 'coverage'
            coverages.z = Coord(time, title='time')
            coverages.x.title = 'species'
            coverages.x.labels = self.surface[0].kinetics.species_names
            coverages.y.title = 'reactor'
            coverages.y.labels = [r.name for r in self.cstr]

        return {'X': X,
                'coverages': coverages}



def fit_to_gas_concentrations(pfr, exp_conc, exp_idx, fit_to_exp_idx,
                                  param_to_optimize, other_param=None, **kwargs):
    r"""
    Function fitting rate parameters and concentrations to a given concentration profile.

    Parameters
    ------------
    pfr: a PFR instance
    exp_conc: NDDataset
        experimental concentration profiles on which to fit the model. Can contain more concentration
        profiles than those to fit. the y Coord should be time
    exp_idx:
        indexes of experimental concentration profiles on which the model will be fitted
    fit_to_exp_idx:
        correspondence between optimized concentration profile and experimental
        concentration profile
    param_to_optimize: dict
        reactive phase parameters to optimize
        param_to_set: dict
        names of kinetic parameters differing from the cti file but fixed during optimization

    **kwargs:
        parameters for the optimization (see scipy.optimize.minimize)

    Returns
    ----------
    a dictionary

    """

    global it

    def objective(guess, param_to_optimize,
                  exp_conc, exp_idx, fit_to_exp_idx,
                  pfr, optimizer,
                  **kwargs):

        global it
        it = it + 1

        for i, param in enumerate(param_to_optimize):
            param_to_optimize[param] = guess[i]

        # create reactor
        if pfr._kin_param_to_set is not None:
            all_param = {**pfr._kin_param_to_set, **param_to_optimize}
        else:
            all_param = param_to_optimize

        newpfr = PFR(pfr._cti, pfr._init_X, pfr._inlet_X, pfr._inlet_F, pfr._volume,
                   P=pfr.P, T=pfr.T, area=pfr._area, kin_param_to_set=all_param)

        fitted_concentrations = newpfr.composition_vs_time(exp_conc.z,
                                                            returnNDDataset=False)['X'][:, -1, :].squeeze()

        se = np.square(exp_conc.data[:, exp_idx] - fitted_concentrations[:, fit_to_exp_idx]).flatten()
        sse = np.sum(se)

        if options['disp']:
            print(f'         Evaluation # {it} | Current function value: {sse} \r', end="")

        if optimizer == 'minimize':
            return sse
        elif optimizer == 'least_squares':
            return se

    method = kwargs.get("method", "Nelder-Mead")
    bounds = kwargs.get("bounds", None)
    tol = kwargs.get("tol", None)
    options = kwargs.get("options", {'disp': True})

    if method in ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "SLSQP",
                  "trust-constr", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"]:
        optimizer = 'minimize'

    elif method in ['trf', 'dogbox', 'lm']:
        optimizer = 'least_squares'
        if bounds is None:
            bounds = (-np.inf, np.inf)

    initial_guess = np.zeros((len(param_to_optimize)))
    for i, param in enumerate(param_to_optimize):
        initial_guess[i] = param_to_optimize[param]
    it = -1

    init_function_value = objective(initial_guess, param_to_optimize,
                                    exp_conc, exp_idx, fit_to_exp_idx,
                                    pfr, optimizer)
    if optimizer == 'least-squares':
        init_function_value = np.sum(init_function_value)

    if options['disp']:
        print('Optimization of the parameters.')
        print(f'         Initial parameters: {initial_guess}')
        print(f'         Initial function value: {init_function_value}')

    tic = datetime.datetime.now()

    if optimizer == 'minimize':
        res = minimize(objective, initial_guess,
                       args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, pfr, optimizer),
                       method=method, bounds=bounds, tol=tol, options=options)

    elif optimizer == 'least_squares':
        res = least_squares(objective, initial_guess,
                            args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, pfr, optimizer),
                            method=method, bounds=bounds)

    toc = datetime.datetime.now()
    optim_param = res.x
    if options['disp']:
        print('         Optimization time: {}'.format((toc - tic)))
        print('         Final parameters: {}'.format(optim_param))

    if other_param is not None:
        all_param = {**other_param, **param_to_optimize}
    else:
        all_param = param_to_optimize

    newpfr = PFR(pfr._cti, pfr._init_X, pfr._inlet_X, pfr._inlet_F, pfr._volume,
                 P=pfr.P, T=pfr.T, area=pfr._area, kin_param_to_set=all_param)

    fitted_concentrations = newpfr.composition_vs_time(exp_conc.z)['X'][:, -1, :].squeeze()

    newargs = (pfr, all_param)

    return {'fitted_concentrations': fitted_concentrations,
            'results': res,
            'newargs': newargs}


#####


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




def modify_rate(reactive_phase, i_reaction, rate):
    """ Changes one of the rates  of a cantera reactive phase.
        """
    rxn = reactive_phase.reaction(i_reaction)
    rxn.rate = rate
    reactive_phase.modify_reaction(i_reaction, rxn)
    return reactive_phase


def modify_surface_kinetics(surface, param_to_set):
    """changes a set of numerical parameters of a an Interface among following:
    site_density, coverages, concentrations,
    pre-exponential factor, temperature_exponent, activation_energy
    """
    # check some parameters
    if type(surface) is not ct.composite.Interface:
        raise ValueError('only implemented of ct.composite.Interface')

    for param in param_to_set:
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
            exec('surface.' + param + '=' + str(param_to_set[param]))
            if param == 'site_density':
                # coverages must be reset
                surface.coverages = init_coverages

        # else use Cantera methods (or derived from cantera)
        elif param.split('.')[-1] == 'pre_exponential_factor':
            str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
            b, E = eval(str_rate + '.temperature_exponent,' + str_rate + '.activation_energy ')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(surface, rxn, ct.Arrhenius(param_to_set[param], b, E))

        elif param.split('.')[-1] == 'temperature_exponent':
            str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
            A, E = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.activation_energy ')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(surface, rxn, ct.Arrhenius(A, param_to_set[param], E))

        elif param.split('.')[-1] == 'activation_energy':
            str_rate = 'surface.' + '.'.join(param.split('.')[-3:-1])
            A, b = eval(str_rate + 'pre_exponential_factor,' + str_rate + '.temperature_exponent')
            rxn = int(param.split('.')[0].split('[')[-1].split(']')[0])
            modify_rate(surface, rxn, ct.Arrhenius(A, b, param_to_set[param]))
    return


def fit_to_surface_concentrations(exp_conc, exp_idx, fit_to_exp_idx, surface, param_to_optimize, **kwargs):
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
        **kwargs:
        parameters for the optimization (see scipy.optimize.minimize)

    Returns
    ----------
    a dictionary

    """

    def objective(guess_value, param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, surface):
        modify_surface_kinetics(surface, param_to_optimize)
        Chat = surface_concentrations_vs_time(surface, exp_conc.y)
        return np.sum(np.square(exp_conc.data[:, exp_idx] - Chat[:, fit_to_exp_idx]))

    method = kwargs.get("method", "Nelder-Mead")
    bounds = kwargs.get("bounds", None)
    tol = kwargs.get("tol", None)
    options = kwargs.get("options", {'disp': True})
    initial_guess = np.zeros((len(param_to_optimize)))
    for i, param in enumerate(param_to_optimize):
        initial_guess[i] = param_to_optimize[param]

    if options['disp']:
        print('Optimization of the parameters.')
        print('         Initial parameters: {}'.format(initial_guess))
        print('         Initial function value: {}'.format(objective(initial_guess, param_to_optimize, exp_conc, exp_idx,
                                                                     fit_to_exp_idx, surface)))


    tic = datetime.datetime.now()

    if method in ["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "SLSQP",
                  "trust-constr", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"]:
        res = minimize(objective, initial_guess, args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, surface),
                       method=method, bounds=bounds, tol=tol, options=options)
    elif method in ['trf', 'dogbox', 'lm']:
        res = least_squares(objective, initial_guess, args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, surface),
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

