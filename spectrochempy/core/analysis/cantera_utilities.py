# -*- coding: utf-8 -*-

#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
Utility functions to deal with Cantera input/output.
"""

# TODO: Testing !

import datetime
import numpy as np
import warnings
import logging
from collections.abc import Iterable

from scipy.optimize import minimize, differential_evolution, least_squares

from spectrochempy.optional import import_optional_dependency
from spectrochempy.core import error_
from spectrochempy.core.dataset.nddataset import NDDataset, Coord

__all__ = [
    "coverages_vs_time",
    "concentrations_vs_time",
    "modify_rate",
    "modify_surface_kinetics",
    "fit_to_concentrations",
    "PFR",
]

ct = import_optional_dependency("cantera", errors="ignore")


def _cantera_is_not_available():
    if ct is None:
        error_(
            "Missing optional dependency 'cantera'.  Use conda or pip to install cantera."
        )
    return ct is None


def coverages_vs_time(surface, t, returnNDDataset=False):
    """
    Returns the surface coverages at time(s) t.

    Parameters
    ----------
    surface: instance of cantera.composite.Interface.
    tim: iterable or spectrochempy.Coord.
        Times at which the coverages must be computed.
    returnNDDataset: boolean, default: False.
        If True returns the concentration matrix as a NDDataset, else as a np.ndarray.
    """

    init_coverages = surface.coverages
    coverages = np.zeros((len(t), surface.coverages.shape[0]))

    if isinstance(t, Coord):
        t = t.data

    for i, tim in enumerate(t):
        surface.coverages = init_coverages
        surface.advance_coverages(tim)
        coverages[i, :] = surface.coverages
        surface.coverages = init_coverages
    if returnNDDataset:
        coverages = NDDataset(coverages)
        coverages.y = Coord(t, title="time")
        coverages.x.title = "coverage / -"
        coverages.x.labels = surface.species_names
    return coverages


def concentrations_vs_time(reactive_phase, t, reactorNet=None, returnNDDataset=False):
    """
    Returns the  concentrations at time(s) t.

    Parameters
    ----------
    surface: instance of cantera.composite.Interface.
    tim: iterable or Coord.
        Times at which the concentrations must be computed.
    return_NDDataset: boolean, default: False.
        If True returns the concentration matrix as a NDDataset, else as a np.ndarray.
    """

    if _cantera_is_not_available():
        return

    if isinstance(reactive_phase, ct.composite.Interface):
        concentrations = (
            coverages_vs_time(reactive_phase, t, returnNDDataset)
            * reactive_phase.site_density
        )
        if returnNDDataset:
            concentrations.x.title = "concentration"
        return concentrations

    raise NotImplementedError(
        f"not implemented for reactive_phase={str(type(reactive_phase))}"
    )
    # # code for reactorNet
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
    """
    Modify the reaction rate with index i_reaction to have the same rate parameters as rate.

    Parameters
    ----------
    reactive_phase : instance of indeal_gas or cantera.composite.Interface
    i_reaction : index of reaction
    rate : the new rate expression or parameters

    Returns
    -------
    reactive_phase
    """

    rxn = reactive_phase.reaction(i_reaction)
    rxn.rate = rate
    reactive_phase.modify_reaction(i_reaction, rxn)
    return reactive_phase


def modify_surface_kinetics(surface, param_to_set):
    """
    Changes a set of numerical parameters of a an Interface among following:
    site_density, coverages, concentrations,
    pre-exponential factor, temperature_exponent, activation_energy.
    """

    if _cantera_is_not_available():
        return

    # check some parameters

    if not isinstance(surface, ct.composite.Interface):
        raise ValueError("only implemented of ct.composite.Interface")

    for param in param_to_set:
        # check that  param_to_change exists
        try:
            eval("surface." + param)
        except ValueError:
            print(f"class {type(surface)} has no '{param}' attribute")
            raise
        # if exists => sets its new value
        # if the attribute is writable:
        if param in ("site_density", "coverages", "concentrations"):
            init_coverages = surface.coverages
            exec("surface." + param + "=" + str(param_to_set[param]))
            if param == "site_density":
                # coverages must be reset
                surface.coverages = init_coverages

        # else use Cantera methods (or derived from cantera)
        elif param.split(".")[-1] == "pre_exponential_factor":
            str_rate = "surface." + ".".join(param.split(".")[-3:-1])
            b, E = eval(
                str_rate + ".temperature_exponent," + str_rate + ".activation_energy "
            )
            rxn = int(param.split(".")[0].split("[")[-1].split("]")[0])
            modify_rate(surface, rxn, ct.Arrhenius(param_to_set[param], b, E))

        elif param.split(".")[-1] == "temperature_exponent":
            str_rate = "surface." + ".".join(param.split(".")[-3:-1])
            A, E = eval(
                str_rate + "pre_exponential_factor," + str_rate + ".activation_energy "
            )
            rxn = int(param.split(".")[0].split("[")[-1].split("]")[0])
            modify_rate(surface, rxn, ct.Arrhenius(A, param_to_set[param], E))

        elif param.split(".")[-1] == "activation_energy":
            str_rate = "surface." + ".".join(param.split(".")[-3:-1])
            A, b = eval(
                str_rate
                + "pre_exponential_factor,"
                + str_rate
                + ".temperature_exponent"
            )
            rxn = int(param.split(".")[0].split("[")[-1].split("]")[0])
            modify_rate(surface, rxn, ct.Arrhenius(A, b, param_to_set[param]))


def fit_to_concentrations(
    C, externalConc, external_to_C_idx, reactive_phase, param_to_optimize, **kwargs
):
    """
    Function fitting rate parameters and concentrations to a given concentration profile.

    Parameters
    ------------

    C: NDDataset
        Experimental concentration profiles on which to fit the model. C can contain more concentration
        profiles than those to fit.
    externalConc:
        Indexes of experimental concentration profiles on which the model will be fitted
    external_to_C_idx:
        Correspondence between optimized (external) concentration profile and experimental
        concentration profile.
    reactivePhase:
        Cantera active phase. Currently implemented for surface only.
    param_to_optimize: dict
        Reactive phase parameters to optimize.

    **kwargs:
        Parameters for the optimization (see scipy.optimize.minimize).

    Returns
    ----------
    result
        A result dictionary.
    """

    def objective(
        param_value, param_to_optimize, C, externalConc, external_to_C_idx, surface
    ):
        modify_surface_kinetics(surface, param_to_optimize)
        Chat = concentrations_vs_time(surface, C.y)
        return np.sum(np.square(C.data[:, externalConc] - Chat[:, external_to_C_idx]))

    method = kwargs.get("method", "Nelder-Mead")
    bounds = kwargs.get("bounds", None)
    tol = kwargs.get("tol", None)
    options = kwargs.get("options", {"disp": True})

    guess_param = np.zeros((len(param_to_optimize)))
    for i, param in enumerate(param_to_optimize):
        guess_param[i] = param_to_optimize[param]

    if options["disp"]:
        print("Optimization of the parameters.")
        print(f"         Initial parameters: {guess_param}")
        print(
            f"         Initial function value: "
            f"{objective(guess_param, param_to_optimize, C, externalConc, external_to_C_idx, reactive_phase)}"
        )
    tic = datetime.datetime.now(datetime.timezone.utc)
    res = minimize(
        objective,
        guess_param,
        args=(param_to_optimize, C, externalConc, external_to_C_idx, reactive_phase),
        method=method,
        bounds=bounds,
        tol=tol,
        options=options,
    )
    toc = datetime.datetime.now(datetime.timezone.utc)
    final_param = res.x
    if options["disp"]:
        print(f"         Optimization time: {toc - tic}")
        print(f"         Final parameters: {final_param}")
    Ckin = concentrations_vs_time(reactive_phase, C.y, returnNDDataset=True)
    newargs = (reactive_phase, param_to_optimize, final_param)
    return {"concentrations": Ckin, "results": res, "new_args": newargs}


class PFR:
    """
    PFR reactor as a CSTR in series.
    """

    def __init__(
        self,
        cti_file,
        init_X,
        inlet_X,
        inlet_F,
        volume,
        n_cstr=0,
        P=None,  # ct.one_atm,
        T=298,
        area=None,
        K=1e-5,
        kin_param_to_set=None,
    ):
        """

        Parameters
        ==========
        cti_file: str,
            The cti file must contain a gas phase named 'gas' and optionally a reactive surface named 'surface'.

        init_X: dict, array or list of them
            Initial composition of the reactors.
        """
        if _cantera_is_not_available():
            raise ImportError

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
        self.P = P if P is not None else ct.one_atm
        self._area = area
        self._K = K
        self._kin_param_to_set = kin_param_to_set

        self.cstr = []  # list of cstrs
        self.surface = []  # reactor surfaces of cstr's
        self._mfc = []  # mass flow
        self.inlet = []  # reservoirs
        self.event = None
        self._pc = []  # pressure controllers

        if isinstance(self._volume, (float, int)):
            self._volume = self._volume * np.ones((n_cstr)) / n_cstr

        if add_surface and isinstance(area, (float, int)):
            self._area = self._area * np.ones((n_cstr)) / n_cstr
        self.n_cstr = len(volume)

        # first cstr
        initial_gas = ct.Solution(self._cti, "gas")
        initial_gas.TPX = self.T, self.P, init_X
        self.n_gas_species = len(initial_gas.X)
        self.cstr.append(ct.IdealGasReactor(initial_gas, name="R_0", energy="off"))
        self.cstr[0].volume = volume[0]

        if add_surface:
            surface = ct.Interface(self._cti, phaseid="surface", phases=[initial_gas])
            if kin_param_to_set is not None:
                modify_surface_kinetics(surface, kin_param_to_set)
            self.n_surface_species = len(surface.X)
            self.surface.append(
                ct.ReactorSurface(kin=surface, r=self.cstr[0], A=area[0])
            )

        # create and connect inlets to R_0
        if not isinstance(inlet_X, Iterable):
            inlet_X = [inlet_X]
            inlet_F = [inlet_F]

        self._inlet_F = inlet_F

        for i, (X, F) in enumerate(zip(inlet_X, self._inlet_F)):
            inlet_gas = ct.Solution(self._cti, "gas")
            inlet_gas.TPX = self.T, self.P, X
            self.inlet.append(ct.Reservoir(contents=inlet_gas, name=f"inlet_{i}"))
            self._mfc.append(
                ct.MassFlowController(self.inlet[-1], self.cstr[0], name=f"MFC_{i}")
            )

            if not callable(F):
                self._mfc[-1].set_mass_flow_rate(F * inlet_gas.density)
            else:
                # it is tricky to pass non explicit lambda functions to MassFlowControllers
                # the following works while use of self._inlet_F[i](t) generate an error
                # when using reactorNet.advance()

                if i == 0:
                    self._mfc[-1].set_mass_flow_rate(
                        lambda t: self._inlet_F[0](t) * inlet_gas.density
                    )
                elif i == 1:
                    self._mfc[-1].set_mass_flow_rate(
                        lambda t: self._inlet_F[1](t) * inlet_gas.density
                    )
                elif i == 2:
                    self._mfc[-1].set_mass_flow_rate(
                        lambda t: self._inlet_F[2](t) * inlet_gas.density
                    )
                elif i == 3:
                    self._mfc[-1].set_mass_flow_rate(
                        lambda t: self._inlet_F[3](t) * inlet_gas.density
                    )
                elif i == 4:
                    self._mfc[-1].set_mass_flow_rate(
                        lambda t: self._inlet_F[4](t) * inlet_gas.density
                    )

                else:
                    raise ValueError(
                        "variable flow rate(s) must be associated within the first"
                        "five MFC(s)"
                    )

        # create other cstrs and link them with the previous one through a pressure controller

        for i in range(1, len(volume)):
            initial_gas = ct.Solution(self._cti, "gas")
            initial_gas.TPX = self.T, self.P, init_X
            self.cstr.append(ct.IdealGasReactor(initial_gas, name="R_0", energy="off"))
            self.cstr[i].volume = volume[i]

            if add_surface:
                surface = ct.Interface(
                    self._cti, phaseid="surface", phases=[initial_gas]
                )
                self.n_surface_species = len(surface.X)
                if kin_param_to_set is not None:
                    modify_surface_kinetics(surface, kin_param_to_set)
                self.surface.append(
                    ct.ReactorSurface(kin=surface, r=self.cstr[i], A=area[i])
                )

            self._pc.append(
                ct.PressureController(
                    self.cstr[i - 1], self.cstr[i], master=self._mfc[-1], K=K
                )
            )

        # create the event

        event_gas = ct.Solution(self._cti, "gas")
        event_gas.TPX = self.T, self.P, init_X
        self.event = ct.Reservoir(contents=event_gas, name="event")
        self._pc.append(
            ct.PressureController(self.cstr[-1], self.event, master=self._mfc[-1], K=K)
        )

        self.X = np.ones((self.n_cstr, self.n_gas_species))
        self.coverages = np.ones((self.n_cstr, self.n_surface_species))

        for i, (r, s) in enumerate(zip(self.cstr, self.surface)):
            self.X[i, :] = r.thermo.X
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

        X = np.zeros((len(time), self.n_cstr, self.n_gas_species))
        coverages = np.zeros((len(time), self.n_cstr, self.n_surface_species))

        for i, t in enumerate(time):
            self.advance(t)
            X[i] = self.X
            coverages[i] = self.coverages

        if returnNDDataset:
            X = NDDataset(X)
            X.title = "mol fraction"

            X.z = Coord(time, title="time")
            X.y.labels = [r.name for r in self.cstr]
            X.x.title = "species"
            X.x.labels = self.cstr[0].kinetics.species_names

            coverages = NDDataset(coverages)
            coverages.title = "coverage"
            coverages.z = Coord(time, title="time")
            coverages.x.title = "species"
            coverages.x.labels = self.surface[0].kinetics.species_names
            coverages.y.title = "reactor"
            coverages.y.labels = [r.name for r in self.cstr]

        return {"X": X, "coverages": coverages}

    def fit_to_gas_concentrations(
        self,
        exp_conc,
        exp_idx,
        fit_to_exp_idx,
        param_to_optimize,
        param_to_set=None,
        logfile=None,
        **kwargs,
    ):
        r"""
         Function fitting rate parameters and concentrations to a given concentration profile at the outlet of the pfr.

         Parameters
        ------------

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

         logfile: None (default) or str
             name of the logfile

         **kwargs:
             parameters for the optimization (see scipy.optimize.minimize)

         Returns
         ----------
         a dictionary
        """
        # global variables to keep track of iterations and optimization history
        global it, trials, func_values, popsize, pop_sse, prev_min_sse

        it = -1  # current total number of function evaluation
        trials = []  # values of the parameters ti optimize
        func_values = []  # values of the objective functions
        popsize = None  # popsize for differential evolution

        start_time = datetime.datetime.now()

        if logfile:
            logging.basicConfig(
                filename=logfile, filemode="w", format="%(message)s", level=logging.INFO
            )

        def objective(
            guess,
            param_to_optimize,
            exp_conc,
            exp_idx,
            fit_to_exp_idx,
            optimizer,
            **kwargs,
        ):

            global it, trials, tic, pop_sse, prev_min_sse
            it = it + 1
            trials.append(guess)

            for i, param in enumerate(param_to_optimize):
                param_to_optimize[param] = guess[i]

            # create reactor
            if self._kin_param_to_set is not None:
                all_param = {**self._kin_param_to_set, **param_to_optimize}
            else:
                all_param = param_to_optimize

            newpfr = PFR(
                self._cti,
                self._init_X,
                self._inlet_X,
                self._inlet_F,
                self._volume,
                P=self.P,
                T=self.T,
                area=self._area,
                kin_param_to_set=all_param,
            )

            try:
                fitted_concentrations = newpfr.composition_vs_time(
                    exp_conc.z, returnNDDataset=False
                )["X"][:, -1, :].squeeze()
            except ct.CanteraError:
                if optimizer == "differential_evolution":
                    integrationError = True
                    warnings.warn(
                        "model could not be integrated with these parameters. Objective function set to Inf",
                        UserWarning,
                    )
                else:
                    raise

            if "integrationError" not in locals():
                se = np.square(
                    exp_conc.data[:, exp_idx] - fitted_concentrations[:, fit_to_exp_idx]
                ).flatten()
                sse = np.sum(se)
            else:
                sse = np.Inf

            if logfile:
                if popsize:
                    pop_sse.append(sse)
                    if not it % (popsize * len(param_to_optimize)):
                        toc = datetime.datetime.now()
                        gen = it // (popsize * len(param_to_optimize))
                        if gen > 0:
                            logging.info(
                                "--------"
                                + 10 * len(param_to_optimize) * "-"
                                + "--------------"
                            )
                            min_sse = min(pop_sse)
                            it_min_sse = (
                                it
                                - popsize * len(param_to_optimize)
                                + np.argmin(pop_sse)
                                + 1
                            )
                            if gen == 1:
                                logging.info(
                                    f"                      Minimum SSE: {min_sse:.3e} (Eval # {it_min_sse})"
                                )
                            else:
                                logging.info(
                                    f"                      Minimum SSE: {min_sse:.3e} "
                                    f"({(min_sse - prev_min_sse) / prev_min_sse:+.3%}, Eval # {it_min_sse})"
                                )
                            logging.info(
                                f"Execution time for the population: {toc - tic}"
                            )
                            logging.info(
                                f"             Total execution time: {toc - start_time}"
                            )
                            logging.info(" ")
                            prev_min_sse = min_sse
                            pop_sse = []

                        tic = datetime.datetime.now()
                        logging.info(f"{tic}: Start calculation of population #{gen}")
                        logging.info(
                            "--------"
                            + 12 * len(param_to_optimize) * "-"
                            + "--------------"
                        )
                        logging.info(
                            "Eval # | Parameters"
                            + (12 * len(param_to_optimize) - 11) * " "
                            + "  | SSE "
                        )
                        logging.info(
                            "-------|"
                            + 12 * len(param_to_optimize) * "-"
                            + "--|-----------"
                        )
                        pop_sse = []

                guess_string = ""
                for val in guess:
                    guess_string += f"{val:.5e} "
                logging.info(f"{it:6} | {guess_string} | {sse:.3e} ")

            if options["disp"]:
                print(
                    f"         Evaluation # {it} | Current function value: {sse} \r",
                    end="",
                )

            if optimizer in ["minimize", "differential_evolution"]:
                func_values.append(sse)
                return sse

            if optimizer == "least_squares":
                func_values.append(se)
                return se

        method = kwargs.get("method", "Nelder-Mead")
        bounds = kwargs.get("bounds", None)
        tol = kwargs.get("tol", None)
        options = kwargs.get("options", {"disp": True})

        if method in [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
        ]:
            optimizer = "minimize"

        elif method in ["trf", "dogbox", "lm"]:
            optimizer = "least_squares"
            if bounds is None:
                bounds = (-np.inf, np.inf)

        elif method == "differential_evolution":
            optimizer = "differential_evolution"
            # then param_to_optimize are expected to be bounds for each varaible

        if optimizer in ["minimize", "least_squares"]:
            initial_guess = np.zeros((len(param_to_optimize)))
            for i, param in enumerate(param_to_optimize):
                initial_guess[i] = param_to_optimize[param]

        else:  # optimizer is 'differential_evolution'
            initial_guess = []
            for param in param_to_optimize:
                initial_guess.append(param_to_optimize[param])
                bounds = initial_guess

        if optimizer in ["minimize", "least_squares"]:
            init_function_value = objective(
                initial_guess,
                param_to_optimize,
                exp_conc,
                exp_idx,
                fit_to_exp_idx,
                optimizer,
            )

        if logfile:
            logging.info("*** Cantera/Spectrochempy kinetic model optimization log ***")
            logging.info(
                f"{datetime.datetime.now()}: Starting optimization of the parameters"
            )
            logging.info("   Parameters to optimize:")
            for param in param_to_optimize:
                logging.info(f"      {param}: {param_to_optimize[param]}")
            logging.info(f"   Optimization Method: {method}")
            logging.info(" ")

        if options["disp"]:
            print("Optimization of the parameters.")
            print(f"         Method: {method}")
            print(f"         Initial parameters: {initial_guess}")
            if optimizer in ["minimize", "least_squares"]:
                print(f"         Initial function value: {init_function_value}")

        # tic = datetime.datetime.now()

        if optimizer == "minimize":
            res = minimize(
                objective,
                initial_guess,
                args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, optimizer),
                method=method,
                bounds=bounds,
                tol=tol,
                options=options,
            )

        elif optimizer == "least_squares":
            res = least_squares(
                objective,
                initial_guess,
                args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, optimizer),
                method=method,
                bounds=bounds,
            )

        elif optimizer == "differential_evolution":
            # set optional parameters / default set to scipy defauls
            if "strategy" in options:
                strategy = options["strategy"]
            else:
                strategy = "best1bin"
            if "maxiter" in options:
                maxiter = options["maxiter"]
            else:
                maxiter = 1000
            if "popsize" in options:
                popsize = options["popsize"]
            else:
                popsize = 15
            if "tol" in options:
                tol = options["tol"]
            else:
                tol = 0.01
            if "mutation" in options:
                mutation = options["mutation"]
            else:
                mutation = 0.5, 1
            if "recombination" in options:
                recombination = options["recombination"]
            else:
                recombination = 0.7
            if "seed" in options:
                seed = options["seed"]
            else:
                seed = None
            if "callback" in options:
                callback = options["callback"]
            else:
                callback = None
            if "polish" in options:
                polish = options["polish"]
            else:
                polish = True
            if "init" in options:
                init = options["init"]
            else:
                init = "latinhypercube"
            if "atol" in options:
                atol = options["atol"]
            else:
                atol = 0
            if "updating" in options:
                updating = options["updating"]
            else:
                updating = "immediate"
            if "workers" in options:
                workers = options["workers"]
                if workers != 1:
                    warnings.warn(
                        "parallelization not implemented yet, workers reset to 1",
                        UserWarning,
                    )
                    workers = 1
            else:
                workers = 1
            if "constraints" in options:
                constraints = options["constraints"]
            else:
                constraints = ()

            pop_sse = []

            res = differential_evolution(
                objective,
                bounds,
                args=(param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx, optimizer),
                strategy=strategy,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                mutation=mutation,
                recombination=recombination,
                seed=seed,
                callback=callback,
                polish=polish,
                init=init,
                atol=atol,
                updating=updating,
                workers=workers,
                constraints=constraints,
            )

            # note: to make it parallel (WIP):
            #  - move objective outside/at the same level as the class PFR, replace self by pfr :
            #    def _objective(guess, pfr, param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx):
            #  - pass self in args of _objectives: differential_evolution(_objective, bounds, args=(self,
            #    param_to_optimize, exp_conc, exp_idx, fit_to_exp_idx),
            #  for the moment can't be parallelized because pfr uses lambda functions (for the pulse..)
            #    that can't be pickled.

        logging.info(f"\nEnd of optimization: {res.message}")
        toc = datetime.datetime.now()

        if res.success:
            best_string = ""
            for val in res.x:
                best_string += f"{val:.5e} "
            logging.info(f"Optimized parameters: {best_string}")
            logging.info(f"             Min SSE: {res.fun:.5e}")
        else:
            if popsize:
                logging.info(
                    "Optimization did not end successfully. You might want to restart an optimization with the"
                )
                logging.info("following array specifying the last population:\n")
                print(f"it: {it}")
                init_array = "init_pop = np.array([\n"
                extra_trials = (it + 1) % (popsize * len(param_to_optimize))
                if not extra_trials:
                    last_pop = trials[it - popsize * len(param_to_optimize) + 1 :]
                else:
                    last_pop = trials[
                        it
                        - popsize * len(param_to_optimize)
                        - extra_trials
                        + 1 : -extra_trials
                    ]
                for trial in last_pop:
                    init_array += "["
                    for par in trial:
                        init_array += f"{par:.5e}, "
                    init_array += "],\n"
                init_array += "])"
                logging.info(init_array)
            else:
                logging.info("Optimization did not end successfully.")

        if options["disp"]:
            print(f"         Optimization time: {(toc - start_time)}")
            print(f"         Final parameters: {res.x}")

        if param_to_set is not None:
            all_param = {**param_to_set, **param_to_optimize}
        else:
            all_param = param_to_optimize

        newpfr = PFR(
            self._cti,
            self._init_X,
            self._inlet_X,
            self._inlet_F,
            self._volume,
            P=self.P,
            T=self.T,
            area=self._area,
            kin_param_to_set=all_param,
        )

        fitted_concentrations = newpfr.composition_vs_time(exp_conc.z)["X"][
            :, -1, :
        ].squeeze()
        newargs = (self, all_param)

        trials = NDDataset(trials)
        trials.title = "Trial solutions"

        if optimizer == "differential_evolution":
            # label trials per generation
            gen_labels = []
            for gen in range(len(func_values) // (popsize * len(param_to_optimize))):
                gen_labels.append(
                    ["G_" + str(gen)] * (popsize * len(param_to_optimize))
                )
            gen_labels.append(
                ["G_polish"] * (len(func_values) % (popsize * len(param_to_optimize)))
            )
            gen_labels = [item for sublist in gen_labels for item in sublist]
        else:
            gen_labels = None

        trials.set_coordset(
            Coord(
                data=func_values, labels=gen_labels, title="objective function values"
            ),
            Coord(
                data=None,
                labels=list(param_to_optimize.keys()),
                title="kinetic parameters",
            ),
        )

        logging.info("**** Optimization exited gracefully ***")
        logging.info(f"Total execution time: {toc - start_time}")

        return {
            "fitted_concentrations": fitted_concentrations,
            "results": res,
            "trials": trials,
            "newargs": newargs,
        }
