# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# pragma: no cover
# excluded for coverage for the moment
"""
Utility functions to deal with Cantera input/output.
"""

import datetime
import logging
import re
import warnings
from collections.abc import Iterable

import numpy as np
import traitlets as tr
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares, minimize

from spectrochempy.core import debug_, error_
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.core.units import Quantity
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.exceptions import SpectroChemPyError
from spectrochempy.utils.optional import import_optional_dependency

__all__ = [
    "ActionMassKinetics",
    "PFR",
]

R = 8.314462618153241


# exception used in this module
class SolverError(SpectroChemPyError):
    """Error raised if solve_ivp (integrate) return a status < 0"""


# Utility
# --------
def _interpret_equation(eq, species):
    # transform an equation given as a string to a dictionary of species with
    # integer stoechiometric coefficients.

    regex = r"(((([\.,0-9]*)((?=[a-zA-Z])[a-zA-Z, 1-9]+))(?=\+?))?(?=(->|→)?))"

    matches = re.finditer(regex, eq.replace(" ", ""))
    equation = {}
    mult = -1  # reactants
    for match in matches:
        if not match.group(5):
            # no species
            continue
        s = match.group(5)
        if s not in species:
            raise ValueError(
                f'Species "{s}" in equation "{eq}" is not listed in species\n'
                f"Available species : {species}"
            )
        coef = match.group(4) if match.group(4) else 1
        equation[s] = float(coef) * mult
        if match.group(6) in ["->", "→"]:
            # shift to products
            mult = 1

    # normalize the coefficients (for float we assume that they have at the maximum
    # 2 decimals
    y = (np.array(list(equation.values())) * 100).astype(int)
    div = np.gcd.reduce(np.abs(y))
    equation = {k: int(v * 100 / div) for k, v in equation.items()}

    return equation


@tr.signature_has_traits
class ActionMassKinetics(tr.HasTraits):
    """
    An object which stores a reaction network of elementary reactions.

    It stores its rate parameterization, initial concentrations, temperature profile,
    with methods for evaluating production rates and concentration profiles assuming
    action mass kinetic and closed reactor.

    Parameters
    ----------
    equations : `list` or `tuple` of `str`
        Strings giving the ``n_equations`` chemical equation of the network.
        Reactants and products must be separated by a ``"->"`` or "→" symbol,
        The name of each species should match a key of the `species` dictionary.
        Examples: ``"A + B -> C"`` or ``"2A -> 0.5 D"``\
    species : `dict`, optional
        Dictionary of initial concentrations for the `n_species` species.
    k : :term:`array-like`
        Iterable of shape `n_equations` x 2 with the Arrhenius rate parameters
        ((:math:`A_1`\ , :math:`Ea_1`\ ), ... (:math:`A_n`\ , :math:`Ea_n`\ )).
    T : `float`, `Quantity` or `callable`\ , optional, default: 298.0
        Temperature. If it is not a temperature quantity, the unit is assumed to be
        in Kelvin. A function can also be provided which output a temperature `T`
        vs. time `t`\ .
    """

    # internal parameters
    _equations = tr.List(tr.Unicode(), help="List of model equations")
    _concentrations = tr.Dict(help="A dictionary of model's species:concentrations")
    _species = tr.List(help="a list of species in this model")
    _M = Array(help="Stoichiometric matrix M = A+B")
    _k = Array(help="Arrhenius rate")
    _T = tr.Union((tr.Float(), tr.Callable()), default_value=298.0, help="Temperature")

    def __init__(self, equations, species, k, T=298.0, **kwargs):

        self._k = k
        self._T = T

        # initialise concentrations, species and equations
        self._concentrations = species
        self._species = list(self._concentrations.keys())
        self._equations = equations

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.validate("_k")
    def _k_validate(self, proposal):
        # k must be an iterable of pairs (A_1, Ea_1)
        k = proposal.value
        # k is an array (even if a list or tuple has been initialy provided (Array)
        if k.shape[-1] != 2:
            raise ValueError("k must be an iterable of pairs: shape=(n_reactions, 2)")
        # Add more validation?
        # ...
        return k

    @tr.validate("_T")
    def _T_validate(self, proposal):
        Tp = proposal.value

        if isinstance(Tp, Quantity):
            Tp = Tp.to("K").magnitude

        if isinstance(Tp, float):
            # if T float, transform it to a callable
            T = lambda t: Tp
        else:
            T = Tp
        return T

    @tr.observe("_equations")
    def _stoichio_matrix(self, change):
        # generate stoichio matrix
        equations = change.new
        M = np.zeros((self.n_equations, self.n_species))
        for i, eq in enumerate(equations):
            equation = _interpret_equation(eq, self._species)
            # fill M matrix in the order of the species list
            M[i] = [equation[k] if k in equation else 0 for k in self._species]
        self._M = M

    # ----------------------------------------------------------------------------------
    # Public properties
    # ----------------------------------------------------------------------------------
    @property
    def A(self):
        """
        Stoichiometry matrix A

        Stoichiometry matrices `A` and `B` are defined in :cite:t:`chellaboina:2009`\ .
        """
        return (-self._M).clip(0)

    @property
    def B(self):
        """
        Stoichiometry matrix B

        Stoichiometry matrices `A` and `B` are defined in :cite:t:`chellaboina:2009`\ .
        """
        return self._M.clip(0)

    @property
    def n_equations(self):
        """Number of reaction equations"""
        return len(self._equations)

    @property
    def n_species(self):
        """Number of species"""
        return len(self._species)

    @property
    def species(self):
        """Components names."""
        return list(self._concentrations.keys())

    @property
    def concentrations(self):
        """Concentrations."""
        return list(self._concentrations.values())

    def integrate(self, t, method="RK45", **kwargs):
        """
        Integrate the kinetic equations at times `t`.

        This function computes and integrates the set of kinetic differential
        equations given the initial concentration values:

        .. math::
            dC / dt =  (B - A).T  K C^A

            C(t_0) = C_0

        where :math:`A` and :math:`B` are the stoichiometry matrices,
        :math:`K` is the diagonal matrix of rate constants and :math:`C^A` is the
        vector-matrix exponentiation of :math:`C` by :math:`A`\ .

        Parameters
        ----------
        t : :term:`array-like` of shape (``t_points``\ ,)
            Iterable with time values at which the concentrations are computed.
        method : `str` or `~scipy.integrate.OdeSolver`\ , optional, default: ``'RK45'``
            Integration method to use:

            * ``'RK45'`` (default): Explicit Runge-Kutta method of order 5(4).
            * ``'RK23'`` : Explicit Runge-Kutta method of order 3(2).
            * ``'DOP853'``: Explicit Runge-Kutta method of order 8.
            * ``'Radau'`` : Implicit Runge-Kutta method of the Radau IIA family of
              order 5.
            * ``'BDF'`` : Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation.
            * ``'LSODA'`` : Adams/BDF method with automatic stiffness detection and
              switching.

            Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') should be used
            for non-stiff problems and implicit methods ('Radau', 'BDF') for
            stiff problems. Among Runge-Kutta methods, 'DOP853' is recommended
            for solving with high precision (low values of `rtol` and `atol` ).
            If not sure, first try to run 'RK45'. If it makes unusually many
            iterations, diverges, or fails, your problem is likely to be stiff and
            you should use 'Radau' or 'BDF'. 'LSODA' can also be a good universal
            choice, but it might be somewhat less convenient to work with as it
            wraps old Fortran code.
            You can also pass an arbitrary class derived from
            `~scipy.integrate.OdeSolver` which implements the solver.
        **kwargs
            Additional keyword parameters. See Other Parameters.

        Other Parameters
        ----------------
        return_NDDataset : `bool`\ , optional, default: `True`
            Whether to return a NDDataset
        return_meta : `bool`\ , optional, default: `False`
            Whether to return a dictionary with the solver results.
            Note that when return_NDDataset is True, meta is always
            included in the meta attribute of the NDDataset.

        Returns
        -------
        C : `~numpy.ndarray` or `NDDataset`, shape ( ``t_points``\ , ``n_species``\ )
            Values of the solution at times `t`\ .
        meta : Bunch object with the following fields defined:

            * t : ndarray, shape (t_points,)
              Time points.
            * sol : `~scipy.integrate.OdeSolution` or None
              Found solution as `~scipy.integrate.OdeSolution` instance;
              None if `dense_output` was
              set to False.
            * t_events : `list` of `~numpy.ndarray` or `None`
              Contains for each event type a list of arrays at which an event of
              that type event was detected. `None` if events` was None.
            * y_events : `list` of `~numpy.ndarray` or `None`
              For each value of `t_events` , the corresponding value of the solution.
              `None` if events was `None`.
            * nfev : `int`
              Number of evaluations of the right-hand side.
            * njev : `int`
              Number of evaluations of the Jacobian.
            * nlu : `int`
              Number of LU decompositions.
            * status : `int`
              Reason for a successful algorithm termination:

                    *  0 : The solver successfully reached the end of `tspan` .
                    *  1 : A termination event occurred.

            * message : `str`
              Human-readable description of the termination reason.
        """

        def production_rates(ti, Ci):
            """
            Compute the production rates :math:`\frac{dC,dt}`.

            Compute the n_s production rates at time :math:`t_i` according to:

            .. math::
                dC / dt =  (B - A).T  K C_i^A
                C_i = C(t_i)

            where :math:`A` and :math:`B` are the stoichiometry matrices, :math:`K` is
            the diagonal matrix of rate constants and :math:`C_i^A` is the vector-matrix
            exponentiation of :math:`C_i` by :math:`A`.

            Parameters
            ----------
            ti: `float`
                Time.
            Ci: `~numpy.ndarray`
                1D vector of the concentrations at time `ti`\ .
            """
            beta = 1 / R / self._T(ti)
            K = np.diag(self._k[:, 0] * np.exp(-beta * self._k[:, 1]))
            A, B = self.A, self.B
            BmAt = (B - A).T
            return np.dot(np.dot(BmAt, K), _vm_exp(Ci, A))

        bunch = solve_ivp(
            production_rates,
            (t[0], t[-1]),
            self.concentrations,
            t_eval=t,
            method=method,
        )

        debug_(bunch.message)
        if bunch.status != 0:
            raise SolverError(bunch.message)

        C = bunch.y.T
        t = bunch.t

        # remove some keys from bunch
        del bunch.y
        del bunch.success

        return_dataset = kwargs.get("return_NDDataset", True)
        if return_dataset:
            C = NDDataset(C, name="Concentrations")
            C.y = Coord(t, title="time")
            C.x = Coord(range(self.n_species), labels=self.species, title="species")
            C.history = "Created using ActionMassKinetics.integrate"
            C.meta.update(bunch)

        if kwargs.get("return_meta", False) and not return_dataset:
            return (C, bunch)
        return C

    def _modify_kinetics(self, dict_param):
        for item in dict_param:
            i_r, p = item.split("[")[-1].split("].")
            if p == "A":
                self._k[int(i_r), 0] = dict_param[item]
            elif p == "Ea":
                self._k[int(i_r), 1] = dict_param[item]
            else:
                raise ValueError("something went wrong in parsing the dict of params")

    def fit_to_concentrations(
        self, Cexp, iexp, i2iexp, dict_param_to_optimize, **kwargs
    ):
        """
        Fit rate parameters and concentrations to a concentration profile.

        Parameters
        ------------
        Cexp : `NDDataset`
            Experimental concentration profiles on which to fit the model.
            `Cexp` can contain more concentration profiles than those to fit.
        iexp : `int`
            Indexes of experimental concentration profiles on which the model will be
            fitted.
        i2iexp : `int`
            Correspondence between optimized (external) concentration profile and
            experimental concentration profile.
        dict_param_to_optimize : `dict`
            rate parameters to optimize. Keys should be 'k[i].A' and 'k[i].Ea' for
            pre-exponential factor.
        **kwargs
            Parameters for the optimization (see `~scipy.optimize.minimize`\ ).

        Returns
        --------
        `dict`
            A result dictionary.
        """

        def objective(params, Cexp, iexp, i2iexp, dict_param_to_optimize):
            for param, item in zip(params, dict_param_to_optimize):
                dict_param_to_optimize[item] = param
            self._modify_kinetics(dict_param_to_optimize)
            Chat = self.integrate(Cexp.y.data, return_NDDataset=False)
            return np.sum(np.square(Cexp.data[:, iexp] - Chat[:, i2iexp]))

        method = kwargs.get("method", "Nelder-Mead")
        bounds = kwargs.get("bounds", None)
        tol = kwargs.get("tol", None)
        options = kwargs.get("options", {"disp": True})

        guess_param = np.zeros((len(dict_param_to_optimize)))
        for i, param in enumerate(dict_param_to_optimize):
            guess_param[i] = dict_param_to_optimize[param]

        if options["disp"]:
            print("Optimization of the parameters.")
            print(f"         Initial parameters: {guess_param}")
            print(
                f"         Initial function value: "
                f"{objective(guess_param, Cexp, iexp, i2iexp, dict_param_to_optimize)}"
            )
        tic = datetime.datetime.now(datetime.timezone.utc)
        optim_res = minimize(
            objective,
            guess_param,
            args=(Cexp, iexp, i2iexp, dict_param_to_optimize),
            method=method,
            bounds=bounds,
            tol=tol,
            options=options,
        )
        toc = datetime.datetime.now(datetime.timezone.utc)
        opt_param = optim_res.x
        if options["disp"]:
            print(f"         Optimization time: {toc - tic}")
            print(f"         Final parameters: {opt_param}")

        Ckin = self.integrate(Cexp.y.data, return_NDDataset=False)

        for i, param in enumerate(dict_param_to_optimize):
            dict_param_to_optimize[param] = opt_param[i]

        return Ckin, (iexp, i2iexp, dict_param_to_optimize), optim_res


def _vm_exp(x: Iterable, A: Iterable):
    """
    Vector matrix exponentiation

    The vector-matrix exponentiation is the operation that maps x
    and A to its vector-matrix power $x^A$ which given by:
    $$ \left[ x_1^{A_{11}}x_2^{A_{21}} ... x_n^{A_{n1}} \;\; x_1^{A_{1n}}x_2^{A_{2n}}
    ... x_n^{A_{nn}} \right]$$


    Parameters
    ----------
    x: (px1) iterable of float
         Columns vector
    A: (pxq) iterable of int
         Matrix

    Returns
    -------
    Vm_exp: (px1) iterable
         x^A vector

    References
    ----------
    [1] Chellaboina et al., "Modeling and analysis of mass-action kinetics", IEEE
        control systems (2009),DOI: 10.1109/MCS.2009.932926
    [2] Gjerrit Meinsma, "Dimensional and Scaling Analysis" SIAM review, Vol. 61, No. 1,
        pp. 159–184 (2009), DOI: 10.1137/16M1107127
    """
    out = [1] * len(A)
    for i, A_i in enumerate(A):
        for x_j, A_ij in zip(x, A_i):
            out[i] *= x_j**A_ij
    return out


# --------
# CANTERA UTILITIES
# -------
ct = import_optional_dependency("cantera", errors="ignore")


def _cantera_is_not_available():
    if ct is None:
        error_(
            ImportError,
            "Missing optional dependency 'cantera'.  "
            "Use conda or pip to install cantera.",
        )
    return ct is None


def _ct_modify_rate(reactive_phase, i_reaction, rate):
    """
    Modify the reaction rate of with index i_reaction to have the same rate parameters
    as rate.

    Parameters
    ----------
    reactive_phase : instance of cantera.ideal_gas or cantera.composite.Interface
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


def _ct_modify_surface_kinetics(surface, param_to_set):
    """
    Changes a set of numerical parameters of an Interface among following:
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
            _ct_modify_rate(surface, rxn, ct.Arrhenius(param_to_set[param], b, E))

        elif param.split(".")[-1] == "temperature_exponent":
            str_rate = "surface." + ".".join(param.split(".")[-3:-1])
            A, E = eval(
                str_rate + "pre_exponential_factor," + str_rate + ".activation_energy "
            )
            rxn = int(param.split(".")[0].split("[")[-1].split("]")[0])
            _ct_modify_rate(surface, rxn, ct.Arrhenius(A, param_to_set[param], E))

        elif param.split(".")[-1] == "activation_energy":
            str_rate = "surface." + ".".join(param.split(".")[-3:-1])
            A, b = eval(
                str_rate
                + "pre_exponential_factor,"
                + str_rate
                + ".temperature_exponent"
            )
            rxn = int(param.split(".")[0].split("[")[-1].split("]")[0])
            _ct_modify_rate(surface, rxn, ct.Arrhenius(A, b, param_to_set[param]))


class PFR:
    """
    PFR reactor as a CSTR in series.

    Parameters
    ----------
    cti_file : `str`
        The cti file must contain a gas phase named 'gas' and optionally a reactive
        surface named 'surface'.
    init_X : `dict`\ , :term:`array-like`
        Initial composition of the reactors.
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
                _ct_modify_surface_kinetics(surface, kin_param_to_set)
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
                # it is tricky to pass non explicit lambda functions to
                # MassFlowControllers
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

        # create other cstrs and link them with the previous one through a pressure
        # controller

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
                    _ct_modify_surface_kinetics(surface, kin_param_to_set)
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
        """
        Fit rate parameters and concentration for a given concentration profile.

        Function fitting rate parameters and concentrations to a given concentration
        profile at the outlet of the pfr.

        Parameters
        ------------

        exp_conc: NDDataset
            experimental concentration profiles on which to fit the model. Can contain
            more concentration profiles than those to fit. the y Coord should be time.

        exp_idx:
            indexes of experimental concentration profiles on which the model
            will be fitted.

        fit_to_exp_idx:
            correspondence between optimized concentration profile and experimental
            concentration profile.

        param_to_optimize: dict
            reactive phase parameters to optimize.

        param_to_set: dict
            names of kinetic parameters differing from the cti file but fixed
            during optimization.

        logfile: `None` (default) or str
            name of the logfile.

        **kwargs
            parameters for the optimization (see scipy.optimize.minimize).

        Returns
        -------
        `dict`
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
                        "model could not be integrated with these parameters. "
                        "Objective function set to Inf",
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
                                    f"                      Minimum SSE: {min_sse:.3e} "
                                    f"(Eval # {it_min_sse})"
                                )
                            else:
                                logging.info(
                                    f"                      Minimum SSE: {min_sse:.3e} "
                                    f"({(min_sse - prev_min_sse) / prev_min_sse:+.3%},"
                                    f" Eval # {it_min_sse})"
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
            # then param_to_optimize are expected to be bounds for each variable

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
            # set optional parameters / default set to scipy defaults
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
                    "Optimization did not end successfully. You might want to restart "
                    "an optimization with the"
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
