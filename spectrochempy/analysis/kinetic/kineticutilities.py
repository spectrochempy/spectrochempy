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
from functools import partial

import numpy as np
import traitlets as tr
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares, minimize

from spectrochempy.application import error_
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
SCIPY_MINIMIZE_METHODS = [
    "NELDER-MEAD",
    "POWELL",
    "CG",
    "BFGS",
    "NEWTON-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "TRUST-CONSTR",
    "DOGLEG",
    "TRUST-NCG",
    "TRUST-KRYLOV",
    "TRUST-EXACT",
]


# exception used in this module
class SolverError(SpectroChemPyError):
    """Error raised if solve_ivp (integrate) return a status < 0"""


# ------------------------------------------------------------------------------------
# ACTION MASS KINETICS
# ------------------------------------------------------------------------------------

# Utility
# --------
def _interpret_equation(eq, species):
    # transform an equation given as a string to dictionaries of species with
    # integer stoechiometric coefficients for the left (reactants) and right (products)
    # side of the equation.

    regex = r"(((([\.,0-9]*)((?=[a-zA-Z])[a-zA-Z, 1-9]+))(?=\+?))?(?=(->|→)?))"

    matches = re.finditer(regex, eq.replace(" ", ""))
    left, right = {}, {}
    is_reactant = True
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
        try:
            coef = int(coef)
        except ValueError:
            raise ValueError(
                f"Stoichiometric coeffcients must be integers. Could not "
                f"convert {coef} in int"
            )
        if is_reactant:
            left[s] = coef
        else:
            right[s] = coef

        if match.group(6) in ["->", "→"]:
            # shift to products
            is_reactant = False
    return left, right


@tr.signature_has_traits
class ActionMassKinetics(tr.HasTraits):
    """
    An object which stores a reaction network of elementary reactions.

    It stores its rate parameterization, set(s) of initial concentrations,
    temperature profile(s), with methods for evaluating production rates and
    concentration profiles assuming action mass kinetic and closed reactor.

    Parameters
    ----------
    reactions : 'dict' or `list` or `tuple` of `str`
        Strings giving the ``n_reactions`` chemical equation of the network.
        Reactants and products must be separated by a ``"->"`` or "→" symbol,
        The name of each species should match a key of the `species` dictionary.
        Examples: ``"A + B -> C"`` or ``"2A → D"``\
    species : `dict` or `list` or `tuple` of `dict`
        Dictionary or list of dictionaries giving  the initial concentrations for the
        `n_species` species.
    arrhenius : :term:`array-like`
        Iterable of shape `n_reactions` x 1, `n_reactions` x 2  or `n_reactions` x 3
        with either the isothermal rate constants (:math:`k_1`\ , ..., :math:`k_n`\ ) or
        the Arrhenius rate parameters ((:math:`A_1`\ , :math:`b_1`\ , :math:`Ea_1`\ ),
        ... (:math:`A_n`\ , :math:`b_n`\ , :math:`Ea_n`\ )) or  ((:math:`A_1`\ ,
        :math:`Ea_1`\ ), ...)).  If a 2-column iterable is provided the temperature
        exponents are set to 0.
    T : `float`\ , `Quantity`\ , `callable` or `list` or `tuple` of , or None, optional default: None
        Temperature. If None, or not given, the system is considered isothermal and T = 298.0
        If it is not a temperature quantity, the unit is assumed to be
        in Kelvin. A function can also be provided which output a temperature `T` in K
        vs. time `t`\ .

    Examples:
    ---------
    # A simple A → B → C:
    >>> reactions = ("A -> B", "B -> C")
    >>> species_concentrations = {"A": 1.0, "B": 0.0, "C": 0.0}
    >>> time = np.arange(0, 10)
    >>> k_exp = np.array(((1.0e8, 52.0e3), (1.0e8, 50.0e3)))
    >>> kin_exp = scp.ActionMassKinetics(reactions, species_concentrations, k_exp, T=298.0)
    >>> C_exp = kin_exp.integrate(time)
    >>> print(f"Concentrations at t = 4 : {C_exp[4.].data}")
    Concentrations at t = 4 : [[  0.7355   0.1879  0.07666]]

    # Several sets of experimental conditions can be used. In this case, `species`, `T`
    # are set using lists or tuples of the same length, even if only one is changed:
    >>> species_concentrations = ({"A": 1.0, "B": 0.0, "C": 0.0}, {"A": 1.0, "B": 0.0, "C": 0.0})
    >>> T = (298.0, 308.0)
    >>> time = (np.arange(0, 10), np.arange(0, 5))
    >>> kin_exp = scp.ActionMassKinetics(reactions, species_concentrations, k_exp, T=T)
    >>> C_exp = kin_exp.integrate(time)
    >>> print(f"Concentrations at {T[0]}K, t = 4 : {C_exp[0][4.].data}")
    >>> print(f"Concentrations at {T[1]}K, t = 4 : {C_exp[1][4.].data}")
    Concentrations at 298.0K, t = 4 : [[  0.7355   0.1879  0.07666]]
    Concentrations at 308.0K, t = 4 : [[  0.5448    0.236   0.2192]]
    """

    # internal parameters
    _reactions = tr.Union(
        (tr.List(tr.Unicode()), tr.Dict()), help="List or dict of model reactions"
    )
    _reactions_names = tr.List(tr.Unicode(), help="List of model reactions names")
    _init_concentrations = tr.Union(
        (tr.Dict(), tr.List(tr.Dict()), tr.Tuple(tr.Dict())),
        help="A dictionary or list/tuple of dictionaries of model's species initial concentrations",
    )
    _species = tr.List(help="a list of species in this model")
    _A = Array(help="Stoichiometric matrix A (reactants)")
    _B = Array(help="Stoichiometric matrix B (products)")
    _arrhenius = Array(help="Arrhenius-like rate constant parameters")
    _T = tr.Union(
        (
            tr.Float(),
            tr.Callable(),
            tr.List(),
            tr.Tuple(),
        ),
        allow_none=False,
        default_value=298.0,
        help="Temperature",
    )

    def __init__(self, reactions, species_concentrations, arrhenius, T=298.0, **kwargs):

        # initialise concentrations, species, reactions, arrhenius, T
        self._init_concentrations = species_concentrations
        if isinstance(self._init_concentrations, (list, tuple)):
            # we have two sets (or more) of experimental data.
            self._nset = len(self._init_concentrations)
            self._species = list(self._init_concentrations[0].keys())
            # check that all keys are the same:
            for i, conc in enumerate(self._init_concentrations[1:]):
                if set(conc.keys()) != set(self._species):
                    raise ValueError(
                        f"species names in species_concentrations[{i}] do not match "
                        f"species names in species_concentrations[0]"
                    )

        else:
            self._species = list(self._init_concentrations.keys())
            self._nset = 1

        if isinstance(reactions, (list, tuple)):
            self._reactions = reactions
            self._reactions_names = [f"equation {i}" for i in range(len(reactions))]
        elif isinstance(reactions, dict):
            self._reactions = list(reactions.values())
            self._reactions_names = list(reactions.keys())
        else:
            raise TypeError(
                f"reactions should contain be a dict, a list or a tuple of strings, "
                f"not a {type(reactions).__name__}."
            )

        self._arrhenius = arrhenius

        self._T = T

        self._reaction_rates = self._write_reaction_rates()
        self._production_rates = self._write_production_rates()
        self._jacobian = self._write_jacobian()

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------

    @tr.validate("_arrhenius")
    def _arrhenius_validate(self, proposal):
        # arrhenius can be an iterable of:
        # - single k values (k_1, ... k_n)
        # - pairs of arrhenius parameters ((A_1, Ea_1), ... (A_n, E_a_n))
        # - triplets of arrhenius parameters  ((A_1, b_1, Ea_1), ... (A_n, b_n, E_a_n))
        arrhenius = proposal.value
        # k is an array (even if a list or tuple has been initially provided (Array)
        if len(arrhenius.shape) == 1:
            # this id a 1D array
            if arrhenius.shape[-1] != self.n_reactions:
                raise ValueError(
                    f"arrhenius should contain {self.n_reactions} rate "
                    f"constants, {arrhenius.shape[-1]} have been provided"
                )
            if any(arrhenius < 0):
                warnings.warn(
                    "at least a rate constant is negative... are you sure of " "that ?!"
                )
        elif arrhenius.shape[-1] == 2:
            # this is a 2D array with lines == [A, Ea]
            if arrhenius.shape[0] != self.n_reactions:
                raise ValueError(
                    f"arrhenius should contain {self.n_reactions} rate "
                    f"constants, {arrhenius.shape[0]} have been provided"
                )
            if (arrhenius < 0).any():
                warnings.warn(
                    "a least a pre-exp factor or activation energy is "
                    "negative... ae you sure of that ?!"
                )
            # now add temperature exponents = 0
            arrhenius = np.array(
                [arrhenius[:, 0].T, np.zeros(arrhenius.shape[0]).T, arrhenius[:, 1].T]
            ).T
        elif arrhenius.shape[-1] == 3:
            # this is a 2D array with lines == [A, b, Ea]
            if arrhenius.shape[0] != self.n_reactions:
                raise ValueError(
                    f"arrhenius should contain {self.n_reactions} rate "
                    f"constants, {arrhenius.shape[0]} have been provided"
                )
            if (arrhenius[:, [0, 2]] < 0).any():
                warnings.warn(
                    "a least a pre-exp factor or activation energy is "
                    "negative... are you sure of that ?!"
                )
        return arrhenius

    @tr.validate("_T")
    def _T_validate(self, proposal):
        Tp = proposal.value
        if isinstance(Tp, Quantity):
            T = Tp.to("K").magnitude
        else:
            T = Tp
        return T

    @tr.observe("_reactions")
    def _stoichio_matrix(self, change):
        # generate stoichio matrix
        reactions = change.new
        A = np.zeros((self.n_reactions, self.n_species))
        B = np.zeros((self.n_reactions, self.n_species))
        for i, eq in enumerate(reactions):
            left, right = _interpret_equation(eq, self._species)
            A[i] = [left[k] if k in left else 0 for k in self._species]
            B[i] = [right[k] if k in right else 0 for k in self._species]
        self._A = A
        self._B = B
        self._BmAt = (B - A).T

    # ----------------------------------------------------------------------------------
    # Public properties
    # ----------------------------------------------------------------------------------
    @property
    def A(self):
        """
        Stoichiometry matrix A

        Stoichiometry matrices `A` and `B` are defined in :cite:t:`chellaboina:2009`\ .
        """
        return self._A

    @property
    def B(self):
        """
        Stoichiometry matrix B

        Stoichiometry matrices `A` and `B` are defined in :cite:t:`chellaboina:2009`\ .
        """
        return self._B

    @property
    def n_reactions(self):
        """Number of reaction reactions"""
        return len(self._reactions)

    @property
    def n_species(self):
        """Number of species"""
        return len(self._species)

    @property
    def species(self):
        """Components names."""
        return self._species

    @property
    def init_concentrations(self):
        """Concentrations."""
        if isinstance(self._init_concentrations, (list, tuple)):
            return [list(init_conc.values()) for init_conc in self._init_concentrations]
        else:
            return list(self._init_concentrations.values())

    def _write_reaction_rates(self):
        """Return the expressions of production rates as a string"""
        block = "["
        for j, line in enumerate(self._A):
            reac_rate = f"k[{j}]"
            for k, nu in enumerate(line):
                if nu == 1:
                    reac_rate += f" * C[{k}]"
                elif nu > 1:
                    reac_rate += f" * C[{k}]**{nu}"
            reac_rate += ", "
            block += reac_rate
        block += "]"
        return block

    def _print_reaction_rates(self):
        print(self._write_reaction_rates().replace(",", ",\n"))

    def _write_production_rates(self):
        """Return the expressions of production rates as a string"""
        block = "["
        for line in (self._B - self._A).T:
            prod_rate = ""
            if not line.any():
                # only zeors => spectator species
                prod_rate = "0"
            else:
                for j, n in enumerate(line):
                    if n != 0:
                        if n == 1:
                            prod_rate += f" + k[{j}]"
                        elif n == -1:
                            prod_rate += f" - k[{j}]"
                        elif n > 1:
                            prod_rate += f" + {n} * k[{j}]"
                        elif n < -1:
                            prod_rate += f" {n} * k[{j}]"

                        for k, nu in enumerate(self.A[j]):
                            if nu == 1:
                                prod_rate += f" * C[{k}]"
                            elif nu > 1:
                                prod_rate += f" * C[{k}]**{nu}"
            prod_rate += ", "
            block += prod_rate
        block += "]"
        return block

    def _print_production_rates(self):
        print(self._write_production_rates().replace(",", ",\n"))

    def _write_jacobian(self):
        """Return the expressions of the jacobian of the production rates as a string"""
        block = "["
        for i, line in enumerate((self._B - self._A).T):
            jac = "["
            for j in range(self.n_species):
                # compute jac[i,j] = d(dCidt)/dCj
                is_null = True
                for k, n in enumerate(line):
                    if self.A[k, j] > 0 and n != 0:
                        # the stoichiometruc coef. of reactant j is non-null
                        # and reactant i is involved
                        is_null = False
                        if n == 1:
                            jac += f" + k[{k}]"
                        elif n == -1:
                            jac += f" - k[{k}]"
                        elif n > 1:
                            jac += f" + {n} * k[{j}]"
                        elif n < -1:
                            jac += f" {n} * k[{j}]"

                        for jj, nu in enumerate(self.A[k]):
                            if nu == 1:
                                jac += f" * C[{jj}]" if jj != j else ""
                            elif nu > 1:
                                jac += (
                                    f" * C[{jj}]**{nu}"
                                    if jj != j
                                    else f" * {nu} * C[{jj}]**{nu-1}"
                                )
                if is_null:
                    jac += "0,"
                else:
                    jac += ", "
            block += jac + " ],"
        block += "]"
        return block

    def _print_jacobian(self):
        print(self._write_jacobian().replace(" ],", "],\n"))

    def integrate(
        self,
        t,
        k_dt=None,
        method="LSODA",
        left_op=None,
        c_names=None,
        use_jac=False,
        atol=1e-6,
        rtol=1e-3,
        **kwargs,
    ):
        """
        Integrate the kinetic equations at times `t`.

        This function computes and integrates the set of kinetic differential
        equations given the initial concentration values.

        Parameters
        ----------
        t : :term:`array-like` of shape (``t_points``\ ,) or list or tuple of
        `arrays-like`.
            Iterable with time values or sets of timle values at which the
            concentrations are computed.

        k_dt : `float` or `None'
            Resolution of the time grid used to compute `k(T(t))`\ . Used only for non
            isothermal reaction.

        method : `str` or `~scipy.integrate.OdeSolver`\ , optional, default: ``'LSODA'``
            Integration method to use:

            * ``'LSODA'`` (default): Adams/BDF method with automatic stiffness detection and
              switching.
            * ``'RK45'`` (default): Explicit Runge-Kutta method of order 5(4).
            * ``'RK23'`` : Explicit Runge-Kutta method of order 3(2).
            * ``'DOP853'``: Explicit Runge-Kutta method of order 8.
            * ``'Radau'`` : Implicit Runge-Kutta method of the Radau IIA family of
              order 5.
            * ``'BDF'`` : Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation.

            'LSODA' is generally faster and seems a good choice for the systems
            treated in scpy so far.
            Explicit Runge-Kutta methods ('RK23', 'RK45', 'DOP853') can be used
            for non-stiff problems and implicit methods ('Radau', 'BDF') for
            stiff problems. Among Runge-Kutta methods, 'DOP853' is recommended
            for solving with high precision (low values of `rtol` and `atol` ).
            If not sure, first try to run 'RK45'. If it makes unusually many
            iterations, diverges, or fails, your problem is likely to be stiff and
            you should use 'Radau' or 'BDF'.
            You can also pass an arbitrary class derived from
            `~scipy.integrate.OdeSolver` which implements the solver.

        left_op : array_like, optional
            A (m x n_species) array to left multiply the (n_species, n_times) array
            obtained after integration:
            `C.T = left_op @ C.T`\ . Can be used to pool and/or remove some
            concentration profiles in/from the output matrix of concentrations

        c_names : list of str
            List of names for each concentration profile. Used if `left_op` is not None
            to name/rename the output concentration profiles.

        use_jac : `Bool`
            Whether to use the jacobian. Useful for stiff problems, can slightly
            increase the execution time for non-stiff problems.

        atol, rtol : `float` or `array_like`, optional
            Relative and absolute tolerances. The solver keeps the local error estimates
            less than `atol + rtol * abs(y)`. Here `rtol` controls a relative accuracy
            (number of correct digits), while `atol` controls absolute accuracy
            (number of correct decimal places). To achieve the desired `rtol`, set
            `atol < min(rtol * C)` so that `rtol` dominates the allowable error.
            If `atol > rtol * C` the number of correct digits is not guaranteed.
            Conversely, to achieve the desired `atol` set `rtol` such that
            `max(rtol * C) < atol` is always smaller than atol. If components of C have
            different scales, it might be beneficial to set different atol values for
            different components by passing array_like with shape (n_species,) for
            atol. Default values are `rtol=1e-3` and `atol=1e-6`\ .

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
        # uncomment for debugging and optimization
        # import time
        # t0 = time.time()

        global_env = {}
        locals_env = {}

        if self._nset == 1:
            conditions = zip([self._T], [self._init_concentrations], [t])
        else:
            conditions = zip(self._T, self._init_concentrations, t)

        C = []
        for i, (T, C0, t) in enumerate(conditions):
            if callable(T):
                # non-isothermal: a grid of k_i values spaced by k_dt time intervals
                # is computed
                t_grid = np.arange(0, t[-1] + k_dt, k_dt)
                T_grid = np.expand_dims(T(t_grid), axis=1)
                k_grid = (
                    self._arrhenius[:, 0]
                    * np.power(T_grid, self._arrhenius[:, 1])
                    * np.exp(-self._arrhenius[:, 2] / 8.314 / T_grid)
                )

                # uncomment for debugging and optimization
                # t1 = time.time()

                exec(
                    f"def f_(self, k_grid, k_dt, t, C): k = k_grid[int(t/k_dt)] ; return self._BmAt @ {self._reaction_rates}",
                    global_env,
                    locals_env,
                )

                if use_jac:
                    # define jac_ = d[dC/dt]/dCi
                    exec(
                        f"def jac_(self, k_grid, k_dt, t, C): k = k_grid[int(t/k_dt)] ; return{self._jacobian}",
                        global_env,
                        locals_env,
                    )
                    jac = partial(locals_env["jac_"], self, k_grid, k_dt)
                else:
                    jac = None

                # uncomment for debugging and optimization
                # t2 = time.time()

                bunch = solve_ivp(
                    partial(locals_env["f_"], self, k_grid, k_dt),
                    (t[0], t[-1]),
                    list(C0.values()),
                    t_eval=t,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    jac=jac,
                )

                # uncomment for debugging and optimization
                # t3 = time.time()

            else:
                if len(self._arrhenius.shape) == 1:
                    # _arrhenius is 1D array of rate constants
                    k = self._arrhenius

                    # uncomment for debugging and optimization
                    # t1 = time.time()

                else:
                    # isothermal, the k_i are computed once
                    k = (
                        self._arrhenius[:, 0]
                        * T ** self._arrhenius[:, 1]
                        * np.exp(-self._arrhenius[:, 2] / R / T)
                    )

                # uncomment for debugging and optimization
                # t1 = time.time()

                exec(
                    f"def f_(self, k, t, C): return{self._production_rates}",
                    global_env,
                    locals_env,
                )
                if use_jac:
                    # define jac_ = d[dC/dt]/dCi
                    exec(
                        f"def jac_(self, k, t, C): return{self._jacobian}",
                        global_env,
                        locals_env,
                    )
                    jac = partial(locals_env["jac_"], self, k)
                else:
                    jac = None

                # uncomment for debugging and optimization
                # t2 = time.time()

                bunch = solve_ivp(
                    partial(locals_env["f_"], self, k),
                    (t[0], t[-1]),
                    list(C0.values()),
                    t_eval=t,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    jac=jac,
                )

                # uncomment for debugging and optimization
                # t2 = time.time()

            # uncomment for debugging (warning: debug_() multiply the exec time by 4...)
            # from from spectrochempy.application import debug_
            # debug_(bunch.message)
            # t4 = time.time()

            if bunch.status != 0:
                raise SolverError(bunch.message)

            C_ = (left_op @ bunch.y).T if left_op is not None else bunch.y.T
            t = bunch.t

            # uncomment for debugging and optimization
            # t5 = time.time()

            return_dataset = kwargs.get("return_NDDataset", True)
            if return_dataset:
                C.append(NDDataset(C_, name="Concentrations"))
                C[i].y = Coord(t, title="time")
                if left_op is None:
                    C[i].x = Coord(
                        range(self.n_species), labels=self.species, title="species"
                    )
                elif c_names is not None:
                    C[i].x = Coord(
                        range(left_op.shape[0]), labels=c_names, title="species"
                    )
                else:
                    C[i].x = Coord(
                        range(left_op.shape[0]),
                        labels=[f"species #{i}" for i in range(left_op.shape[0])],
                        title="species",
                    )
                C[i].history = "Created using ActionMassKinetics.integrate"
                C[i].meta.update(bunch)

                # uncomment for debugging and optimization
                # t6 = time.time()
                # print(f"time compute k       : {t1 - t0:f}, {100*(t1 - t0)/(t6-t0):f}%")
                # print(f"time load f (and jac): {t2 - t1:f}, {100*(t2 - t1)/(t6-t0):f}%")
                # print(f"time integration     : {t3 - t2:f}, {100*(t3 - t2)/(t6-t0):f}%")
                # print(f"time debug           : {t4 - t3:f}, {100*(t4 - t3)/(t6-t0):f}%")
                # print(f"time C,t             : {t5 - t4:f}, {100*(t5 - t4)/(t6-t0):f}%")
                # print(f"time to NDDataset    : {t6 - t5:f}, {100*(t6 - t5)/(t6-t0):f}%")

            elif kwargs.get("return_meta", False) and not return_dataset:
                C.append((C_, bunch))
            else:
                C.append(C_)

        if len(C) == 1:
            return C[0]
        else:
            return C

    def _modify_kinetics(self, dict_param, left_op=None):

        if len(self._arrhenius.shape) == 2:
            for item in dict_param:
                i_r, p = item.split("[")[-1].split("].")
                if p == "A":
                    self._arrhenius[int(i_r), 0] = dict_param[item]
                elif p == "b":
                    self._arrhenius[int(i_r), 1] = dict_param[item]
                elif p == "Ea":
                    self._arrhenius[int(i_r), 2] = dict_param[item]
                else:
                    raise ValueError(
                        "something went wrong in parsing the dict of params"
                    )
        else:
            for item in dict_param:
                i_r = item.split("[")[-1].split("]")[0]
                self._arrhenius[int(i_r)] = dict_param[item]

        if left_op is not None:
            self._arrhenius = left_op @ self._arrhenius

    def fit_to_concentrations(
        self,
        Cexp,
        iexp,
        i2iexp,
        dict_param_to_optimize,
        ivp_solver_kwargs={},
        optimizer_kwargs={},
    ):
        """
        Fit rate parameters and concentrations to a concentration profile.

        Parameters
        ------------
        Cexp : `NDDataset` or `list` ot `tuple` of NDDatasets
            Experimental concentration profiles on which to fit the model.
            each set of concentrations can contain more concentration profiles than those to fit.
        iexp : `int`
            Indexes of experimental concentration profiles on which the model will be
            fitted.
        i2iexp : `int`
            Correspondence between optimized (external) concentration profile and
            experimental concentration profile.
        dict_param_to_optimize : `dict` or None
            rate parameters to optimize. Keys should be 'k[i].A' and 'k[i].Ea' for
            pre-exponential factor.
        ivp_solver_kwargs : `dict`
            keyword arguments for the ode solver. Defaults are the same as for
            `~scipy.integrate.solve_ivp`\ , except for `method=LSDOA`
        optimizer_kwargs: `dict`
            keyword arguments the optimization (see `~scipy.optimize.minimize`\ ).

        Returns
        --------
        `dict`
            A result dictionary.
        """

        def objective(
            params,
            Cexp,
            iexp,
            i2iexp,
            dict_param_to_optimize,
            optimizer_left_op,
            ivp_solver_method,
            k_dt,
            C_op,
        ):
            """returns the SSE on concentrations profiles"""

            for param, item in zip(params, dict_param_to_optimize):
                dict_param_to_optimize[item] = param

            self._modify_kinetics(dict_param_to_optimize, optimizer_left_op)

            if self._nset == 1:
                t = Cexp.y.data
                Carray = Cexp.data
            else:
                t = [C.y.data for C in Cexp]
                Carray = np.concatenate([C.data for C in Cexp])

            Chat = self.integrate(
                t,
                return_NDDataset=False,
                method=ivp_solver_method,
                k_dt=k_dt,
                left_op=C_op,
            )

            if self._nset > 1:
                Chat = np.concatenate([C for C in Chat])

            return np.sum(np.square(Carray[:, iexp] - Chat[:, i2iexp]))

        # optimizer (kw)arguments:
        # ... parameters for scipy.minimize
        optimizer_method = optimizer_kwargs.get("Method", "Nelder-Mead")
        optimizer_bounds = optimizer_kwargs.get("bounds", None)
        optimizer_tol = optimizer_kwargs.get("tol", None)
        optimizer_options = optimizer_kwargs.get("options", {"disp": True})
        # optimizer_callback = optimizer_kwargs.get("callback", None)
        # ... other parameters
        optimizer_left_op = optimizer_kwargs.get("left_op", None)

        # ivp solver (kw)arguments:
        # ... parameters for integrate.ivp_solve
        ivp_solver_method = ivp_solver_kwargs.get("method", "LSODA")
        # ... other parameters
        ivp_solver_k_dt = ivp_solver_kwargs.get("k_dt", None)
        ivp_solver_left_op = ivp_solver_kwargs.get("left_op", None)

        # get x0
        x0 = np.zeros(len(dict_param_to_optimize))
        for i, param in enumerate(dict_param_to_optimize):
            x0[i] = dict_param_to_optimize[param]

        if optimizer_options["disp"]:
            init_val = objective(
                x0,
                Cexp,
                iexp,
                i2iexp,
                dict_param_to_optimize,
                optimizer_left_op,
                ivp_solver_method,
                ivp_solver_k_dt,
                ivp_solver_left_op,
            )
            print("Optimization of the parameters.")
            print(f"         Initial parameters: {x0}")
            print(f"         Initial function value: {init_val:f}")
        tic = datetime.datetime.now(datetime.timezone.utc)

        optim_res = minimize(
            objective,
            x0,
            args=(
                Cexp,
                iexp,
                i2iexp,
                dict_param_to_optimize,
                optimizer_left_op,
                ivp_solver_method,
                ivp_solver_k_dt,
                ivp_solver_left_op,
            ),
            method=optimizer_method,
            bounds=optimizer_bounds,
            tol=optimizer_tol,
            options=optimizer_options,
        )
        toc = datetime.datetime.now(datetime.timezone.utc)

        if optimizer_options["disp"]:
            print(f"         Optimization time: {toc - tic}")
            print(f"         Final parameters: {optim_res['x']}")

        # compute the final concentration profiles
        if self._nset == 1:
            t = Cexp.y.data
        else:
            t = [C.y.data for C in Cexp]

        Ckin = self.integrate(
            t,
            return_NDDataset=False,
            method=ivp_solver_method,
            k_dt=ivp_solver_k_dt,
            left_op=ivp_solver_left_op,
        )

        for i, param in enumerate(dict_param_to_optimize):
            dict_param_to_optimize[param] = optim_res["x"][i]

        return Ckin, (iexp, i2iexp, dict_param_to_optimize), optim_res


# ------------------------------------------------------------------------------------
# CANTERA UTILITIES
# ------------------------------------------------------------------------------------
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

        if method.upper() in SCIPY_MINIMIZE_METHODS:
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
