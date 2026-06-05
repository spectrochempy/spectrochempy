# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the kinetic utilities module.

Uses deterministic synthetic kinetic data throughout.
No plotting, no debug print calls, no network dependencies.
"""

import numpy as np
import pytest

from spectrochempy.analysis.kinetic import kineticutilities as ku


# ---------------------------------------------------------------------------
# Cantera plugin availability
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    ku._cantera_is_not_available(),
    reason="requires the experimental spectrochempy-cantera plugin with Cantera",
)
def test_cu():
    """Cantera plugin should be available if not skipped."""
    assert not ku._cantera_is_not_available()


# ---------------------------------------------------------------------------
# Reaction equation regex parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_str, left_expected, right_expected",
    [
        ("A ->E", {"A": 1}, {"E": 1}),
        ("A +b   -> 43 C + 2 D", {"A": 1, "b": 1}, {"C": 43, "D": 2}),
        ("A+V+H->X+Y", {"A": 1, "V": 1, "H": 1}, {"X": 1, "Y": 1}),
        ("A->2E+F", {"A": 1}, {"E": 2, "F": 1}),
        ("A   +  s->2E+4F", {"A": 1, "s": 1}, {"E": 2, "F": 4}),
        ("A+3V->X+Y", {"A": 1, "V": 3}, {"X": 1, "Y": 1}),
        ("2+V->X+5Y", {"V": 1}, {"X": 1, "Y": 5}),
        ("AT+CF->SCP+2X", {"AT": 1, "CF": 1}, {"SCP": 1, "X": 2}),
        ("4 NH3 + 7 O2 → 4 NO2 + 6 H2O", {"NH3": 4, "O2": 7}, {"NO2": 4, "H2O": 6}),
    ],
)
def test_equations_regex(test_str, left_expected, right_expected):
    """Reaction equation parsing should correctly separate reactants and products."""
    species = list(left_expected.keys()) + list(right_expected.keys())
    left, right = ku._interpret_equation(test_str, species)
    assert left_expected == left
    assert right_expected == right


# ---------------------------------------------------------------------------
# Kinetic model construction
# ---------------------------------------------------------------------------


def _reactions():
    return ("A -> B", "B -> C")


def _species_concentrations():
    return {"A": 1.0, "B": 0.0, "C": 0.0}


def _k_exp():
    return np.array(((1.0e8, 52.0e3), (1.0e8, 50.0e3)))


def _k_guess():
    return np.array(((1.5e8, 52.0e3), (1.0e8, 55.0e3)))


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------


class TestActionMassKinetics:
    """ActionMassKinetics model fitting."""

    def test_initialization(self):
        """Model construction should set expected attributes."""
        kin = ku.ActionMassKinetics(
            _reactions(), _species_concentrations(), _k_exp(), T=298.0
        )
        assert kin.n_reactions == 2
        assert kin.n_species == 3
        assert kin.species == ["A", "B", "C"]
        assert kin.A.shape == (2, 3)
        assert kin.B.shape == (2, 3)

    def test_integrate_isothermal(self):
        """Isothermal integration should return finite NDdata."""
        kin = ku.ActionMassKinetics(
            _reactions(), _species_concentrations(), _k_exp(), T=298.0
        )
        time = np.arange(0, 10)
        C = kin.integrate(time)
        assert C.shape == (10, 3)
        assert np.all(np.isfinite(C.data))

    def test_fit_isothermal(self):
        """Isothermal fitting should recover true Arrhenius parameters."""
        reactions = _reactions()
        species_concentrations = _species_concentrations()
        k_exp = _k_exp()
        time = np.arange(0, 10)

        kin_exp = ku.ActionMassKinetics(
            reactions, species_concentrations, k_exp, T=298.0
        )
        C_exp = kin_exp.integrate(time)

        k_guess = _k_guess()
        kin_guess = ku.ActionMassKinetics(reactions, species_concentrations, k_guess)
        res = kin_guess.fit_to_concentrations(
            C_exp,
            iexp=[0, 1, 2],
            i2iexp=[0, 1, 2],
            dict_param_to_optimize={
                "k[0].A": 1.1e8,
                "k[1].Ea": 49.0e3,
            },
            optimizer_kwargs={"xtol": 0.01, "ftol": 0.1},
        )

        assert isinstance(res, tuple) and len(res) == 3
        assert res[2].success
        assert max(res[2]["x"] - [1.0e8, 50.0e3]) < 1e-3

        C_opt = kin_guess.integrate(time)
        assert C_opt.shape == (10, 3)
        assert np.all(np.isfinite(C_opt.data))

    def test_fit_non_isothermal_single(self):
        """Non-isothermal fitting (single condition) should recover parameters."""
        reactions = _reactions()
        species_concentrations = _species_concentrations()
        k_exp = _k_exp()
        time = np.arange(0, 10)

        def T(t):
            T = np.zeros_like(t)
            for i, ti in enumerate(t):
                if ti < 2.5:
                    T[i] = 298.0
                elif ti < 7.5:
                    T[i] = 298.0 + (308.0 - 298.0) * ti
                else:
                    T[i] = 308.0
            return T

        kin = ku.ActionMassKinetics(reactions, species_concentrations, k_exp, T=T)
        C_exp = kin.integrate(time, k_dt=0.01)
        assert C_exp.shape == (10, 3)
        assert np.all(np.isfinite(C_exp.data))

        k_guess = _k_guess()
        kin_guess = ku.ActionMassKinetics(
            reactions, species_concentrations, k_guess, T=T
        )
        res = kin_guess.fit_to_concentrations(
            C_exp,
            iexp=[0, 1, 2],
            i2iexp=[0, 1, 2],
            dict_param_to_optimize={
                "k[0].A": 1.1e8,
                "k[1].Ea": 49.0e3,
            },
            optimizer_kwargs={"xtol": 0.01, "ftol": 0.1},
            ivp_solver_kwargs={"k_dt": 0.1},
        )

        assert isinstance(res, tuple) and len(res) == 3
        assert res[2].success
        assert max((res[2]["x"] - [1.0e8, 50.0e3]) / [1.0e8, 50.0e3]) < 0.05

        C_opt = kin_guess.integrate(time, k_dt=0.1)
        assert C_opt.shape == (10, 3)
        assert np.all(np.isfinite(C_opt.data))

    def test_fit_non_isothermal_multi(self):
        """Non-isothermal fitting (multiple conditions) should recover parameters."""
        reactions = _reactions()
        k_exp = _k_exp()
        time = (np.arange(0, 10), np.arange(0, 10, 0.5))

        def T(t):
            T = np.zeros_like(t)
            for i, ti in enumerate(t):
                if ti < 2.5:
                    T[i] = 298.0
                elif ti < 7.5:
                    T[i] = 298.0 + (308.0 - 298.0) * ti
                else:
                    T[i] = 308.0
            return T

        def T2(t):
            T = np.zeros_like(t)
            for i, ti in enumerate(t):
                if ti < 2.5:
                    T[i] = 298.0
                elif ti < 7.5:
                    T[i] = 298.0 + (320.0 - 298.0) * ti
                else:
                    T[i] = 320.0
            return T

        species_concentrations = (
            {"A": 1.0, "B": 0.0, "C": 0.0},
            {"A": 1.0, "B": 0.2, "C": 0.0},
        )

        kin = ku.ActionMassKinetics(reactions, species_concentrations, k_exp, T=(T, T2))
        C_exp = kin.integrate(time, k_dt=0.01)
        assert isinstance(C_exp, list) and len(C_exp) == 2
        for c in C_exp:
            assert np.all(np.isfinite(c.data))
        assert C_exp[0].shape[1] == 3
        assert C_exp[1].shape[1] == 3

        k_guess = _k_guess()
        kin_guess = ku.ActionMassKinetics(
            reactions, species_concentrations, k_guess, T=(T, T2)
        )
        res = kin_guess.fit_to_concentrations(
            C_exp,
            iexp=[0, 1, 2],
            i2iexp=[0, 1, 2],
            dict_param_to_optimize={
                "k[0].A": 1.1e8,
                "k[1].Ea": 49.0e3,
            },
            optimizer_kwargs={"xtol": 0.01, "ftol": 0.1},
            ivp_solver_kwargs={"k_dt": 0.1},
        )

        assert isinstance(res, tuple) and len(res) == 3
        assert res[2].success
        assert max((res[2]["x"] - [1.0e8, 50.0e3]) / [1.0e8, 50.0e3]) < 0.05

        C_opt = kin_guess.integrate(time, k_dt=0.1)
        assert isinstance(C_opt, list) and len(C_opt) == 2
        for c in C_opt:
            assert np.all(np.isfinite(c.data))
