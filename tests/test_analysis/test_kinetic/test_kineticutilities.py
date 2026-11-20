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
        assert C.history_entries[-1]["operation"] is None
        assert C.history_entries[-1]["message"] == "Integrated action-mass kinetics"

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


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


def test_jacobian_with_stoichiometric_coefficient_above_one():
    """The analytical jacobian should match finite differences of the rates."""
    kin = ku.ActionMassKinetics(
        ["B -> C", "A -> 2B", "2C -> D"],
        {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0},
        [5.0, 0.1, 0.7],
    )
    k = kin._arrhenius
    C0 = np.array([0.6, 0.3, 0.2, 0.1])

    def rates(C):
        return np.array(
            eval(kin._write_production_rates(), {"k": k, "C": C}), dtype=float
        )

    jac = np.array(eval(kin._write_jacobian(), {"k": k, "C": C0}), dtype=float)
    eps = 1e-7
    numerical = np.empty_like(jac)
    for j in range(len(C0)):
        dC = np.zeros_like(C0)
        dC[j] = eps
        numerical[:, j] = (rates(C0 + dC) - rates(C0 - dC)) / (2 * eps)
    np.testing.assert_allclose(jac, numerical, rtol=1e-6, atol=1e-9)


def test_fit_uses_lowercase_method(monkeypatch):
    """`method` in optimizer_kwargs should select the optimizer."""
    used = []
    real_minimize = ku.minimize

    def spy(*args, **kwargs):
        used.append(kwargs["method"])
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(ku, "minimize", spy)
    time = np.arange(0, 10)
    C_exp = ku.ActionMassKinetics(
        _reactions(), _species_concentrations(), [1.0, 0.3]
    ).integrate(time)
    kin = ku.ActionMassKinetics(_reactions(), _species_concentrations(), [0.5, 0.5])
    kin.fit_to_concentrations(
        C_exp,
        [0, 1, 2],
        [0, 1, 2],
        {"k[0]": 0.5, "k[1]": 0.5},
        optimizer_kwargs={"method": "L-BFGS-B", "options": {"disp": False}},
    )
    kin.fit_to_concentrations(
        C_exp,
        [0, 1, 2],
        [0, 1, 2],
        {"k[0]": 0.5, "k[1]": 0.5},
        optimizer_kwargs={"Method": "Powell", "options": {"disp": False}},
    )
    assert used == ["L-BFGS-B", "Powell"]


def test_fit_left_op_ties_parameters_and_sets_optimum():
    """left_op should tie parameters and the model should hold the optimum."""
    reactions = ["A -> B", "B -> C", "C -> D"]
    conc = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    time = np.linspace(0, 10, 41)
    C_exp = ku.ActionMassKinetics(reactions, conc, [1.0, 0.3, 0.3]).integrate(time)
    kin = ku.ActionMassKinetics(reactions, conc, [0.5, 0.1, 0.9])
    left_op = np.array([[1.0, 0, 0], [0, 1, 0], [0, 1, 0]])  # k[2] := k[1]
    Ckin, _, res = kin.fit_to_concentrations(
        C_exp,
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        {"k[0]": 0.5, "k[1]": 0.1},
        optimizer_kwargs={"left_op": left_op, "options": {"disp": False}},
    )
    np.testing.assert_allclose(res.x, [1.0, 0.3], rtol=1e-3)
    np.testing.assert_allclose(kin._arrhenius, [res.x[0], res.x[1], res.x[1]])
    np.testing.assert_allclose(np.sum((C_exp.data - Ckin) ** 2), res.fun)


def test_fit_non_idempotent_left_op_is_not_applied_cumulatively():
    """left_op should be applied once per evaluation, not accumulated."""
    reactions = ["A -> B", "B -> C"]
    conc = {"A": 1.0, "B": 0.0, "C": 0.0}
    time = np.linspace(0, 10, 21)
    C_exp = ku.ActionMassKinetics(reactions, conc, [1.0, 0.6]).integrate(time)
    kin = ku.ActionMassKinetics(reactions, conc, [0.5, 0.5])
    left_op = np.array([[1.0, 0.0], [0.6, 0.0]])  # k[1] := 0.6 k[0]
    _, _, res = kin.fit_to_concentrations(
        C_exp,
        [0, 1, 2],
        [0, 1, 2],
        {"k[0]": 0.5},
        optimizer_kwargs={
            "left_op": left_op,
            "method": "Powell",
            "options": {"disp": False},
        },
    )
    np.testing.assert_allclose(res.x, [1.0], rtol=1e-3)
    np.testing.assert_allclose(kin._arrhenius, [res.x[0], 0.6 * res.x[0]])


def test_callable_temperature_without_k_dt():
    """k(T(t)) should be evaluated at each step when k_dt is not given."""
    arrhenius = [[1.0e8, 52.0e3], [1.0e8, 50.0e3]]
    time = np.linspace(0, 10, 21)

    def T(t):
        return 298.0 + 2.0 * t

    kin = ku.ActionMassKinetics(_reactions(), _species_concentrations(), arrhenius, T=T)
    C = kin.integrate(time, rtol=1e-8, atol=1e-10)
    C_jac = kin.integrate(time, rtol=1e-8, atol=1e-10, use_jac=True, method="BDF")
    C_grid = kin.integrate(time, k_dt=1e-3, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(C.data, C_jac.data, atol=1e-6)
    np.testing.assert_allclose(C.data, C_grid.data, atol=1e-3)
    np.testing.assert_allclose(C.data.sum(axis=1), 1.0, atol=1e-6)


def test_callable_temperature_requires_arrhenius_parameters():
    """Plain rate constants cannot be combined with T(t)."""
    kin = ku.ActionMassKinetics(
        _reactions(), _species_concentrations(), [1.0, 0.3], T=lambda t: 300.0 + t
    )
    with pytest.raises(ValueError, match="requires Arrhenius parameters"):
        kin.integrate(np.arange(0, 5))
