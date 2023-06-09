# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
import numpy as np
import pytest

from spectrochempy import show

# import spectrochempy
from spectrochempy.analysis.kinetics import kineticutilities as cu


@pytest.mark.skipif(
    cu._cantera_is_not_available(), reason="Cantera must be installed first"
)
def test_cu(monkeypatch):

    # availability of cantera (# should be installed if the test wa not skipped)
    assert not cu._cantera_is_not_available()

    # simulate abscense of cantera
    with monkeypatch.context() as m:
        m.setattr(cu, "ct", None)
        assert cu._cantera_is_not_available()

    # context restored with ct = cantera
    assert not cu._cantera_is_not_available()


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
    species = list(left_expected.keys()) + list(right_expected.keys())
    left, right = cu._interpret_equation(test_str, species)

    assert left_expected == left
    assert right_expected == right


def test_ABC():
    reactions = ("A -> B", "B -> C")
    species_concentrations = {"A": 1.0, "B": 0.0, "C": 0.0}
    time = np.arange(10)
    k_exp = np.array(((1.0, 50.0), (1.0, 50.0)))
    kin_exp = cu.ActionMassKinetics(reactions, species_concentrations, k_exp, T=298.0)
    C_exp = kin_exp.integrate(time)

    k_guess = np.array(((1.5, 50.0), (1.0, 55.0)))
    kin_guess = cu.ActionMassKinetics(reactions, species_concentrations, k_guess)
    res = kin_guess.fit_to_concentrations(
        C_exp,
        iexp=[0, 1, 2],
        i2iexp=[0, 1, 2],
        dict_param_to_optimize={
            "k[0].A": 1.1,
            "k[1].Ea": 49.0,
        },
        optimizer_kwargs={"xtol": 0.01, "ftol": 0.1},
    )
    C_opt = kin_guess.integrate(time)

    print(res[2]["x"])
    assert max(res[2]["x"] - [1.0, 50.0]) < 1e-3

    _ = C_exp.T.plot(markers="o")
    _ = C_opt.T.plot(clear=False)

    show()
