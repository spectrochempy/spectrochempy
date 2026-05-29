# ruff: noqa: PLC0415, D200
"""
Legacy Cantera functions — preserved but not exposed through plugin hooks.

These were part of the initial plugin but are deferred until a stable
API for equilibrium, flame, and spectral coupling is defined.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrochempy import NDDataset


def equilibrium_composition(
    mechanism: str,
    temperature: float,
    pressure: float,
    reactants: dict[str, float],
) -> dict:
    """
    Compute chemical equilibrium composition for a given mechanism and conditions.
    """
    import cantera as ct
    import numpy as np

    import spectrochempy as scp

    gas = ct.Solution(mechanism)
    gas.TPX = temperature, pressure, reactants
    gas.equilibrate("TP")
    species_names = gas.species_names
    mole_fracs = gas.X
    mf_ds = scp.NDDataset(
        np.array([mole_fracs]),
        title="Equilibrium mole fractions",
    )
    return {
        "species": species_names,
        "mole_fractions": mf_ds,
        "temperature": temperature,
        "pressure": pressure,
        "n_species": len(species_names),
    }


def reactor_profile(
    mechanism: str,
    initial_conditions: dict,
    residence_time: float = 1.0,
    n_points: int = 100,
) -> dict:
    """
    Simulate an ideal reactor time profile using a Cantera constant-volume reactor.
    """
    import cantera as ct
    import numpy as np

    import spectrochempy as scp

    gas = ct.Solution(mechanism)
    X = initial_conditions.get("X", {"CH4": 1.0, "O2": 2.0, "N2": 7.52})
    gas.TPX = initial_conditions["T"], initial_conditions["P"], X
    reactor = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([reactor])
    times = np.linspace(0, residence_time, n_points)
    temperatures = np.zeros(n_points)
    mole_fracs = np.zeros((n_points, gas.n_species))
    for i, t in enumerate(times):
        sim.advance(t)
        temperatures[i] = reactor.T
        mole_fracs[i] = gas.X
    mf_ds = scp.NDDataset(mole_fracs, title="Reactor mole fractions")
    return {
        "time": times,
        "temperature": temperatures,
        "species": gas.species_names,
        "mole_fractions": mf_ds,
    }


def flame_speed(
    mechanism: str,
    temperature: float = 300.0,
    pressure: float = 101325.0,
    equivalence_ratio: float = 1.0,
    fuel: str = "CH4",
    oxidizer: str = "O2:1.0, N2:3.76",
) -> dict:
    """
    Calculate laminar flame speed using Cantera's free flame.
    """
    import cantera as ct

    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)
    gas.TP = temperature, pressure
    flame = ct.FreeFlame(gas)
    flame.set_refine_criteria(ratio=3, slope=0.3, curve=0.3)
    flame.solve(loglevel=0, auto=True)
    return {
        "flame_speed": float(flame.velocity[0]),
        "temperature_profile": flame.T,
        "species_profiles": {
            name: flame.X[:, i] for i, name in enumerate(gas.species_names)
        },
        "grid": flame.grid,
    }


def thermo_properties(
    mechanism: str,
    species: str = "CH4",
    T_min: float = 300.0,
    T_max: float = 2000.0,
    n_points: int = 50,
    pressure: float = 101325.0,
) -> dict:
    """
    Compute thermodynamic properties (Cp, H, S, G) over a temperature range.
    """
    import cantera as ct
    import numpy as np

    import spectrochempy as scp

    gas = ct.Solution(mechanism)
    temperatures = np.linspace(T_min, T_max, n_points)
    cp = np.zeros(n_points)
    h = np.zeros(n_points)
    s = np.zeros(n_points)
    for i, T in enumerate(temperatures):
        gas.TP = T, pressure
        idx = gas.species_index(species)
        cp[i] = gas.standard_cp_R[idx] * ct.gas_constant
        h[i] = gas.partial_molar_enthalpies[idx]
        s[i] = gas.partial_molar_entropies[idx]
    g = h - temperatures * s
    return {
        "temperature": scp.NDDataset(temperatures, title="Temperature (K)"),
        "cp": scp.NDDataset(cp, title=f"Cp({species}) J/mol/K"),
        "enthalpy": scp.NDDataset(h, title=f"H({species}) J/mol"),
        "entropy": scp.NDDataset(s, title=f"S({species}) J/mol/K"),
        "gibbs": scp.NDDataset(g, title=f"G({species}) J/mol"),
    }


def kinetic_sensitivity(
    mechanism: str,
    temperature: float = 1000.0,
    pressure: float = 101325.0,
    reactants: dict | None = None,
) -> dict:
    """
    Perform first-order kinetic sensitivity analysis.
    """
    import cantera as ct
    import numpy as np

    gas = ct.Solution(mechanism)
    if reactants is None:
        reactants = {"CH4": 1.0, "O2": 2.0, "N2": 7.52}
    gas.TPX = temperature, pressure, reactants
    reactor = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([reactor])
    for i in range(gas.n_reactions):
        reactor.add_sensitivity_reaction(i)
    sim.advance(1e-3)
    sensitivities = np.array(
        [reactor.sensitivity(i, 0) for i in range(gas.n_reactions)]
    )
    return {
        "reactions": [str(r) for r in gas.reactions()],
        "sensitivities": sensitivities,
        "net_rates": gas.net_rates_of_progress.copy(),
    }


def spectral_coupling(
    mechanism: str,
    dataset: NDDataset,
    temperature_range: tuple[float, float] = (300.0, 1500.0),
    n_points: int = 20,
) -> dict:
    """
    Couple thermodynamic states with spectroscopic data.
    """
    import cantera as ct
    import numpy as np

    import spectrochempy as scp

    gas = ct.Solution(mechanism)
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n_points)
    n_species = gas.n_species
    evolution = np.zeros((n_points, n_species))
    for i, T in enumerate(temperatures):
        gas.TP = T, 101325.0
        gas.equilibrate("TP")
        evolution[i] = gas.X
    evo_ds = scp.NDDataset(evolution, title="Species evolution vs T")
    return {
        "temperatures": temperatures,
        "species": gas.species_names,
        "species_evolution": evo_ds,
        "spectral_coordinates": dataset.coordset,
    }


def read_cantera_mechanism(path: str) -> dict:
    """
    Read a Cantera mechanism file and return summary info.
    """
    import cantera as ct

    gas = ct.Solution(path)
    return {
        "n_species": gas.n_species,
        "n_reactions": gas.n_reactions,
        "species": gas.species_names,
        "elements": gas.element_names,
    }
