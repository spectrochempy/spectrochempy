# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""
spectrochempy-cantera — Cantera-based thermodynamics and reactive chemistry
for SpectroChemPy.

Provides equilibrium computation, reactor simulation, kinetic analysis,
and spectral interpretation support coupled with Cantera.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin

if TYPE_CHECKING:
    from spectrochempy import NDDataset


class CanteraPlugin(SpectroChemPyPlugin):
    """Cantera thermodynamics plugin for SpectroChemPy."""

    name = "cantera"
    version = "0.1.0"
    description = "Thermodynamic equilibrium, reactor simulation, and kinetic analysis via Cantera"
    spectrochempy_min_version = "0.8.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [
        PluginCapability.SIMULATION,
        PluginCapability.ANALYSIS,
        PluginCapability.ACCESSOR,
    ]
    requires = ["cantera"]

    # ------------------------------------------------------------------
    # Declarative hooks
    # ------------------------------------------------------------------

    def register_simulations(self) -> list[dict]:
        return [
            {
                "name": "equilibrium",
                "func": equilibrium_composition,
                "description": "Chemical equilibrium composition calculation",
            },
            {
                "name": "reactor_profile",
                "func": reactor_profile,
                "description": "Ideal reactor time/temperature profile simulation",
            },
            {
                "name": "flame_speed",
                "func": flame_speed,
                "description": "Laminar flame speed calculation",
            },
        ]

    def register_analyses(self) -> list[dict]:
        return [
            {
                "name": "thermo_properties",
                "func": thermo_properties,
                "description": "Thermodynamic properties (Cp, H, S, G) over temperature range",
            },
            {
                "name": "kinetic_sensitivity",
                "func": kinetic_sensitivity,
                "description": "First-order kinetic sensitivity analysis",
            },
            {
                "name": "spectral_coupling",
                "func": spectral_coupling,
                "description": "Couple thermodynamic states with spectroscopic signature",
            },
        ]

    def register_readers(self) -> list[dict]:
        return [
            {
                "name": "cantera_yaml",
                "func": read_cantera_mechanism,
                "description": "Read a Cantera YAML/CTI mechanism file into a workspace",
            },
        ]

    # ------------------------------------------------------------------
    # Dataset accessor methods
    # ------------------------------------------------------------------

    def register_accessors(self) -> list[dict]:
        return [
            {
                "name": "cantera_equilibrium",
                "func": _ndd_cantera_equilibrium,
                "description": "Compute equilibrium composition from dataset conditions",
            },
        ]


# ------------------------------------------------------------------
# Dataset accessor implementation
# ------------------------------------------------------------------


def _ndd_cantera_equilibrium(
    self: NDDataset, mechanism: str, temperature: float | None = None
) -> dict:
    """Compute equilibrium composition using dataset temperature/pressure metadata."""
    import numpy as np

    T = temperature or float(np.mean(self.data)) if hasattr(self, "data") else 300.0
    return equilibrium_composition(
        mechanism=mechanism,
        temperature=T,
        pressure=101325.0,
        reactants={"H2": 1.0, "O2": 0.5},
    )


# ------------------------------------------------------------------
# Simulation functions (deferred cantera imports)
# ------------------------------------------------------------------


def equilibrium_composition(
    mechanism: str,
    temperature: float,
    pressure: float,
    reactants: dict[str, float],
) -> dict:
    """
    Compute chemical equilibrium composition for a given mechanism and conditions.

    Parameters
    ----------
    mechanism : str
        Cantera mechanism file (e.g. ``"gri30.yaml"``).
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in Pa.
    reactants : dict
        Reactant species and their mole fractions (e.g. ``{"CH4": 1.0, "O2": 2.0}``).

    Returns
    -------
    dict with keys ``species`` (list of names), ``mole_fractions`` (NDDataset),
    ``temperature``, ``pressure``, ``n_species``.
    """
    import cantera as ct

    gas = ct.Solution(mechanism)
    gas.TPX = temperature, pressure, reactants

    # Equilibrate at constant T and P
    gas.equilibrate("TP")

    species_names = gas.species_names
    mole_fracs = gas.X

    import numpy as np

    import spectrochempy as scp

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
    Simulate an ideal reactor time profile.

    Uses a Cantera ideal gas constant-volume reactor.

    Parameters
    ----------
    mechanism : str
        Cantera mechanism file.
    initial_conditions : dict
        Must contain ``"T"`` (K) and ``"P"`` (Pa). Optionally ``"X"`` (dict of mole fractions).
    residence_time : float
        Total simulation time in seconds.
    n_points : int
        Number of output time points.

    Returns
    -------
    dict with keys ``time``, ``temperature``, ``species``, ``mole_fractions``.
    """
    import cantera as ct
    import numpy as np

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

    import spectrochempy as scp

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

    Parameters
    ----------
    mechanism : str
        Cantera mechanism file.
    temperature : float
        Unburned gas temperature in K.
    pressure : float
        Pressure in Pa.
    equivalence_ratio : float
        Fuel/oxidizer equivalence ratio.
    fuel : str
        Fuel species name.
    oxidizer : str
        Oxidizer composition (Cantera format).

    Returns
    -------
    dict with keys ``flame_speed`` (m/s), ``temperature_profile``,
    ``species_profiles``.
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


# ------------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------------


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

    Parameters
    ----------
    mechanism : str
        Cantera mechanism file.
    species : str
        Species name.
    T_min, T_max : float
        Temperature range in K.
    n_points : int
        Number of temperature points.

    Returns
    -------
    dict with keys ``temperature``, ``cp``, ``enthalpy``, ``entropy``,
    ``gibbs`` (all as NDDataset).
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

    Parameters
    ----------
    mechanism : str
        Cantera mechanism file.
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in Pa.
    reactants : dict or None
        Reactant composition.

    Returns
    -------
    dict with keys ``reactions``, ``sensitivities``, ``net_rates``.
    """
    import cantera as ct
    import numpy as np

    gas = ct.Solution(mechanism)
    if reactants is None:
        reactants = {"CH4": 1.0, "O2": 2.0, "N2": 7.52}
    gas.TPX = temperature, pressure, reactants

    reactor = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([reactor])

    # Attach sensitivity analysis
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

    Computes equilibrium composition at multiple temperatures and
    returns a joint structure that pairs spectroscopic coordinates
    with thermodynamic states.

    Parameters
    ----------
    mechanism : str
        Cantera mechanism file.
    dataset : NDDataset
        Spectroscopic dataset whose coordinates define the coupling.
    temperature_range : (float, float)
        (T_min, T_max) in K.
    n_points : int
        Number of temperature points.

    Returns
    -------
    dict with keys ``temperatures``, ``species_evolution``,
    ``spectral_coordinates``.
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


# ------------------------------------------------------------------
# Reader functions
# ------------------------------------------------------------------


def read_cantera_mechanism(path: str) -> dict:
    """
    Read a Cantera mechanism file and return summary info.

    Parameters
    ----------
    path : str
        Path to a Cantera YAML or CTI mechanism file.

    Returns
    -------
    dict with keys ``n_species``, ``n_reactions``, ``species``, ``elements``.
    """
    import cantera as ct

    gas = ct.Solution(path)
    return {
        "n_species": gas.n_species,
        "n_reactions": gas.n_reactions,
        "species": gas.species_names,
        "elements": gas.element_names,
    }


# Export accessor names so NDDataset can discover them
__dataset_methods__ = ["cantera_equilibrium"]

# Re-export PFR from the _pfr module
