# spectrochempy-cantera

Cantera-based thermodynamics and reactive chemistry for SpectroChemPy.

Provides equilibrium computation, reactor simulation, kinetic analysis,
and spectral interpretation support coupled with Cantera.

## Installation

```bash
pip install spectrochempy-cantera
```

## Usage

```python
import spectrochempy as scp
from spectrochempy_cantera import equilibrium_composition, reactor_profile

# Equilibrium calculation
species, conditions = equilibrium_composition(
    mechanism="gri30.yaml",
    temperature=1000,
    pressure=101325,
    reactants={"CH4": 1.0, "O2": 2.0},
)

# Reactor simulation
profile = reactor_profile(
    mechanism="gri30.yaml",
    initial_conditions={"T": 1000, "P": 101325},
    residence_time=0.1,
)
```

## Development

```bash
pip install -e .
python -m pytest tests/
```
