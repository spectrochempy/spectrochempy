# spectrochempy-cantera

Cantera-based thermodynamics and reactive chemistry for SpectroChemPy.

Provides equilibrium computation, reactor simulation, kinetic analysis,
and spectral interpretation support coupled with Cantera.

## Installation

```bash
pip install spectrochempy[cantera]
```

## Usage

```python
import spectrochempy as scp

# Package-level plugin APIs are available under scp.cantera after installation.
result = scp.cantera.equilibrium(
    mechanism="gri30.yaml",
    temperature=1000,
    pressure=101325,
    reactants={"CH4": 1.0, "O2": 2.0},
)

profile = scp.cantera.reactor_profile(
    mechanism="gri30.yaml",
    initial_conditions={"T": 1000, "P": 101325},
    residence_time=0.1,
)

# Operations using an existing dataset are exposed through dataset accessors.
eq_from_dataset = dataset.cantera.equilibrium("gri30.yaml")
```

The legacy flat accessor `dataset.cantera_equilibrium(...)` is kept for
compatibility.

## Development

```bash
pip install -e .
pip install -e plugins/spectrochempy-cantera
cd plugins/spectrochempy-cantera
python -m pytest tests/
```
