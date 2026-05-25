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
```

The Cantera plugin currently exposes package-level simulation and analysis
functions only. It does not install `dataset.cantera.*` accessors because the
current equilibrium and reactor workflows require explicit thermodynamic inputs
rather than deriving them unambiguously from an `NDDataset`.

## Development

```bash
pip install -e .
pip install -e plugins/spectrochempy-cantera
cd plugins/spectrochempy-cantera
python -m pytest tests/
```
