# spectrochempy-cantera

**EXPERIMENTAL PLUGIN** — Not officially supported. API subject to change.

Cantera-based thermodynamics and reactive chemistry for SpectroChemPy.

Currently provides:
- ``PFR``: Plug-flow reactor model using Cantera.

## Installation

```bash
pip install spectrochempy-cantera
```

## Usage

```python
import spectrochempy as scp

# The PFR class is available under the scp.cantera namespace.
PFR = scp.cantera.PFR
```

Or import directly from the plugin package:

```python
from spectrochempy_cantera import PFR
```

## Development

```bash
pip install -e plugins/spectrochempy-cantera
cd plugins/spectrochempy-cantera
python -m pytest tests/
```
