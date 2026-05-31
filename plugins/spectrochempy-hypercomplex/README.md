# spectrochempy-hypercomplex

Hypercomplex / quaternion support plugin for SpectroChemPy.

Provides quaternion-based dataset types, quaternion-aware processing
operations, and utilities for working with hypercomplex NMR data.

## Installation

```bash
pip install spectrochempy[hypercomplex]
```

## Usage

```python
import spectrochempy as scp

# Enable hypercomplex support.
dataset = scp.read_topspin("path/to/2rr")
dataset.set_quaternion()
```

This plugin requires SpectroChemPy.

## Development

```bash
pip install -e .
pip install -e plugins/spectrochempy-hypercomplex
cd plugins/spectrochempy-hypercomplex
python -m pytest tests/
```
