# spectrochempy-carroucell

Carroucell experiment reader for SpectroChemPy.

Provides a reader for Carroucell multi-experiment data files generated
by the LCS laboratory instrument, exposed under the `scp` namespace.

## Installation

```bash
pip install spectrochempy[carroucell]
```

## Usage

```python
import spectrochempy as scp

dataset = scp.read_carroucell("path/to/data.carroucell")
```

This plugin requires SpectroChemPy.

## Development

```bash
pip install -e .
pip install -e plugins/spectrochempy-carroucell
cd plugins/spectrochempy-carroucell
python -m pytest tests/
```
