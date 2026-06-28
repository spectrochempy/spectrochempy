# spectrochempy-carroucell

Carroucell experiment reader for SpectroChemPy.

Provides a reader for Carroucell multi-experiment data files generated
by the LCS laboratory instrument, exposed under the `scp.carroucell` namespace.

## Installation

```bash
pip install spectrochempy[carroucell]
```

## Usage

```python
import spectrochempy as scp

dataset = scp.carroucell.read("path/to/data.carroucell")
```

The compatibility alias `scp.read_carroucell(...)` is kept for historical
usage.

## Development

```bash
pip install -e .
pip install -e plugins/spectrochempy-carroucell
cd plugins/spectrochempy-carroucell
python -m pytest tests/
```
