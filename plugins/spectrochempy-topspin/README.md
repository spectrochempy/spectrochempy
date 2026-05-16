# spectrochempy-topspin

Bruker TopSpin NMR reader plugin for SpectroChemPy.

Reads Bruker NMR data files: FID, SER, and processed data.

## Installation

```bash
pip install spectrochempy-topspin
```

## Usage

```python
import spectrochempy as scp

# Read a TopSpin FID file
dataset = scp.read_topspin("path/to/1/fid")

# Read a processed 2D spectrum
dataset = scp.read_topspin("path/to/2rr")

# Read from a directory
dataset = scp.read_topspin("path/to/experiment", expno=1, procno=1)
```

## Development

```bash
pip install -e .
python -m pytest tests/ -v
```
