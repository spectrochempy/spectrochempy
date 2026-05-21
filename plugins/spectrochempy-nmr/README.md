# spectrochempy-nmr

NMR plugin for SpectroChemPy.

This package is the home for NMR-specific readers and tools that are useful in
SpectroChemPy but should not live in the core package. It currently contributes
the Bruker TopSpin reader, exposed as `scp.nmr.read_topspin(...)` and registered
under the reader name `topspin`.

Future NMR readers or processing helpers can be added here without creating a
new plugin package for each vendor format.

## Installation

```bash
pip install spectrochempy[nmr]
```

For development inside the SpectroChemPy repository:

```bash
pip install -e .
pip install -e plugins/spectrochempy-nmr
```

## Usage

```python
import spectrochempy as scp

dataset = scp.nmr.read_topspin("path/to/1/fid")
dataset = scp.nmr.read_topspin("path/to/2rr")
dataset = scp.nmr.read_topspin("path/to/experiment", expno=1, procno=1)
```

The legacy alias `scp.read_topspin(...)` is kept for compatibility. TopSpin is
a reader, so it is not exposed as `dataset.read_topspin(...)` or
`dataset.nmr.read_topspin(...)`.

## Development

```bash
python -m pytest tests/ -v
```
