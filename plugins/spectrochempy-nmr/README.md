# spectrochempy-nmr

NMR plugin for SpectroChemPy.

This package is the home for NMR-specific readers and tools that are useful in
SpectroChemPy but should not live in the core package. It currently provides a
validated public workflow for 1D NMR data across the supported readers, exposed
through `scp.nmr.read(...)` and `scp.nmr.Experiment(...)`.

Future NMR readers or processing helpers can be added here without creating a
new plugin package for each vendor format. Multi-dimensional NMR remains under
active characterization and is not currently part of the public supported
processing workflow.

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

dataset = scp.nmr.read("path/to/1/fid")
dataset = scp.nmr.read("path/to/1/pdata/1/1r")
dataset = scp.nmr.read("path/to/experiment", expno=1, procno=1)

experiment = scp.nmr.Experiment(dataset)
spectrum = experiment.process(apodization="em", lb=2.0, size=32768)
```

The NMR ppm/frequency unit context is also provided by this plugin:

```python
from spectrochempy_nmr.units import set_nmr_context

set_nmr_context(104.3 * scp.ur.MHz)
```

The compatibility aliases `scp.nmr.read_topspin(...)` and
`scp.read_topspin(...)` are kept for historical usage. TopSpin is a reader, so
it is not exposed as `dataset.read_topspin(...)` or
`dataset.nmr.read_topspin(...)`.

## Development

```bash
python -m pytest tests/ -v
```
