# spectrochempy-nmr

NMR plugin for SpectroChemPy.

This package is the home for NMR-specific readers and tools that are useful in
SpectroChemPy but should not live in the core package. It currently provides a
validated public 1D workflow for reading data and processing raw 1D FIDs
through `scp.nmr.read(...)` and `scp.nmr.Experiment(...)`. The readers also
load raw 2D time-domain data and vendor-processed 2D spectra and can represent
them as hypercomplex arrays; this reading/representation scope is separate from
processing, which remains limited to validated 1D experiments.

Future NMR readers or processing helpers can be added here without creating a
new plugin package for each vendor format. Multi-dimensional NMR processing
remains under active characterization and is not currently part of the public
supported processing workflow. A pseudo-2D series (a list of ordinary 1D
spectra sharing a secondary coordinate, such as a relaxation series) must be
distinguished from a genuine 2D experiment with two Fourier dimensions: the
former follows the validated 1D processing path per spectrum.

## Installation

The next release, 0.1.13, requires SpectroChemPy 1.1.0 or later (and remains
restricted to SpectroChemPy versions below 2). Users who stay on
SpectroChemPy 1.0.0 should keep an earlier compatible version of this plugin.


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

This explicit processing example does not imply that vendor processing
parameters are automatically imported and replayed from metadata. That contract
remains under active characterization.

The current explicit apodization contract covers the public modes already
exposed by `Experiment.process()`: `em(lb=...)`, `gm(lb=..., gb=...)`, and
`sp(ssb=..., pow=...)`.

The result of `Experiment.process()` also records the SpectroChemPy-owned
processing trace in `result.meta.nmr_processing["scp_processing"]`.
`requested` contains only the arguments explicitly provided by the user,
whereas `applied` contains only the operations that were actually executed and
the values they really consumed. This trace is attached to the result only:
the source dataset is not mutated, the vendor `procs` profile remains purely
descriptive, and `phase="metadata"` does not replay TopSpin `PHC0`/`PHC1`.

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
