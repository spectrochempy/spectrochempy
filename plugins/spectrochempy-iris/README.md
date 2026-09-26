# spectrochempy-iris

IRIS analysis plugin for SpectroChemPy.

Provides IRIS analysis objects, batch workflows, kernel comparison helpers,
visualisation helpers, and dataset-bound accessors.

## Installation

The next release, 0.1.10, requires SpectroChemPy 1.1.0 or later (and remains
restricted to SpectroChemPy versions below 2). Users who stay on
SpectroChemPy 1.0.0 should keep an earlier compatible version of this plugin.


```bash
pip install spectrochempy[iris]
```

## Usage

```python
import spectrochempy as scp

# Package-level plugin APIs are available under scp.iris after installation.
results = scp.iris.batch_iris([dataset], kernel_type="langmuir", q=[-6, 1, 6])
report = scp.iris.iris_report(results[0]["iris"])

# Operations on an existing dataset are exposed through dataset accessors.
kernel = dataset.iris.kernel_matrix(kernel_type="langmuir", q=[-6, 1, 6])
```

The legacy flat accessor `dataset.iris_kernel_matrix(...)` is kept for
compatibility.

## Development

```bash
pip install -e .
pip install -e plugins/spectrochempy-iris
cd plugins/spectrochempy-iris
python -m pytest tests/
```
