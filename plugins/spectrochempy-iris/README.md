# spectrochempy-iris

IRIS analysis plugin for SpectroChemPy.

Provides IRIS analysis objects, batch workflows, kernel comparison helpers,
visualisation helpers, and dataset-bound accessors.

## Installation

Version 0.2.0 requires SpectroChemPy 1.1.0 or later (and remains
restricted to SpectroChemPy versions below 2). Users who stay on
SpectroChemPy 1.0.0 should keep an earlier compatible version of this plugin.


```bash
pip install spectrochempy[iris]
```

IRIS uses SciPy NNLS by default for regularized fits. To use the OSQP
or quadprog solver, install the corresponding extra and set
`qpsolver="osqp"` or `qpsolver="quadprog"` on `IRIS`.

```bash
pip install "spectrochempy-iris[osqp]"
pip install "spectrochempy-iris[quadprog]"
```

For conda installations, install `osqp` or `quadprog` separately.
Select `qpsolver="osqp"` to retain the previous solver or to use the
`workers` option on fixed regularization grids. The `batch_iris` and
`compare_kernels` helpers also accept `qpsolver`.

## Changes in 0.2.0

- Regularized fits use SciPy NNLS by default. OSQP and quadprog are optional;
  install the corresponding extra and select `qpsolver` explicitly to use them.
- Fixed regularization grids can run in parallel with the OSQP builtin backend
  by setting `workers`. Solver setup and wide-spectrum sharpness calculations
  have also been optimized.
- Descending kernel grids preserve positive quadrature and regularization
  spacing. Adaptive searches associate RSS and sharpness with the evaluated
  regularization strength and restart from the configured bounds on refit.
  Setting `reg_par=None` disables regularization on the next fit.
- Python distributions and the Conda package include the project license.

The change of default solver can change fitted distributions and the selected
regularization strength. Compare results when migrating an existing analysis;
use `qpsolver="osqp"` to retain the previous solver selection.

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
pip install -e "plugins/spectrochempy-iris[osqp]"
cd plugins/spectrochempy-iris
python -m pytest tests/
```
