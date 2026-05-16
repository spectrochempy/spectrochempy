# spectrochempy-iris

IRIS analysis extension for SpectroChemPy.

Provides additional multivariate analysis workflows (PCA, PARAFAC, MCR)
and custom visualisation for spectroscopic decomposition.

## Installation

```bash
pip install spectrochempy-iris
```

## Usage

```python
import spectrochempy as scp

# PCA decomposition
from spectrochempy_iris import pca_decomposition
scores, loadings = pca_decomposition(dataset, n_components=3)

# MCR analysis
from spectrochempy_iris import mcr_analysis
concentrations, spectra = mcr_analysis(dataset, n_components=4)
```

## Development

```bash
pip install -e .
python -m pytest tests/
```
