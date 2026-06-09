# spectrochempy-tensor

Tensor decomposition plugin for SpectroChemPy.

This official plugin provides TensorLy-backed tensor decomposition classes,
starting with CP/PARAFAC decomposition exposed as `scp.tensor.CP`.

## Installation

```bash
pip install spectrochempy-tensor
```

For local development from the SpectroChemPy monorepo:

```bash
pip install -e plugins/spectrochempy-tensor
```

## Usage

```python
import spectrochempy as scp

model = scp.tensor.CP(n_components=2)
model.fit(dataset)
```

The historical `scp.CP` alias is retained as a deprecated compatibility path.
