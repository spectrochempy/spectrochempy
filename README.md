<h1 align ="center">
<img src='https://github.com/spectrochempy/spectrochempy/raw/master/docs/_static/scpy.png' width="150">
<br>
SpectroChemPy
</h1>

[![Tests Actions Status](https://github.com/spectrochempy/spectrochempy/actions/workflows/test_package.yml/badge.svg)](https://github.com/spectrochempy/spectrochempy/actions)
[![codecov](https://codecov.io/gh/spectrochempy/spectrochempy/graph/badge.svg?token=A0NHMQNQ1E)](https://codecov.io/gh/spectrochempy/spectrochempy)
![Conda](https://img.shields.io/conda/v/spectrocat/spectrochempy)
[![PyPI version](https://badge.fury.io/py/spectrochempy.svg)](https://badge.fury.io/py/spectrochempy)
[![Docs](https://readthedocs.org/projects/docs/badge/?version=latest)](https://www.spectrochempy.fr/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3823841.svg)](https://doi.org/10.5281/zenodo.3823841)
![Conda](https://img.shields.io/conda/l/spectrocat/spectrochempy)

## What is SpectroChemPy?

SpectroChemPy (SCPy) is a framework for processing, analyzing and modeling spectroscopic data for chemistry with Python. It is cross-platform (Linux, Windows, macOS) and features a lightweight core with optional plugins for domain-specific workflows.

## Key Features

* **Core Data Structure**: `NDDataset` object with labeled axes and metadata
* **Project Management**: Work with multiple `NDDataset` objects simultaneously
* **Data Processing**:
  * Physical units support
  * Mathematical operations, baseline correction, automatic subtraction
  * Generic FFT with plugin-specific post-processing
* **Analysis Tools**:
  * SVD, PCA, MCR-ALS, EFA, PLS, fitting …
* **I/O Support**:
  * Import from various experiment formats
  * Export to CSV, JCAMP-DX, MATLAB …
  * Plugin-based readers (NMR TopSpin, Carroucell …)
* **Plugin System**:
  * Automatic discovery of optional plugins
  * Namespaced APIs: `scp.nmr.read_topspin(...)`, `scp.iris.IRIS()`
  * Dataset accessors for plugin-bound operations

## Optional Plugins

Extend SpectroChemPy with official plugins (installed separately):

| Plugin | Install | What it provides |
|--------|---------|------------------|
| `spectrochempy-nmr` | `pip install spectrochempy-nmr` | Bruker TopSpin reader, NMR-specific processing |
| `spectrochempy-iris` | `pip install spectrochempy-iris` | 2D-IRIS analysis tools |
| `spectrochempy-cantera` | `pip install spectrochempy-cantera` | Cantera-based reactor simulations |
| `spectrochempy-hypercomplex` | `pip install spectrochempy-hypercomplex` | Quaternion / hypercomplex support |
| `spectrochempy-carroucell` | `pip install spectrochempy-carroucell` | Carroucell experiment reader |

Or via conda:

```bash
mamba install -c spectrocat -c conda-forge spectrochempy-nmr
```

Plugins are discovered automatically once installed — no manual loading step required.

> **⚠️ WARNING**: SpectroChemPy is under active development. The current design may undergo major changes. Please report any issues to our [Issue Tracker](https://github.com/spectrochempy/spectrochempy/issues).

## Quick Links

* 📚 [Documentation](https://www.spectrochempy.fr)
* 🔌 [Plugins](https://www.spectrochempy.fr/userguide/plugins/index.html)
* 🚀 [Installation Guide](https://www.spectrochempy.fr/gettingstarted/install/index.html)
* 📓 [Example Notebooks](https://www.spectrochempy.fr/gettingstarted/examples/index.html)
* 🐛 [Issue Tracker](https://github.com/spectrochempy/spectrochempy/issues)
* 📝 [How to Cite](https://www.spectrochempy.fr/credits/citing.html)
* 💻 [Source Code](https://github.com/spectrochempy/spectrochempy)

## License

[CeCILL-B FREE SOFTWARE LICENSE AGREEMENT](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)
