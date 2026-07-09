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
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.3823841-blue.svg)](https://doi.org/10.5281/zenodo.3823841)
![Conda](https://img.shields.io/conda/l/spectrocat/spectrochempy)

## What is SpectroChemPy?

SpectroChemPy (SCPy) is a framework for processing, analyzing, and modeling
spectroscopic data for chemistry with Python. It is cross-platform (Linux,
Windows, macOS) and combines a lightweight scientific core with optional
plugins for domain-specific workflows.

Its central object, `NDDataset`, provides numerical data together with
coordinates, units, masks, labels, provenance, and metadata, making it easier
to build reproducible spectroscopy workflows in Python.

## Key Features

* **Core Data Structure**: `NDDataset` with coordinates, units, masks, labels,
  provenance, and metadata
* **Project Management**: Organize and manipulate multiple datasets within a
  `Project`
* **Data Processing**:
  * Unit-aware mathematical operations
  * Baseline correction, automatic subtraction, interpolation, FFT workflows
  * Core processing with plugin-specific extensions where appropriate
* **Analysis Tools**:
  * SVD, PCA, MCR-ALS, EFA, PLS, fitting, and related result objects
* **I/O and Interoperability**:
  * Import from major spectroscopy and scientific data formats
  * Export to lightweight interchange formats such as CSV, JCAMP-DX, and
    minimal MATLAB `.mat` exchange files
  * Portable `NDDataset ↔ xarray.Dataset ↔ NetCDF` round-trips for the
    maintained portable subset
  * Safe native `.scp` / `.pscp` persistence by default, with explicit legacy
    opt-in only for historical trusted archives
* **Plugin System**:
  * Automatic discovery of optional plugins
  * Namespaced APIs: `scp.nmr.read_topspin(...)`, `scp.iris.IRIS()`
  * Dataset accessors for plugin-bound operations

## Installation

The recommended way to install SpectroChemPy is with `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv venv scpy --python 3.13
source scpy/bin/activate
uv pip install spectrochempy
```

Alternatively, with `mamba` / `conda`:

```bash
mamba install -c spectrocat -c conda-forge spectrochempy
```

Or with `pip`:

```bash
pip install spectrochempy
```

See the [full installation guide](https://www.spectrochempy.fr/gettingstarted/install/index.html) for platform-specific instructions, optional dependencies, and development setup.

## Official Plugins

Extend SpectroChemPy with official plugins (installed separately):

| Plugin | Install | What it provides |
|--------|---------|------------------|
| `spectrochempy-carroucell` | `uv pip install spectrochempy-carroucell` | Carroucell experiment reader |
| `spectrochempy-hypercomplex` | `uv pip install spectrochempy-hypercomplex` | Quaternion / hypercomplex support |
| `spectrochempy-iris` | `uv pip install spectrochempy-iris` | 2D-IRIS analysis tools |
| `spectrochempy-nmr` | `uv pip install spectrochempy-nmr` | Bruker TopSpin reader, NMR-specific processing |
| `spectrochempy-perkinelmer` | `uv pip install spectrochempy-perkinelmer` | PerkinElmer ``.sp`` IR file reader |
| `spectrochempy-tensor` | `uv pip install spectrochempy-tensor` | Tensor learning tools |

`pip install` works as well if you are not using `uv`.

Plugins are also available via conda from the `spectrocat` channel:

```bash
mamba install -c spectrocat -c conda-forge spectrochempy-nmr
```

Plugins are discovered automatically once installed — no manual loading step required.

### Experimental plugins

`spectrochempy-cantera` is available as an **experimental** plugin. It is not
officially supported, not included in aggregate extras, and must be installed
manually::

    pip install spectrochempy-cantera

## Quick Links

* 📚 [Documentation](https://www.spectrochempy.fr)
* 🔌 [Plugins](https://www.spectrochempy.fr/userguide/plugins/index.html)
* 🚀 [Installation Guide](https://www.spectrochempy.fr/gettingstarted/install/index.html)
* 📓 [Example Notebooks](https://www.spectrochempy.fr/gettingstarted/examples/index.html)
* 🐛 [Issue Tracker](https://github.com/spectrochempy/spectrochempy/issues)
* 📝 [How to Cite](https://www.spectrochempy.fr/credits/citing.html)
* 💻 [Source Code](https://github.com/spectrochempy/spectrochempy)

## For contributors

If you want to contribute, start with:

* [`CONTRIBUTING.md`](CONTRIBUTING.md)
* the developer guide under [`docs/sources/devguide/`](docs/sources/devguide/)
* the [issue tracker](https://github.com/spectrochempy/spectrochempy/issues)

Maintainer release procedures live in [`maintainers/`](maintainers/).
Architecture, RFCs, and roadmap discussions are maintained in the
**SpectroChemPy Maintainer Repository**.

## License

[CeCILL-B FREE SOFTWARE LICENSE AGREEMENT](https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)
