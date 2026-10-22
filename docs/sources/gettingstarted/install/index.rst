.. _installation:

Installation Guide
==================

.. contents:: Table of Contents
   :local:
   :depth: 3

Prerequisites
-------------

`SpectroChemPy` requires Python 3.11 or higher (tested up to 3.14). Python is a widely-adopted
scientific programming language, particularly suited for data analysis and scientific computing.
You do not need to install Python beforehand if you use ``uv`` or ``conda`` / ``mamba``: they
install it for you.

Choosing an Installation Method
-------------------------------

The best installation method depends on your workflow and experience level:

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Situation
     - Recommended method
   * - New to Python
     - ``uv`` — simple, fast, single-command setup
   * - Existing conda environment
     - ``mamba`` / ``conda`` (Miniforge) — integrates with existing environments
   * - Scientific workflow (Jupyter, Spyder)
     - ``mamba`` / ``conda`` (Miniforge) — full scientific distribution
   * - HPC / CI / containers
     - ``uv`` — fast, reproducible, minimal footprint
   * - Installing from source / contributing
     - ``uv`` — matches the project's own development tooling
   * - Any environment / maximum compatibility
     - ``pip`` — universal fallback

.. note::

   **uv** is a modern, fast Python package and project manager. It is the recommended
   choice for most users. The SpectroChemPy development team uses ``uv`` daily.

   **conda** / **mamba** (via `Miniforge <https://github.com/conda-forge/miniforge>`_)
   is the recommended choice if you already rely on conda-based scientific environments.

   **pip** works everywhere and is documented as a universal fallback.

Installation Guides
-------------------

Each guide below gives complete, step-by-step instructions, from installing the
tools to checking that SpectroChemPy works:

* :doc:`install_win` - Windows
* :doc:`install_mac_linux` - macOS and Linux
* :doc:`install_colab` - Using Google Colab, with nothing to install on your computer
* :doc:`install_sources` - Installing from sources (contributors)
* :doc:`install_adds` - Additional components

Installing a Specific Version
-----------------------------

The installation commands of these guides select the latest stable release.
For a reproducible environment, install an exact version, for instance:

.. code-block:: bash

   uv pip install "spectrochempy==1.0.0"
   # or: python -m pip install "spectrochempy==1.0.0"
   # or: mamba install -c spectrocat "spectrochempy=1.0.0"

and check it with:

.. code-block:: bash

   python -c "import spectrochempy as scp; print(scp.__version__)"

.. toctree::
    :maxdepth: 3
    :hidden:

    install_win
    install_mac_linux
    install_colab
    install_adds
    install_sources
