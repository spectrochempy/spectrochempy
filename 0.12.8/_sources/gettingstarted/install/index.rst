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

Installing Python
-----------------

``uv`` (Recommended)
~~~~~~~~~~~~~~~~~~~~~

`uv <https://docs.astral.sh/uv/>`_ manages Python installations automatically.
Install uv, then let it download and manage the required Python version:

.. code-block:: bash

   # Install uv (macOS/Linux)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install uv (Windows)
   # powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Install Python
   uv python install 3.13

See `uv's installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_
for alternatives (``pipx``, ``brew``, ``winget``, etc.).

``Miniforge`` (conda / mamba)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Miniforge provides Python along with the ``conda`` and ``mamba`` package managers:

* Download `Miniforge <https://github.com/conda-forge/miniforge>`_
* Run the installer for your platform
* Verify the installation:

.. code-block:: bash

   conda --version  # or mamba --version

For users who prefer a full distribution:

* `Anaconda <https://www.anaconda.com/distribution/>`_ - Full scientific distribution (~3GB)
* `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ - Minimal distribution (~400MB)

System package manager
~~~~~~~~~~~~~~~~~~~~~~

If you prefer a standalone Python installation, use your platform's method:

.. tabs::

   .. tab:: macOS

      **Using Homebrew:**

      .. code-block:: bash

         /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
         brew install python
         python3 --version

      **Using Official Installer:**

      * Download from `Python.org (macOS) <https://www.python.org/downloads/macos/>`_
      * Run the installer
      * Verify with ``python3 --version``

   .. tab:: Windows

      **Using Official Installer:**

      * Download from `Python.org (Windows) <https://www.python.org/downloads/windows/>`_
      * Run installer (check "Add Python to PATH")
      * Verify with ``python --version``

      **Using Microsoft Store:**

      * Search for "Python" in Microsoft Store
      * Install the latest version
      * Verify in Command Prompt

   .. tab:: Linux

      Most distributions include Python. To install a specific version:

      **Ubuntu/Debian:**

      .. code-block:: bash

         sudo apt update
         sudo apt install python3.13

      **Fedora:**

      .. code-block:: bash

         sudo dnf install python3.13

Installing SpectroChemPy
-----------------------

Create Environment
~~~~~~~~~~~~~~~~~~

First, create and activate a dedicated environment:

.. tabs::

   .. tab:: Using uv (Recommended)

      .. code-block:: bash

         uv venv scpy --python 3.13
         source scpy/bin/activate  # On Windows: scpy\Scripts\activate

   .. tab:: Using conda / mamba

      .. code-block:: bash

         mamba create -n scpy python=3.13
         mamba activate scpy

   .. tab:: Using pip / venv

      .. code-block:: bash

         python -m venv scpy
         source scpy/bin/activate  # On Windows: scpy\Scripts\activate

Install Package
~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Using uv (Recommended)

      .. code-block:: bash

         uv pip install spectrochempy

   .. tab:: Using mamba

      .. code-block:: bash

         mamba install -c spectrocat spectrochempy

   .. tab:: Using conda

      .. code-block:: bash

         conda install -c spectrocat spectrochempy

   .. tab:: Using pip

      .. code-block:: bash

         python -m pip install spectrochempy


Additional Resources
--------------------

For more detailed instructions and alternative installation methods:

* :doc:`install_win` - Windows installation
* :doc:`install_mac_linux` - macOS/Linux installation
* :doc:`install_colab` - Using Google Colab
* :doc:`install_adds` - Additional components
* :doc:`install_sources` - Installing from sources

.. toctree::
    :maxdepth: 3
    :hidden:

    install_win
    install_mac_linux
    install_colab
    install_adds
    install_sources
