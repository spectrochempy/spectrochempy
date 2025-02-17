.. _installation:

Installation Guide
==================

.. contents:: Table of Contents
   :local:
   :depth: 3

Prerequisites
-------------

`SpectroChemPy` requires Python 3.10 or higher (tested up to 3.13). Python is a widely-adopted
scientific programming language, particularly suited for data analysis and scientific computing.

Installing Python
-----------------

Conda Package Managers (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using one of these package managers for installing Python and managing dependencies:

* `Anaconda <https://www.anaconda.com/distribution/>`_ - Full scientific distribution (~3GB)
* `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ - Minimal distribution (~400MB)
* `Mamba <https://mamba.readthedocs.io/en/latest/installation.html>`_ - Fast alternative to conda
* `Micromamba <https://mamba.readthedocs.io/en/latest/installation.html#micromamba>`_ - Minimal mamba

Installation Steps
~~~~~~~~~~~~~~~~~~

1. Download your chosen package manager
2. Run the installer for your platform
3. Open a terminal/command prompt
4. Verify the installation:

.. code-block:: bash

   conda --version  # or mamba --version

Alternative Installation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: macOS

      **Using Homebrew:**

      .. code-block:: bash

         # Install Homebrew
         /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

         # Install Python
         brew install python

         # Verify installation
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
         sudo apt install python3.10

      **Fedora:**

      .. code-block:: bash

         sudo dnf install python3.10

Installing SpectroChemPy
------------------------

Create Environment
~~~~~~~~~~~~~~~~~~

First, create and activate a dedicated environment:

.. code-block:: bash

   # Using conda/mamba
   mamba create -n scpy python=3.10
   mamba activate scpy

   # OR using venv (if not using conda/mamba)
   python -m venv scpy
   source scpy/bin/activate  # On Windows: scpy\Scripts\activate

Install Package
~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Using Mamba (Recommended)

      .. code-block:: bash

         mamba install -c spectrocat spectrochempy

   .. tab:: Using Conda

      .. code-block:: bash

         conda install -c spectrocat spectrochempy

   .. tab:: Using Pip

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
