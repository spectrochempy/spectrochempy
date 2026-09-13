.. _install_win:

Installation Guide for Windows
==============================

This guide covers different methods for installing SpectroChemPy on Windows.

.. contents:: Table of Contents
   :local:
   :depth: 2

Requirements
------------

* Python 3.11 or newer

.. code-block:: bash

    # Check python version
    python --version  # Should be 3.11 or newer

Environment Setup
-----------------

.. tabs::

   .. tab:: Using uv (Recommended)

      1. Install uv:

         .. code-block:: powershell

            powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

         Restart your terminal after installation.

      2. Create environment:

         .. code-block:: bat

            uv venv scpy --python 3.13
            scpy\Scripts\activate

   .. tab:: Using conda / mamba

      1. Install Miniforge:

         * Download `Miniforge <https://github.com/conda-forge/miniforge>`_
         * Run the installer
         * Open cmd prompt

      2. Add required channels:

         .. code-block:: bat

            mamba config --add channels conda-forge
            mamba config --add channels spectrocat

      3. Create environment:

         .. code-block:: bat

            mamba create --name scpy python=3.13
            mamba activate scpy

   .. tab:: Using pip and venv

      1. Open Command Prompt
      2. Create virtual environment:

         .. code-block:: bat

            python -m venv scpy
            scpy\Scripts\activate

      3. Upgrade pip:

         .. code-block:: bat

            python -m pip install --upgrade pip

Installing SpectroChemPy
-----------------------

.. tabs::

   .. tab:: Using uv (Recommended)

      .. code-block:: bat

         uv pip install spectrochempy

         :: or with interactive extras
         uv pip install "spectrochempy[interactive]"

      Development version:

      .. code-block:: bat

         uv pip install spectrochempy --pre

   .. tab:: Using conda / mamba

      .. code-block:: bat

         mamba install -c spectrocat spectrochempy

      Development version:

      .. code-block:: bat

         mamba install -c spectrocat/label/dev spectrochempy

   .. tab:: Using pip

      .. code-block:: bat

         python -m pip install spectrochempy

         :: or if you want to install interactive version (including jupyter)
         python -m pip install "spectrochempy[interactive]"

Verifying Installation
----------------------

1. Start IPython:

   .. code-block:: bat

      ipython

2. Import SpectroChemPy:

   .. code-block:: ipython

      In [1]: import spectrochempy as scp

Additional Tips
---------------

Creating a Permanent Environment Shortcut
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a batch file (`.bat`) with:

.. code-block:: bat

    @REM launch cmd in scpy environment
    @CALL CD C:\<yourWorkingFolder>
    @CALL CMD /K C:\<yourMiniforgeFolder>\Scripts\activate.bat scpy

Save as `activate-scpy.bat` and create a shortcut named "Miniforge prompt (scpy)".

Next Steps
----------

Proceed to the :ref:`user_guide` to start using SpectroChemPy.

.. note::
   If you encounter any issues, see :doc:`../getting_help` for support options.
