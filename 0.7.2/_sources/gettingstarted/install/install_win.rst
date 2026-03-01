.. _install_win:

Installation Guide for Windows
==============================

This guide covers different methods for installing SpectroChemPy on Windows.

.. contents:: Table of Contents
   :local:
   :depth: 2

Requirements
------------

* Python 3.10 or newer

.. code-block:: bash

    # Check python version
    python --version  # Should be 3.10 or newer

Environment Setup
-----------------

.. tabs::

   .. tab:: Mamba (Recommended)

      1. Install Mambaforge:

         * Download `Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_
         * Run the installer
         * Open cmd prompt




      3. Create environment:

         .. code-block:: bat

            mamba create --name scpy python=3.10
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
------------------------

.. tabs::

   .. tab:: Using Mamba (Recommended)

      .. code-block:: bat

         mamba install -c spectrocat spectrochempy

      Development version:

      .. code-block:: bat

         mamba install -c spectrocat/label/dev spectrochempy

   .. tab:: Using Pip

      .. code-block:: bat

         # Install SpectroChemPy
         python -m pip install spectrochempy

         # or if you want to install interactive version (including jupyter)
         python -m pip install "spectrochempy[interactive]"

Verifying Installation
----------------------

1. Start IPython:

   .. code-block:: bat

      ipython

2. Import SpectroChemPy:

   .. code-block:: ipython

      In [1]: from spectrochempy import *

   You should see:

   .. code-block:: text

      SpectroChemPy's API - v.0.6.10
      © Copyright 2014-2025 - A.Travert & C.Fernandez @ LCS

Additional Tips
---------------

Creating a Permanent Environment Shortcut
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a batch file (`.bat`) with:

.. code-block:: bat

    @REM launch cmd in scpy environment
    @CALL CD C:\<yourWorkingFolder>
    @CALL CMD /K C:\<yourMambaForgeFolder>\Scripts\activate.bat scpy

Save as `activate-scpy.bat` and create a shortcut named "Mamba prompt (scpy)".

Next Steps
----------

Proceed to the :ref:`userguide` to start using SpectroChemPy.

.. note::
   If you encounter any issues, see :doc:`../getting_help` for support options.
