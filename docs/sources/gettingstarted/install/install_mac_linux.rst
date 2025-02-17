.. _install_mac:

Installation Guide for Mac OSX and Linux
========================================

This guide covers different methods for installing SpectroChemPy on MacOS or Linux.

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

    .. tab:: Using conda/mamba (recommended)

        We strongly recommend using mamba as it's significantly faster at resolving dependencies.

        1. Install Mambaforge

        Download and install `Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_


        2. Add required channels:

        .. code-block:: bat

            mamba config --add channels conda-forge
            mamba config --add channels spectrocat

        3. Create and activate environment

        .. code-block:: bash

            # Create new environment
            mamba create -n scpy python=3.10

            # Activate it
            mamba activate scpy

    .. tab:: Using pip and venv

        .. code-block:: bash

            # Create environment
            python -m venv scpy-env

            # Activate it
            source scpy-env/bin/activate

            # Update pip
            python -m pip install --upgrade pip


Install SpectroChemPy
---------------------

.. tabs::

    .. tab:: Using conda/mamba

        Stable version

        .. code-block:: bash

            mamba install -c spectrocat spectrochempy

        Development version

        .. code-block:: bash

            mamba install -c spectrocat/label/dev spectrochempy

    .. tab:: Using pip

        .. code-block:: bash

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

Next Steps
----------

Proceed to the :ref:`userguide` to start using SpectroChemPy.

.. note::
   If you encounter any issues, see :doc:`../getting_help` for support options.
