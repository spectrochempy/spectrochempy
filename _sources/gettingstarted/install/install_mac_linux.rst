.. _install_mac:

Installation Guide for Mac OSX and Linux
========================================

This guide covers different methods for installing SpectroChemPy on MacOS or Linux.

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

        `uv <https://docs.astral.sh/uv/>`_ is the recommended tool for most users.
        Install uv if you do not have it yet:

        .. code-block:: bash

            curl -LsSf https://astral.sh/uv/install.sh | sh

        Then create a virtual environment:

        .. code-block:: bash

            # Create environment with Python 3.13
            uv venv scpy-env --python 3.13

            # Activate it
            source scpy-env/bin/activate

    .. tab:: Using conda / mamba

        Mamba offers faster dependency resolution than conda.

        1. Install Miniforge

        Download and install `Miniforge <https://github.com/conda-forge/miniforge>`_


        2. Add required channels:

        .. code-block:: bash

            mamba config --add channels conda-forge
            mamba config --add channels spectrocat

        3. Create and activate environment

        .. code-block:: bash

            # Create new environment
            mamba create -n scpy python=3.13

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

    .. tab:: Using uv (Recommended)

        .. code-block:: bash

            uv pip install spectrochempy

            # or if you want to install interactive version (including jupyter)
            uv pip install "spectrochempy[interactive]"

        Development version:

        .. code-block:: bash

            uv pip install spectrochempy --pre

    .. tab:: Using conda / mamba

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

      In [1]: import spectrochempy as scp

Next Steps
----------

Proceed to the :ref:`user_guide` to start using SpectroChemPy.

.. note::
   If you encounter any issues, see :doc:`../getting_help` for support options.
