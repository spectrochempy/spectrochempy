.. _install_sources:

Installation from Sources
=========================

.. contents:: Contents
   :local:
   :depth: 2

Prerequisites
-------------

Git Installation
^^^^^^^^^^^^^^^^

Install ``git`` if not already present:

* macOS: ``brew install git`` or via `XCode <https://developer.apple.com/xcode/>`__
* Linux: ``sudo apt-get install git`` (Ubuntu/Debian) or equivalent
* Windows: Download from `git-scm.com <https://git-scm.com/download/win>`__

Verify installation:

.. sourcecode:: bash

    git --version

Getting the Source Code
-----------------------

Clone the repository:

.. sourcecode:: bash

    git clone https://github.com/spectrochempy/spectrochempy.git
    cd spectrochempy

Installation Methods
--------------------

.. tabs::

    .. tab:: mamba (recommended)

        .. code-block:: bash

            # Install Mambaforge
            # Get it from: https://github.com/conda-forge/miniforge#mambaforge

            # Create environment
            mamba env create -n scpy -f environments/environment.yml
            mamba activate scpy

            # Install package
            python -m pip install -e . --no-deps

    .. tab:: pip

        .. code-block:: bash

            # Create environment
            python -m venv scpy-env
            source scpy-env/bin/activate  # Linux/macOS
            # or
            scpy-env\Scripts\activate     # Windows

            # Install package
            python -m pip install -e .

Verifying Installation
----------------------

.. sourcecode:: python

    from spectrochempy import *
    print(version)

Updating SpectroChemPy
----------------------

Update source code:

.. sourcecode:: bash

    git pull origin master --ff-only

Reinstall package:

.. sourcecode:: bash

    python -m pip install .

.. note::

    For development setup and contribution guidelines, see the :ref:`develguide`.
