.. _install_sources:

Installation from Sources
=========================

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
-------------

Git Installation
^^^^^^^^^^^^^^^

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

    .. tab:: Using mamba (recommended)

        1. Install Mambaforge from `miniforge <https://github.com/conda-forge/miniforge>`_

        2. Create and activate environment:

        .. sourcecode:: bash

            mamba env create -n scpy -f environments/environment.yml
            mamba activate scpy

        3. Install SpectroChemPy:

        .. sourcecode:: bash

            python -m pip install . --no-deps

    .. tab:: Using pip

        1. Create virtual environment:

        .. sourcecode:: bash

            python -m venv scpy-env
            source scpy-env/bin/activate  # Linux/macOS
            # or
            scpy-env\Scripts\activate  # Windows

        2. Install with optional components:

        .. sourcecode:: bash
         
            python -m pip install .



Verifying Installation
-------------------

.. sourcecode:: python

    from spectrochempy import *
    print(version)

Updating SpectroChemPy
-------------------

Update source code:

.. sourcecode:: bash

    git pull origin master --ff-only

Reinstall package:

.. sourcecode:: bash

    python -m pip install .

.. note::

    For development setup and contribution guidelines, see the :ref:`develguide`.
