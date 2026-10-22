.. _install_sources:

Installation from Sources
=========================

This page explains how to install SpectroChemPy from its source code, to use
the development version (the current ``master`` branch, documented as
*latest* on this site) or to contribute to the project. If you only want to
use SpectroChemPy, follow instead the step-by-step guide for your system in
:doc:`index`.

Contributors should clone their own fork of the repository instead of the main
one: see the :ref:`develguide`.

.. contents:: Contents
   :local:
   :depth: 2

Prerequisites
-------------

``git`` is required to get the source code. Check whether it is installed:

.. code-block:: bash

    git --version

If it is not:

* Windows: download and run the installer from
  `git-scm.com <https://git-scm.com/download/win>`__
* macOS: run ``xcode-select --install`` (Apple command line tools), or
  ``brew install git`` with `Homebrew <https://brew.sh>`__
* Linux: ``sudo apt install git`` (Debian / Ubuntu) or
  ``sudo dnf install git`` (Fedora)

Each method below also needs its package manager: uv, Miniforge (conda /
mamba) or Python 3.11 or newer. The platform guides explain how to install
them (:doc:`install_win`, :doc:`install_mac`, :doc:`install_linux`).

Installation
------------

Pick one method and follow only its tab. Commands are given for macOS / Linux
(``bash``) and, where they differ, for the Windows Command Prompt.

.. tabs::

    .. tab:: Using uv (Recommended)

        **1. Get the source code:**

        .. code-block:: bash

            git clone https://github.com/spectrochempy/spectrochempy.git
            cd spectrochempy

        **2. Create and activate the environment** in the ``.venv`` folder of
        the source tree:

        .. code-block:: bash

            uv venv --python 3.13
            source .venv/bin/activate     # macOS / Linux
            .venv\Scripts\activate        # Windows

        **3. Install SpectroChemPy in editable mode**, with the development
        tools (tests, documentation, linters):

        .. code-block:: bash

            uv pip install -e ".[dev]"

        **4. Check the installation:**

        .. code-block:: bash

            python -c "import spectrochempy as scp; print(scp.__version__)"

        The version ends with ``.devN`` (for instance ``1.1.4.dev9``), which
        identifies the development version.

        **5. Update.** Changes to the source code are taken into account
        immediately (editable install). To get the latest changes, from the
        ``spectrochempy`` folder with the environment active:

        .. code-block:: bash

            git pull origin master --ff-only
            uv pip install -e ".[dev]"    # only needed if dependencies changed

        **6. Next time**, go to the ``spectrochempy`` folder and activate the
        environment again (step 2, ``activate`` line only).

    .. tab:: Using conda / mamba

        **1. Get the source code:**

        .. code-block:: bash

            git clone https://github.com/spectrochempy/spectrochempy.git
            cd spectrochempy

        **2. Create and activate the environment** from the development
        environment file, which lists all dependencies, including the
        development tools (tests, documentation, linters). On Windows, use
        the Miniforge Prompt:

        .. code-block:: bash

            mamba env create -n scpy-dev -f environments/environment_dev.yml
            conda activate scpy-dev

        **3. Install SpectroChemPy in editable mode**, without dependencies
        (they come from the environment file):

        .. code-block:: bash

            python -m pip install -e . --no-deps

        **4. Check the installation:**

        .. code-block:: bash

            python -c "import spectrochempy as scp; print(scp.__version__)"

        The version ends with ``.devN`` (for instance ``1.1.4.dev9``), which
        identifies the development version.

        **5. Update.** Changes to the source code are taken into account
        immediately (editable install). To get the latest changes, from the
        ``spectrochempy`` folder with the environment active:

        .. code-block:: bash

            git pull origin master --ff-only
            mamba env update -n scpy-dev -f environments/environment_dev.yml   # only needed if dependencies changed

        **6. Next time**, go to the ``spectrochempy`` folder and run
        ``conda activate scpy-dev``.

    .. tab:: Using pip and venv

        **1. Get the source code:**

        .. code-block:: bash

            git clone https://github.com/spectrochempy/spectrochempy.git
            cd spectrochempy

        **2. Create and activate the environment** in the ``.venv`` folder of
        the source tree (Python 3.11 or newer; use ``python`` instead of
        ``python3`` on Windows):

        .. code-block:: bash

            python3 -m venv .venv
            source .venv/bin/activate     # macOS / Linux
            .venv\Scripts\activate        # Windows
            python -m pip install --upgrade pip

        **3. Install SpectroChemPy in editable mode**, with the development
        tools (tests, documentation, linters):

        .. code-block:: bash

            python -m pip install -e ".[dev]"

        **4. Check the installation:**

        .. code-block:: bash

            python -c "import spectrochempy as scp; print(scp.__version__)"

        The version ends with ``.devN`` (for instance ``1.1.4.dev9``), which
        identifies the development version.

        **5. Update.** Changes to the source code are taken into account
        immediately (editable install). To get the latest changes, from the
        ``spectrochempy`` folder with the environment active:

        .. code-block:: bash

            git pull origin master --ff-only
            python -m pip install -e ".[dev]"    # only needed if dependencies changed

        **6. Next time**, go to the ``spectrochempy`` folder and activate the
        environment again (step 2, ``activate`` line only).

.. note::

    For the development workflow (fork, branches, tests, pre-commit hooks,
    pull requests), see the :ref:`develguide`.
