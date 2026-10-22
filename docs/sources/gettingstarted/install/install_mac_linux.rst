.. _install_mac:

Installation Guide for Mac OSX and Linux
========================================

This guide walks you through installing SpectroChemPy on macOS or Linux, step
by step. Every command can be copied and pasted in a terminal.

.. contents:: Table of Contents
   :local:
   :depth: 2

Choosing an Installation Method
-------------------------------

Three installation methods are described below, each based on a different
*package manager* (the tool that installs Python packages). **Pick one and
follow only its tab, from the first step to the last.** The steps of one tab
do not apply to the others.

* **uv** (recommended): the simplest choice if you are new to Python. uv
  also downloads Python for you.
* **conda / mamba**: choose it if you already use Miniforge, Anaconda or
  conda environments, or if you also want other scientific tools from
  conda-forge.
* **pip and venv**: uses a Python that you install yourself.

Two folders are involved, and it is good practice to keep them separate:

* the **environment**, named ``scpy``, which contains Python, SpectroChemPy
  and its dependencies. You never need to open or modify it. With uv and
  venv, this guide puts it in ``~/scpy`` (``~`` is your home folder); with
  conda / mamba, Miniforge stores it in its own folder.
* your **working folder**, which contains your data, scripts and notebooks.
  It can be any folder you like, for instance ``~/Documents/spectrochempy``.
  Below, it is written ``<yourWorkingFolder>``: replace it with the full path
  of your folder.

.. warning::

   Do not create the environment in a folder synchronized with the cloud
   (iCloud Drive, OneDrive, Google Drive, Dropbox...): it contains thousands
   of files, which slows down the synchronization and can break the
   environment. On macOS, iCloud may synchronize the *Documents* and
   *Desktop* folders. Your working folder, on the other hand, can be
   synchronized.

.. tip::

   To open a **terminal**: on macOS, press :kbd:`Cmd+Space`, type
   ``Terminal`` and press :kbd:`Enter`; on most Linux desktops, press
   :kbd:`Ctrl+Alt+T`. To paste a command, press :kbd:`Cmd+V` (macOS) or
   :kbd:`Ctrl+Shift+V` (Linux), then press :kbd:`Enter`.

Step-by-step Installation
-------------------------

.. tabs::

   .. tab:: Using uv (Recommended)

      **1. Install uv.** Open a terminal and run:

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      Then close the terminal and open a new one. Other ways to install uv
      (Homebrew, ``pipx``...) are listed in
      `uv's installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.

      **2. Create and activate the environment** in ``~/scpy`` (uv downloads
      Python if needed):

      .. code-block:: bash

         uv venv ~/scpy --python 3.13
         source ~/scpy/bin/activate

      The prompt now starts with ``(scpy)``.

      **3. Install SpectroChemPy:**

      .. code-block:: bash

         uv pip install spectrochempy

         # or with interactive extras (Jupyter, etc.)
         uv pip install "spectrochempy[interactive]"

      Latest prerelease:

      .. code-block:: bash

         uv pip install spectrochempy --pre

      **4. Create a shortcut command.** It opens your working folder with the
      ``scpy`` environment already active, so you do not have to type the
      activation commands each time. Run the following line once, after replacing
      ``<yourWorkingFolder>`` with the path of your working folder (keep the
      quotes). On macOS, the shell is ``zsh`` and its configuration file is
      ``~/.zshrc``; on most Linux distributions, it is ``bash`` and ``~/.bashrc``:

      .. code-block:: bash

         # macOS
         echo 'alias scpy="cd \"<yourWorkingFolder>\" && source ~/scpy/bin/activate"' >> ~/.zshrc

         # Linux
         echo 'alias scpy="cd \"<yourWorkingFolder>\" && source ~/scpy/bin/activate"' >> ~/.bashrc

      **5. Check the installation.** Close the terminal, open a new one and
      type ``scpy``: the prompt starts with ``(scpy)`` and you are in your working
      folder. Start IPython and import SpectroChemPy:

      .. code-block:: bash

         scpy
         ipython

      .. code-block:: ipython

         In [1]: import spectrochempy as scp

      Type ``exit`` to leave IPython. From now on, open a terminal and type
      ``scpy`` each time you work with SpectroChemPy.

   .. tab:: Using conda / mamba

      **1. Install Miniforge.** Skip this step if Miniforge (or Anaconda) is
      already installed. Otherwise, open a terminal and run:

      .. code-block:: bash

         curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
         bash Miniforge3-$(uname)-$(uname -m).sh

      Accept the license, keep the proposed installation folder (by default
      ``~/miniforge3``), and answer ``yes`` when asked whether to initialize
      conda. Then close the terminal and open a new one: the prompt starts
      with ``(base)``.

      **2. Add the required channels** (needed only once):

      .. code-block:: bash

         conda config --add channels conda-forge
         conda config --add channels spectrocat

      With Miniforge, the first command may print ``Warning: 'conda-forge'
      already in 'channels' list, moving to the top``: this is harmless.

      **3. Create and activate the environment:**

      .. code-block:: bash

         mamba create --name scpy python=3.13
         conda activate scpy

      At the end, ``mamba create`` suggests ``mamba activate scpy``; both
      commands work, this guide uses ``conda activate``. The prompt now starts
      with ``(scpy)``. The environment is stored in
      Miniforge's own folder (``envs/scpy``), separate from your working
      folder.

      **4. Install SpectroChemPy:**

      .. code-block:: bash

         mamba install -c spectrocat spectrochempy

      Development versions are not published to a Conda ``dev`` channel.
      To use unreleased code, follow the source installation instructions in
      :doc:`install_sources`.

      **5. Create a shortcut command.** It opens your working folder with the
      ``scpy`` environment already active, so you do not have to type the
      activation commands each time. Run the following line once, after replacing
      ``<yourWorkingFolder>`` with the path of your working folder (keep the
      quotes). On macOS, the shell is ``zsh`` and its configuration file is
      ``~/.zshrc``; on most Linux distributions, it is ``bash`` and ``~/.bashrc``:

      .. code-block:: bash

         # macOS
         echo 'alias scpy="cd \"<yourWorkingFolder>\" && conda activate scpy"' >> ~/.zshrc

         # Linux
         echo 'alias scpy="cd \"<yourWorkingFolder>\" && conda activate scpy"' >> ~/.bashrc

      **6. Check the installation.** Close the terminal, open a new one and
      type ``scpy``: the prompt starts with ``(scpy)`` and you are in your working
      folder. Start IPython and import SpectroChemPy:

      .. code-block:: bash

         scpy
         ipython

      .. code-block:: ipython

         In [1]: import spectrochempy as scp

      Type ``exit`` to leave IPython. From now on, open a terminal and type
      ``scpy`` each time you work with SpectroChemPy.

   .. tab:: Using pip and venv

      **1. Check your Python version.** Open a terminal and run:

      .. code-block:: bash

         python3 --version

      SpectroChemPy needs Python 3.11 or newer. Some Linux distributions
      still ship an older one (for instance Python 3.10 on Ubuntu 22.04):
      in that case, the simplest is to use the uv or conda / mamba method
      instead, which install a recent Python for you.

      **2. Install Python, if needed:**

      * macOS: download the installer from
        `python.org <https://www.python.org/downloads/macos/>`_, or, with
        `Homebrew <https://brew.sh>`_, run ``brew install python``.
      * Debian / Ubuntu: ``sudo apt install python3 python3-venv`` (the
        ``python3-venv`` package is required to create environments, even if
        Python is already installed).
      * Fedora: ``sudo dnf install python3``

      **3. Create and activate the environment** in ``~/scpy``:

      .. code-block:: bash

         python3 -m venv ~/scpy
         source ~/scpy/bin/activate
         python -m pip install --upgrade pip

      The prompt now starts with ``(scpy)``.

      **4. Install SpectroChemPy:**

      .. code-block:: bash

         python -m pip install spectrochempy

         # or if you want to install interactive version (including jupyter)
         python -m pip install "spectrochempy[interactive]"

      **5. Create a shortcut command.** It opens your working folder with the
      ``scpy`` environment already active, so you do not have to type the
      activation commands each time. Run the following line once, after replacing
      ``<yourWorkingFolder>`` with the path of your working folder (keep the
      quotes). On macOS, the shell is ``zsh`` and its configuration file is
      ``~/.zshrc``; on most Linux distributions, it is ``bash`` and ``~/.bashrc``:

      .. code-block:: bash

         # macOS
         echo 'alias scpy="cd \"<yourWorkingFolder>\" && source ~/scpy/bin/activate"' >> ~/.zshrc

         # Linux
         echo 'alias scpy="cd \"<yourWorkingFolder>\" && source ~/scpy/bin/activate"' >> ~/.bashrc

      **6. Check the installation.** Close the terminal, open a new one and
      type ``scpy``: the prompt starts with ``(scpy)`` and you are in your working
      folder. Start IPython and import SpectroChemPy:

      .. code-block:: bash

         scpy
         ipython

      .. code-block:: ipython

         In [1]: import spectrochempy as scp

      Type ``exit`` to leave IPython. From now on, open a terminal and type
      ``scpy`` each time you work with SpectroChemPy.

Next Steps
----------

Proceed to the :ref:`user_guide` to start using SpectroChemPy.

.. note::
   If you encounter any issues, see :doc:`../getting_help` for support options.
