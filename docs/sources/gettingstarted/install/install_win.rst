.. _install_win:

Installation Guide for Windows
==============================

This guide walks you through installing SpectroChemPy on Windows, step by
step. No previous experience with the command line is needed: every command
can be copied and pasted.

.. contents:: Table of Contents
   :local:
   :depth: 2

Choosing a Method
-----------------

Three methods are described below. **Pick one and follow only its tab, from
the first step to the last.** The steps of one tab do not apply to the others.

* **uv** (recommended): the simplest choice if you are new to Python. uv
  also downloads Python for you.
* **conda / mamba**: choose it if you already use Miniforge, Anaconda or
  conda environments, or if you also want other scientific tools from
  conda-forge.
* **pip and venv**: uses a Python that you install yourself from python.org.

In all methods, you work in a *working folder* that holds your SpectroChemPy
environment and, if you like, your data and notebooks. The examples use
``%USERPROFILE%\Documents\spectrochempy`` (that is,
``C:\Users\<yourName>\Documents\spectrochempy``); you can choose another
folder, as long as you use it consistently in every step.

.. tip::

   To open the **Command Prompt**, click the Start menu, type ``cmd``, and
   select "Command Prompt". To paste a command in it, right-click or press
   :kbd:`Ctrl+V`, then press :kbd:`Enter`.

Step-by-step Installation
-------------------------

.. tabs::

   .. tab:: Using uv (Recommended)

      **1. Install uv.** Open **Windows PowerShell** (Start menu, type
      ``powershell``) and run:

      .. code-block:: powershell

         powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

      Then close PowerShell.

      **2. Open the Command Prompt** and go to your working folder. If the folder already exists,
      ``mkdir`` prints an error message that you can ignore:

      .. code-block:: bat

         mkdir "%USERPROFILE%\Documents\spectrochempy"
         cd /D "%USERPROFILE%\Documents\spectrochempy"

      **3. Create and activate the environment.** This creates a ``scpy``
      folder in your working folder; uv downloads Python if needed:

      .. code-block:: bat

         uv venv scpy --python 3.13
         scpy\Scripts\activate

      The prompt now starts with ``(scpy)``.

      **4. Install SpectroChemPy:**

      .. code-block:: bat

         uv pip install spectrochempy

         :: or with interactive extras (Jupyter, etc.)
         uv pip install "spectrochempy[interactive]"

      Latest prerelease:

      .. code-block:: bat

         uv pip install spectrochempy --pre

      **5. Next time**, open the Command Prompt and run:

      .. code-block:: bat

         cd /D "%USERPROFILE%\Documents\spectrochempy"
         scpy\Scripts\activate

      or create the desktop shortcut described in
      `Creating a Desktop Shortcut`_, with this content:

      .. code-block:: bat

         @REM open a Command Prompt in the scpy environment
         @CD /D "%USERPROFILE%\Documents\spectrochempy"
         @CMD /K "%USERPROFILE%\Documents\spectrochempy\scpy\Scripts\activate.bat"

   .. tab:: Using conda / mamba

      **1. Install Miniforge.** Download the Windows installer
      (``Miniforge3-Windows-x86_64.exe``) from
      `Miniforge <https://github.com/conda-forge/miniforge>`_ and run it,
      keeping the default options. Note the installation folder it proposes
      (by default ``%USERPROFILE%\miniforge3``): you need it for the desktop
      shortcut. Skip this step if Miniforge (or Anaconda) is already
      installed.

      **2. Open the Miniforge Prompt** from the Start menu (type
      ``miniforge``). Use it instead of the plain Command Prompt, where
      ``conda`` and ``mamba`` are not available. Then go to your working
      folder. If the folder already exists, ``mkdir`` prints an error message
      that you can ignore:

      .. code-block:: bat

         mkdir "%USERPROFILE%\Documents\spectrochempy"
         cd /D "%USERPROFILE%\Documents\spectrochempy"

      **3. Add the required channels** (needed only once):

      .. code-block:: bat

         conda config --add channels conda-forge
         conda config --add channels spectrocat

      **4. Create and activate the environment:**

      .. code-block:: bat

         mamba create --name scpy python=3.13
         conda activate scpy

      The prompt now starts with ``(scpy)``. The environment is stored in
      Miniforge's own folder, not in your working folder.

      **5. Install SpectroChemPy:**

      .. code-block:: bat

         mamba install -c spectrocat spectrochempy

      Development versions are not published to a Conda ``dev`` channel.
      To use unreleased code, follow the source installation instructions in
      :doc:`install_sources`.

      **6. Next time**, open the Miniforge Prompt and run:

      .. code-block:: bat

         cd /D "%USERPROFILE%\Documents\spectrochempy"
         conda activate scpy

      or create the desktop shortcut described in
      `Creating a Desktop Shortcut`_, with this content (replace
      ``%USERPROFILE%\miniforge3`` if Miniforge is installed elsewhere):

      .. code-block:: bat

         @REM open a Command Prompt in the scpy environment
         @CD /D "%USERPROFILE%\Documents\spectrochempy"
         @CMD /K "%USERPROFILE%\miniforge3\Scripts\activate.bat" scpy

   .. tab:: Using pip and venv

      **1. Install Python** (3.11 or newer) from
      `python.org <https://www.python.org/downloads/windows/>`_. In the first
      screen of the installer, check **"Add python.exe to PATH"**.

      **2. Open the Command Prompt** and check that Python is found:

      .. code-block:: bat

         python --version

      It should print ``Python 3.11`` or newer. If the Microsoft Store opens
      instead, Python is not on the PATH: run the installer again and check
      "Add python.exe to PATH".

      **3. Go to your working folder.** If the folder already exists, ``mkdir``
      prints an error message that you can ignore:

      .. code-block:: bat

         mkdir "%USERPROFILE%\Documents\spectrochempy"
         cd /D "%USERPROFILE%\Documents\spectrochempy"

      **4. Create and activate the environment.** This creates a ``scpy``
      folder in your working folder:

      .. code-block:: bat

         python -m venv scpy
         scpy\Scripts\activate
         python -m pip install --upgrade pip

      The prompt now starts with ``(scpy)``.

      **5. Install SpectroChemPy:**

      .. code-block:: bat

         python -m pip install spectrochempy

         :: or if you want to install interactive version (including jupyter)
         python -m pip install "spectrochempy[interactive]"

      **6. Next time**, open the Command Prompt and run:

      .. code-block:: bat

         cd /D "%USERPROFILE%\Documents\spectrochempy"
         scpy\Scripts\activate

      or create the desktop shortcut described in
      `Creating a Desktop Shortcut`_, with this content:

      .. code-block:: bat

         @REM open a Command Prompt in the scpy environment
         @CD /D "%USERPROFILE%\Documents\spectrochempy"
         @CMD /K "%USERPROFILE%\Documents\spectrochempy\scpy\Scripts\activate.bat"

Verifying Installation
----------------------

With the ``scpy`` environment active (the prompt starts with ``(scpy)``):

1. Start IPython:

   .. code-block:: bat

      ipython

2. Import SpectroChemPy:

   .. code-block:: ipython

      In [1]: import spectrochempy as scp

Creating a Desktop Shortcut
---------------------------

A shortcut opens a Command Prompt in your working folder with the ``scpy``
environment already active, so you do not have to type the activation
commands each time.

1. Open Notepad and paste the content given at the end of your method's tab
   above (it differs between methods).
2. Choose *File > Save as*, select "All files (\*.*)" as file type, and save
   it as ``activate-scpy.bat`` in your working folder.
3. In the File Explorer, right-click ``activate-scpy.bat`` and choose
   *Send to > Desktop (create shortcut)* (on Windows 11, first choose
   *Show more options*).
4. Rename the shortcut on the desktop, for instance "SpectroChemPy prompt
   (scpy)".

Double-clicking the shortcut opens the prompt, which starts with ``(scpy)``.

Next Steps
----------

Proceed to the :ref:`user_guide` to start using SpectroChemPy.

.. note::
   If you encounter any issues, see :doc:`../getting_help` for support options.
