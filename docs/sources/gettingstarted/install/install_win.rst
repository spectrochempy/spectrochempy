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

Two folders are involved, and it is good practice to keep them separate:

* the **environment**, named ``scpy``, which contains Python, SpectroChemPy
  and its dependencies. You never need to open or modify it. With uv and
  venv, this guide puts it in ``%USERPROFILE%\scpy`` (that is,
  ``C:\Users\<yourName>\scpy``); with conda / mamba, Miniforge stores it in
  its own folder.
* your **working folder**, which contains your data, scripts and notebooks.
  It can be any folder you like, for instance a ``spectrochempy`` folder in
  your *Documents*. Below, it is written ``<yourWorkingFolder>``: replace it
  with the full path of your folder, which you can copy from the address bar
  of the File Explorer (for instance
  ``C:\Users\<yourName>\Documents\spectrochempy``).

.. warning::

   Do not create the environment in a folder synchronized with the cloud
   (OneDrive, Google Drive, Dropbox...): it contains thousands of files,
   which slows down the synchronization and can break the environment. On
   many computers, OneDrive synchronizes the *Documents* and *Desktop*
   folders. Your working folder, on the other hand, can be synchronized.

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

      **2. Open the Command Prompt.**

      **3. Create and activate the environment** in ``%USERPROFILE%\scpy``
      (uv downloads Python if needed):

      .. code-block:: bat

         uv venv "%USERPROFILE%\scpy" --python 3.13
         "%USERPROFILE%\scpy\Scripts\activate"

      The prompt now starts with ``(scpy)``.

      **4. Install SpectroChemPy:**

      .. code-block:: bat

         uv pip install spectrochempy

         :: or with interactive extras (Jupyter, etc.)
         uv pip install "spectrochempy[interactive]"

      Latest prerelease:

      .. code-block:: bat

         uv pip install spectrochempy --pre

      **5. Create a desktop shortcut.** It opens a Command Prompt in your
      working folder with the ``scpy`` environment already active, so you do not
      have to type the activation commands each time.

      * Open Notepad and paste the following lines. Replace
        ``<yourWorkingFolder>`` with the path of your working folder, keeping the
        quotes:

        .. code-block:: bat

            @REM open a Command Prompt in the scpy environment
            @CD /D "<yourWorkingFolder>"
            @CMD /K "%USERPROFILE%\scpy\Scripts\activate.bat"

      * Choose *File > Save as*, select "All files (\*.*)" as file type, and
        save it as ``activate-scpy.bat`` in your working folder.
      * In the File Explorer, right-click ``activate-scpy.bat`` and choose
        *Send to > Desktop (create shortcut)* (on Windows 11, first choose
        *Show more options*).
      * Rename the shortcut on the desktop, for instance "SpectroChemPy prompt
        (scpy)".

      **6. Check the installation.** Double-click the shortcut: the prompt
      opens in your working folder and starts with ``(scpy)``. Start IPython and
      import SpectroChemPy:

      .. code-block:: bat

         ipython

      .. code-block:: ipython

         In [1]: import spectrochempy as scp

      Type ``exit`` to leave IPython. From now on, use the shortcut each time
      you work with SpectroChemPy. Without the shortcut, open the
      Command Prompt and run:

      .. code-block:: bat

         cd /D "<yourWorkingFolder>"
         "%USERPROFILE%\scpy\Scripts\activate"

   .. tab:: Using conda / mamba

      **1. Install Miniforge.** Download the Windows installer
      (``Miniforge3-Windows-x86_64.exe``) from
      `Miniforge <https://github.com/conda-forge/miniforge>`_ and run it,
      keeping the default options. Note the installation folder it proposes
      (by default ``%USERPROFILE%\miniforge3``): you need it for the desktop
      shortcut (step 6). Skip this step if Miniforge (or Anaconda) is already
      installed.

      **2. Open the Miniforge Prompt** from the Start menu (type
      ``miniforge``). Use it instead of the plain Command Prompt, where
      ``conda`` and ``mamba`` are not available.

      **3. Add the required channels** (needed only once):

      .. code-block:: bat

         conda config --add channels conda-forge
         conda config --add channels spectrocat

      **4. Create and activate the environment:**

      .. code-block:: bat

         mamba create --name scpy python=3.13
         conda activate scpy

      The prompt now starts with ``(scpy)``. The environment is stored in
      Miniforge's own folder (``envs\scpy``), separate from your working
      folder.

      **5. Install SpectroChemPy:**

      .. code-block:: bat

         mamba install -c spectrocat spectrochempy

      Development versions are not published to a Conda ``dev`` channel.
      To use unreleased code, follow the source installation instructions in
      :doc:`install_sources`.

      **6. Create a desktop shortcut.** It opens a Command Prompt in your
      working folder with the ``scpy`` environment already active, so you do not
      have to type the activation commands each time.

      * Open Notepad and paste the following lines. Replace
        ``<yourWorkingFolder>`` with the path of your working folder, keeping the
        quotes. If Miniforge is not installed in ``%USERPROFILE%\miniforge3``,
        also replace that path with its installation folder:

        .. code-block:: bat

            @REM open a Command Prompt in the scpy environment
            @CD /D "<yourWorkingFolder>"
            @CMD /K "%USERPROFILE%\miniforge3\Scripts\activate.bat" scpy

      * Choose *File > Save as*, select "All files (\*.*)" as file type, and
        save it as ``activate-scpy.bat`` in your working folder.
      * In the File Explorer, right-click ``activate-scpy.bat`` and choose
        *Send to > Desktop (create shortcut)* (on Windows 11, first choose
        *Show more options*).
      * Rename the shortcut on the desktop, for instance "SpectroChemPy prompt
        (scpy)".

      **7. Check the installation.** Double-click the shortcut: the prompt
      opens in your working folder and starts with ``(scpy)``. Start IPython and
      import SpectroChemPy:

      .. code-block:: bat

         ipython

      .. code-block:: ipython

         In [1]: import spectrochempy as scp

      Type ``exit`` to leave IPython. From now on, use the shortcut each time
      you work with SpectroChemPy. Without the shortcut, open the
      Miniforge Prompt and run:

      .. code-block:: bat

         cd /D "<yourWorkingFolder>"
         conda activate scpy

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

      **3. Create and activate the environment** in ``%USERPROFILE%\scpy``:

      .. code-block:: bat

         python -m venv "%USERPROFILE%\scpy"
         "%USERPROFILE%\scpy\Scripts\activate"
         python -m pip install --upgrade pip

      The prompt now starts with ``(scpy)``.

      **4. Install SpectroChemPy:**

      .. code-block:: bat

         python -m pip install spectrochempy

         :: or if you want to install interactive version (including jupyter)
         python -m pip install "spectrochempy[interactive]"

      **5. Create a desktop shortcut.** It opens a Command Prompt in your
      working folder with the ``scpy`` environment already active, so you do not
      have to type the activation commands each time.

      * Open Notepad and paste the following lines. Replace
        ``<yourWorkingFolder>`` with the path of your working folder, keeping the
        quotes:

        .. code-block:: bat

            @REM open a Command Prompt in the scpy environment
            @CD /D "<yourWorkingFolder>"
            @CMD /K "%USERPROFILE%\scpy\Scripts\activate.bat"

      * Choose *File > Save as*, select "All files (\*.*)" as file type, and
        save it as ``activate-scpy.bat`` in your working folder.
      * In the File Explorer, right-click ``activate-scpy.bat`` and choose
        *Send to > Desktop (create shortcut)* (on Windows 11, first choose
        *Show more options*).
      * Rename the shortcut on the desktop, for instance "SpectroChemPy prompt
        (scpy)".

      **6. Check the installation.** Double-click the shortcut: the prompt
      opens in your working folder and starts with ``(scpy)``. Start IPython and
      import SpectroChemPy:

      .. code-block:: bat

         ipython

      .. code-block:: ipython

         In [1]: import spectrochempy as scp

      Type ``exit`` to leave IPython. From now on, use the shortcut each time
      you work with SpectroChemPy. Without the shortcut, open the
      Command Prompt and run:

      .. code-block:: bat

         cd /D "<yourWorkingFolder>"
         "%USERPROFILE%\scpy\Scripts\activate"

Next Steps
----------

Proceed to the :ref:`user_guide` to start using SpectroChemPy.

.. note::
   If you encounter any issues, see :doc:`../getting_help` for support options.
