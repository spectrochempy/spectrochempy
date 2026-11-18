.. _installation:

Installation Guide
==================

Step-by-step Guides
-------------------

Choose the guide for your computer. Each one gives complete, step-by-step
instructions, from installing the tools to checking that SpectroChemPy works:

* :doc:`install_win`
* :doc:`install_mac`
* :doc:`install_linux`
* :doc:`install_colab`: run SpectroChemPy in your browser, with nothing to
  install on your computer

Other cases:

* :doc:`install_sources`: development version, contributors
* :doc:`install_adds`: plugins, example data and other optional packages

.. note::

   **Installation security notice**

   Due to a recent security issue affecting Python package managers, please
   make sure your installation tools are up to date before installing
   SpectroChemPy.

   See the `official Python Packaging Authority guidance
   <https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576>`_ for instructions.

Quick Install (experienced users)
---------------------------------

For users already familiar with Python environments: SpectroChemPy requires
Python 3.11 or newer and is available on PyPI and on the ``spectrocat`` conda
channel. In an activated environment:

.. code-block:: bash

   uv pip install spectrochempy                    # or: python -m pip install spectrochempy
   mamba install -c conda-forge -c spectrocat spectrochempy

To run the tutorial notebooks, also install JupyterLab (``uv pip install jupyterlab``,
``python -m pip install jupyterlab`` or ``mamba install -c conda-forge jupyterlab``).
To pin an exact version, use ``"spectrochempy==1.0.0"`` (pip, uv) or
``"spectrochempy=1.0.0"`` (conda, mamba). To install a pre-release, when one is
available: ``uv pip install spectrochempy --prerelease allow`` (or
``python -m pip install --pre spectrochempy``).

To install the development version (the current ``master`` branch, documented
as *latest* on this site), `git <https://git-scm.com>`_ must be installed:

.. code-block:: bash

   uv pip install "spectrochempy @ git+https://github.com/spectrochempy/spectrochempy.git"
   # or: python -m pip install "spectrochempy @ git+https://github.com/spectrochempy/spectrochempy.git"

Development versions are not published on the conda channels. To modify the
code or contribute, see :doc:`install_sources`.

Update notifications
--------------------

After application initialization, SpectroChemPy checks public PyPI for newer
versions of the core package and installed plugins registered through
``spectrochempy.plugins``. Plugin modules are not loaded for this check.
Plugins installed directly from a directory, archive URL or version-control
repository are excluded. Results are cached separately for each Python
environment; the default interval for checks and reminders is one week.
Checks run in the background and do not install anything.

Available updates are grouped into one native desktop notification. Linux
requires ``notify-send`` and a desktop session bus; macOS uses ``osascript``;
Windows uses Windows PowerShell and its registered notification identity.
The operating system may identify the sender as Script Editor or PowerShell.
If the tool is missing, fails or the session uses SSH, a text warning is emitted
instead. A notebook on a remote server cannot notify your local desktop through
this mechanism. Desktop settings, including Do Not Disturb, can hide notifications
even when delivery succeeds.

Configure delivery and frequency with preferences:

.. code-block:: python

   import spectrochempy as scp

   scp.preferences.update_notifications = "system"    # native desktop (default)
   scp.preferences.check_update_frequency = "week"    # day, week, month, never

Use ``update_notifications = "terminal"`` for text warnings, or ``"off"`` to
disable automatic checks and notifications. ``check_update_frequency = "never"``
also disables automatic checks. Preferences are saved for subsequent sessions.
To disable checks before the first initialization, set the environment variable
``SCP_CHECK_UPDATES=0`` before starting Python. Documentation builds do not check
for updates.

To request a fresh status report without generating a notification, run this
command in the environment you want to inspect:

.. code-block:: bash

   python -m spectrochempy.application.check_update

Add ``--notify`` to also request a system notification. This explicit command
bypasses the saved check interval and delivery preferences, but respects
``SCP_CHECK_UPDATES=0`` and documentation-build suppression. Reports distinguish
available updates, current packages, unavailable responses and checks deferred
by the request budget. Failed requests are retried no sooner than one hour later
during automatic checks. The checker starts no further requests after its
15-second request budget; individual request timeouts are at most five seconds.
These are request limits, not a hard deadline for the whole background task.
Cache writes are atomic and best effort; simultaneous processes can still emit
duplicate reminders.

The selected release matches the installed stable/prerelease channel and the
running Python version. Availability on PyPI does not guarantee compatibility
with other installed packages or the operating system. Review the upgrade with
your existing pip, uv or conda workflow. No automatic installation is performed.

.. toctree::
    :maxdepth: 3
    :hidden:

    install_win
    install_mac
    install_linux
    install_colab
    install_adds
    install_sources
