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

.. warning::

   A certificate-authority incident announced in October 2026 affects old
   package-management tools. Before downloading SpectroChemPy or its plugins,
   update the package manager you use: ``pip`` to 26.2.2 or newer, or ``uv`` to
   0.13.4 or newer. Update ``certifi`` to 2026.10.19 or newer when it is
   installed. The Python
   Packaging Authority provides a `hash-verified pip upgrade command and
   current guidance <https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576>`_.

Quick Install (experienced users)
---------------------------------

For users already familiar with Python environments: SpectroChemPy requires
Python 3.11 or newer and is available on PyPI and on the ``spectrocat`` conda
channel. In an activated environment:

.. code-block:: bash

   uv pip install spectrochempy                    # or: python -m pip install spectrochempy
   mamba install -c conda-forge -c spectrocat spectrochempy

With pip or uv, the ``[interactive]`` extra (``"spectrochempy[interactive]"``)
also installs Jupyter; with conda / mamba, install ``jupyterlab`` separately.
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

.. toctree::
    :maxdepth: 3
    :hidden:

    install_win
    install_mac
    install_linux
    install_colab
    install_adds
    install_sources
