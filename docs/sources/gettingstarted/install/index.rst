.. _installation:

Installation Guide
==================

Choose the guide for your computer. Each one gives complete, step-by-step
instructions, from installing the tools to checking that SpectroChemPy works:

* :doc:`install_win`
* :doc:`install_mac_linux`
* :doc:`install_colab`: run SpectroChemPy in your browser, with nothing to
  install on your computer

Other cases:

* :doc:`install_sources`: development version, contributors
* :doc:`install_adds`: optional components and plugins

Quick Install
-------------

For users already familiar with Python environments: SpectroChemPy requires
Python 3.11 or newer and is available on PyPI and on the ``spectrocat`` conda
channel.

.. code-block:: bash

   uv pip install spectrochempy                    # or: python -m pip install spectrochempy
   mamba install -c conda-forge -c spectrocat spectrochempy

Add the ``[interactive]`` extra (``"spectrochempy[interactive]"``) to also
install Jupyter. To pin an exact version, use ``"spectrochempy==1.0.0"`` (pip,
uv) or ``"spectrochempy=1.0.0"`` (conda, mamba).

.. toctree::
    :maxdepth: 3
    :hidden:

    install_win
    install_mac_linux
    install_colab
    install_adds
    install_sources
