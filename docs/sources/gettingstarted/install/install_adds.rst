.. _install_adds:

Optional Dependencies
=====================

.. warning::
   Make sure your base installation is working before adding optional dependencies.

Overview
--------
SpectroChemPy can be extended with optional dependencies for specific functionality.
Missing dependencies will only raise an ``ImportError`` when the specific feature is used.

Examples and Testing Data
-------------------------

.. note::
   Example data is not included in the main package to reduce installation size.

Installation options:

.. tabs::

   .. tab:: Using mamba (recommended)

      .. code-block:: bash

         mamba install -c spectrocat spectrochempy_data

   .. tab:: Manual download

      Download from `GitHub releases <https://github.com/spectrochempy/spectrochempy_data/tags>`__
      and extract to your preferred location.

      .. note::
         You may need to configure data paths when using manual installation.

Scientific Computing Extensions
-------------------------------

Cantera
~~~~~~~
Required for chemical kinetics, thermodynamics and transport processes.
(See `Cantera documentation <https://cantera.org>`__)

.. tabs::

   .. tab:: Using mamba (recommended)

      Stable version:

      .. code-block:: bash

         mamba install -c cantera cantera>=2.6.0

      Development version:

      .. code-block:: bash

         mamba install -c cantera/label/dev cantera

   .. tab:: Using pip

      .. code-block:: bash

         pip install cantera>=2.6.0

GUI Support
-----------

Qt Backend
~~~~~~~~~~
For interactive matplotlib plots:

.. tabs::

   .. tab:: Using mamba (recommended)

      .. code-block:: bash

         mamba install "pyqt>=5.15.0"

   .. tab:: Using pip

      .. code-block:: bash

         pip install "pyqt5>=5.15.0"

Usage in Jupyter:

.. code-block:: ipython

   %matplotlib qt

.. note::
   Qt backend is recommended for detailed plot manipulation and better interactivity.

Troubleshooting
---------------
If you encounter issues:

1. Verify base installation: ``python -c "import spectrochempy"``
2. Check dependency versions: ``mamba list`` / ``pip list``
3. Update all packages: ``mamba update --all`` / ``pip install --upgrade spectrochempy``
4. See :doc:`../getting_help` for support
