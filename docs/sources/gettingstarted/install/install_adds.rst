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

.. _install_adds_plugins:

Optional Plugins
----------------

SpectroChemPy includes several optional plugins that extend its capabilities
for specific domains. Install them as needed:

.. tabs::

    .. tab:: Using pip (recommended)

       .. code-block:: bash

          pip install spectrochempy[cantera]   # Chemical kinetics, thermodynamics
          pip install spectrochempy[iris]      # 2D-IRIS analysis
          pip install spectrochempy[nmr]       # Bruker TopSpin reader & NMR processing

    .. tab:: Using mamba (recommended)

       .. code-block:: bash

          mamba install -c spectrocat spectrochempy-cantera   # Chemical kinetics
          mamba install -c spectrocat spectrochempy-iris    # 2D-IRIS analysis
          mamba install -c spectrocat spectrochempy-nmr     # Bruker TopSpin & NMR
          mamba install -c spectrocat spectrochempy-hypercomplex  # Quaternion support

    .. tab:: Development install (from source)

      .. code-block:: bash

         pip install -e plugins/spectrochempy-cantera
         pip install -e plugins/spectrochempy-iris
         pip install -e plugins/spectrochempy-nmr

Each plugin is independently versioned and maintained. Missing plugins
raise a clear ``MissingPluginError`` with installation instructions when
the corresponding feature is accessed.

.. seealso::

   :doc:`/userguide/plugins/index` for more details on the plugin system.

Cantera
~~~~~~~

.. note::
   Cantera workflows are now provided by the official
   ``spectrochempy-cantera`` plugin (see
   :ref:`Optional Plugins <install_adds_plugins>` above). This keeps the
   standard installation lighter while preserving access to the workflow when
   the plugin is installed.

For users who still need the Cantera library outside of SpectroChemPy,
it can be installed separately:

.. tabs::

   .. tab:: Using mamba (recommended)

      .. code-block:: bash

         mamba install -c cantera cantera

   .. tab:: Using pip

      .. code-block:: bash

         pip install cantera

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
