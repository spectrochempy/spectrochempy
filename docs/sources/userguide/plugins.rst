.. _plugins:

=================
Optional Plugins
=================

SpectroChemPy's core works standalone for common spectroscopic data processing
and analysis.  Additional functionality is provided by **optional plugins** that
extend the framework with domain-specific readers, analyses, and simulations.

.. _plugins-install:

Installing a plugin
===================

Plugins are distributed as separate packages on PyPI.  Install them with
``pip``:

.. code-block:: bash

    pip install spectrochempy[nmr]      # NMR / TopSpin reader
    pip install spectrochempy[iris]     # 2D-IRIS analysis
    pip install spectrochempy[cantera]  # PFR reactor simulation

Or install a specific plugin directly:

.. code-block:: bash

    pip install spectrochempy-nmr
    pip install spectrochempy-iris
    pip install spectrochempy-cantera

.. _plugins-usage:

Using a plugin
==============

Once installed, the plugin registers itself automatically when you import
SpectroChemPy.  No extra import is needed:

.. code-block:: python

    import spectrochempy as scp

    # NMR plugin — read TopSpin files
    ds = scp.read_topspin("path/to/fid")

    # Cantera plugin — PFR simulation
    reactor = scp.PFR(...)

    # IRIS plugin — 2D-IRIS analysis
    from spectrochempy_iris import IRIS
    iris = IRIS(reg_par=[-10, 1, 12])

If a plugin is not installed, the corresponding function raises a clear
:class:`~spectrochempy.plugins.deps.MissingPluginError` with installation
instructions.

.. _plugins-list:

Available plugins
=================

.. list-table::
   :header-rows: 1

   * - Plugin
     - Package
     - Provides
   * - NMR
     - ``spectrochempy-nmr``
     - :func:`~spectrochempy.read_topspin` (Bruker TopSpin reader),
       NMR processing utilities
   * - IRIS
     - ``spectrochempy-iris``
     - 2D-IRIS analysis
       (:class:`~spectrochempy_iris.IRIS`,
       :class:`~spectrochempy_iris.IrisKernel`)
   * - Cantera
     - ``spectrochempy-cantera``
     - :class:`~spectrochempy.PFR` plug flow reactor simulation

.. _plugins-developer:

Developing a plugin
===================

See the :ref:`devguide-plugins` section for the plugin API documentation,
packaging guide, and a full reference example (``spectrochempy-nmr``).
