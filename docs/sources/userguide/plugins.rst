.. _plugins:

================
Optional Plugins
================

SpectroChemPy's core works standalone for common spectroscopic data processing
and analysis. Additional functionality is provided by **optional plugins** that
extend the framework with domain-specific readers, analyses, and simulations.

.. _plugins-install:

Installing a plugin
===================

Plugins are distributed as separate packages on PyPI. Install them with
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
SpectroChemPy. Plugin functions are available from package-level namespaces,
and dataset-bound operations are available from dataset accessors:

.. code-block:: python

    import spectrochempy as scp

    # NMR plugin: read TopSpin files
    ds = scp.nmr.read_topspin("path/to/fid")

    # Cantera plugin: access the PFR simulation callable
    PFR = scp.cantera.PFR

    # IRIS plugin: build an IRIS kernel from an existing dataset
    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")

For backward compatibility, some former top-level APIs remain available as thin
aliases. For example, ``scp.read_topspin(...)`` delegates to
``scp.nmr.read_topspin(...)`` when the NMR plugin is installed.

If a plugin is not installed, the corresponding official optional feature gives
a clear installation hint. For example:

.. code-block:: pycon

    >>> import spectrochempy as scp
    >>> scp.read_topspin("path/to/fid")
    Traceback (most recent call last):
    ...
    MissingPluginError: The 'read_topspin' feature requires the optional plugin
    'spectrochempy-nmr'. Install it with: pip install spectrochempy[nmr]

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
     - ``scp.cantera.PFR`` plug flow reactor simulation callable

.. _plugins-developer:

Developing a plugin
===================

See the :ref:`devguide-plugins` section for the plugin API documentation,
packaging guide, and a full reference example (``spectrochempy-nmr``).
