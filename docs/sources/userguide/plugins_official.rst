.. _official-plugins:

================
Official plugins
================

SpectroChemPy can be extended by plugins. Some plugins are maintained as part
of the SpectroChemPy project and are considered **official plugins**. They use
the same plugin discovery mechanism as third-party plugins, but their API and
documentation are coordinated with the main project.

Current official plugins
========================

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Domain
     - Package
     - Namespace
     - Typical use
   * - IRIS
     - ``spectrochempy-iris``
     - ``scp.iris``, ``dataset.iris``
     - 2D-IRIS analysis, IRIS kernels, dataset-bound IRIS operations.
   * - NMR
     - ``spectrochempy-nmr``
     - ``scp.nmr``
     - TopSpin/Bruker NMR reading and future NMR-specific processing.
   * - Cantera / simulation
     - ``spectrochempy-cantera``
     - ``scp.cantera``
     - Plug-flow reactor and simulation workflows based on Cantera.

Install commands
================

Official plugins can be installed through SpectroChemPy extras:

.. code-block:: bash

    pip install spectrochempy[iris]
    pip install spectrochempy[nmr]
    pip install spectrochempy[cantera]

or directly:

.. code-block:: bash

    pip install spectrochempy-iris
    pip install spectrochempy-nmr
    pip install spectrochempy-cantera

Once installed, plugins are discovered automatically. No explicit plugin
loading call is required in user code.

API convention
==============

The recommended form is namespaced:

.. code-block:: python

    import spectrochempy as scp

    analysis = scp.iris.IRIS()
    dataset = scp.nmr.read_topspin("path/to/fid")
    reactor = scp.cantera.PFR

Dataset accessors are reserved for operations that act on an existing dataset:

.. code-block:: python

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")

Some official plugins may expose limited root-level compatibility aliases, such
as ``scp.IRIS``. New examples should prefer the namespaced form,
``scp.iris.IRIS``.

Examples and gallery convention
===============================

Examples remain organized by scientific topic in the central SpectroChemPy
gallery. Plugin-dependent examples should be clearly marked with their required
plugin, but they do not need a separate plugin-only gallery.
