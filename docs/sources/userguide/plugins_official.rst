.. _official-plugins:

================
Official plugins
================

SpectroChemPy can be extended by plugins. Some plugins are maintained as part
of the SpectroChemPy project and are considered **official plugins**. They use
the same plugin discovery mechanism as third-party plugins, but their API,
examples, tests, and documentation are coordinated with the main project.

Official plugins should feel like part of SpectroChemPy once installed. They are
not loaded manually: import SpectroChemPy, then use the documented namespace.

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

    from spectrochempy.iris import IRIS

    analysis = IRIS()
    dataset = scp.nmr.read_topspin("path/to/fid")
    reactor = scp.cantera.PFR

Dataset accessors are reserved for operations that act on an existing dataset:

.. code-block:: python

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")

Some official plugins may expose limited root-level compatibility aliases, such
as ``scp.IRIS``. New examples should prefer the namespaced form,
``scp.iris.IRIS``.

Plugin summaries
================

IRIS
----

The ``spectrochempy-iris`` plugin provides 2D-IRIS analysis tools for
spectroscopic adsorption and diffusion studies.

.. code-block:: bash

    pip install spectrochempy[iris]

.. code-block:: python

    import spectrochempy as scp
    from spectrochempy.iris import IRIS

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")
    analysis = IRIS()

NMR
---

The ``spectrochempy-nmr`` plugin provides Bruker TopSpin reading and future
NMR-specific workflows.

.. code-block:: bash

    pip install spectrochempy[nmr]

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/fid")

Cantera
-------

The ``spectrochempy-cantera`` plugin provides simulation workflows based on
Cantera, currently centered on plug-flow reactor support.

.. code-block:: bash

    pip install spectrochempy[cantera]

.. code-block:: python

    import spectrochempy as scp

    PFR = scp.cantera.PFR

Examples and gallery convention
===============================

Examples remain organized by scientific topic in the central SpectroChemPy
gallery. Plugin-dependent examples live with the plugin source and are staged
into the central gallery from the plugin ``examples/gallery.toml`` manifest.
They should be clearly marked with their required plugin, but they do not need
a separate plugin-only gallery.
