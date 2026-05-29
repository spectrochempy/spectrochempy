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
     - TopSpin/Bruker NMR reading and NMR-specific processing utilities.
   * - Hypercomplex
     - ``spectrochempy-hypercomplex``
     - ``dataset.hyper``
     - Quaternion/hypercomplex support for phase-sensitive 2D NMR.
   * - Carroucell
     - ``spectrochempy-carroucell``
     - ``scp.carroucell``
     - Carroucell experiment reader for spectroscopic data.

Install commands
================

Official plugins can be installed through SpectroChemPy extras:

.. code-block:: bash

    pip install spectrochempy[iris]
    pip install spectrochempy[nmr]
    pip install spectrochempy[nmr,hypercomplex]  # NMR with 2D hypercomplex support

or directly:

.. code-block:: bash

    pip install spectrochempy-iris
    pip install spectrochempy-nmr
    pip install spectrochempy-hypercomplex
    pip install spectrochempy-carroucell

Once installed, plugins are discovered automatically. No explicit plugin
loading call is required in user code.

Inspect installed plugins
=========================

Use ``scp.plugins()`` to see which official plugins are installed and which
plugin namespaces are available:

.. code-block:: python

    import spectrochempy as scp

    scp.plugins()

Use ``scp.plugins(verbose=True)`` to include lightweight package metadata such
as versions. This inspection command reads entry-point and package metadata; it
does not import optional plugin implementation modules.

API convention
==============

The recommended form is namespaced:

.. code-block:: python

    import spectrochempy as scp

    analysis = scp.iris.IRIS()
    dataset = scp.nmr.read_topspin("path/to/fid")

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

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")
    analysis = scp.iris.IRIS()

NMR
---

The ``spectrochempy-nmr`` plugin provides Bruker TopSpin reading and
NMR-specific workflows.

.. code-block:: bash

    pip install spectrochempy[nmr]

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/fid")

Hypercomplex
------------

The ``spectrochempy-hypercomplex`` plugin provides quaternion/hypercomplex
support for multi-dimensional complex data, primarily used in phase-sensitive
2D NMR.

.. code-block:: bash

    pip install spectrochempy-hypercomplex

.. code-block:: python

    import spectrochempy as scp

    # After reading 2D NMR data, convert to hypercomplex
    dataset = scp.nmr.read_topspin("path/to/ser")
    dataset.hyper.set_quaternion(inplace=True)

    # Extract components
    rr = dataset.hyper.RR
    ri = dataset.hyper.component("RI")

Carroucell
----------

The ``spectrochempy-carroucell`` plugin provides the Carroucell experiment
reader for spectroscopic data analysis.

.. code-block:: bash

    pip install spectrochempy-carroucell

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.carroucell.read_carroucell("path/to/carroucell_dir")

Examples and gallery convention
===============================

Examples remain organized by scientific topic in the central SpectroChemPy
gallery. Plugin-dependent examples live with the plugin source and are staged
into the central gallery from the plugin ``examples/gallery.toml`` manifest.
They should be clearly marked with their required plugin, but they do not need
a separate plugin-only gallery.
