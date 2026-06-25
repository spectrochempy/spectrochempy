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
   * - Tensor
     - ``spectrochempy-tensor``
     - ``scp.tensor``
     - TensorLy-backed tensor decompositions such as CP/PARAFAC.
   * - Hypercomplex
     - ``spectrochempy-hypercomplex``
     - ``dataset.hyper``
     - Quaternion/hypercomplex support for phase-sensitive 2D NMR.
   * - Carroucell
     - ``spectrochempy-carroucell``
     - ``scp.carroucell``
     - Carroucell experiment reader for spectroscopic data.
   * - PerkinElmer
     - ``spectrochempy-perkinelmer``
     - ``scp.perkinelmer``
     - PerkinElmer ``.sp`` IR file reader.

Install commands
================

Official plugins can be installed through SpectroChemPy extras:

.. code-block:: bash

    pip install spectrochempy[iris]
    pip install spectrochempy[nmr]
    pip install spectrochempy[tensor]
    pip install spectrochempy[nmr,hypercomplex]  # NMR with 2D hypercomplex support
    pip install spectrochempy[perkinelmer]

or directly:

.. code-block:: bash

    pip install spectrochempy-iris
    pip install spectrochempy-nmr
    pip install spectrochempy-tensor
    pip install spectrochempy-hypercomplex
    pip install spectrochempy-carroucell
    pip install spectrochempy-perkinelmer

Once installed, plugins are discovered automatically. No explicit plugin
loading call is required in user code.

Versions and compatibility
==========================

Official plugins are versioned independently from the SpectroChemPy core
package. A core release and a plugin release therefore do not need to share the
same version number. For example, a stable ``spectrochempy 0.9.2`` environment
may use ``spectrochempy-nmr 0.1.3``.

Stable plugin packages are published on PyPI and on the main ``spectrocat``
conda channel. Conda development builds for official plugins are currently not
published automatically; they will be re-enabled once plugin development builds
use versions distinct from stable releases.

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
    model = scp.tensor.CP(n_components=2)
    dataset = scp.nmr.read_topspin("path/to/fid")

Dataset accessors are reserved for operations that act on an existing dataset:

.. code-block:: python

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")

Some official plugins may expose limited root-level compatibility aliases, such
as ``scp.IRIS``. New examples should prefer the namespaced form,
``scp.iris.IRIS`` or ``scp.tensor.CP``.

Plugin summaries
================

The table above is the quick reference. The dedicated plugin pages below
provide the user-facing details:

* :doc:`iris` for 2D-IRIS workflows and the ``scp.iris`` / ``dataset.iris`` APIs
* :doc:`nmr` for TopSpin reading and NMR-specific processing workflows
* :doc:`tensor` for TensorLy-backed tensor decompositions such as CP/PARAFAC
* :doc:`hypercomplex` for quaternion support used in phase-sensitive 2D NMR
* ``scp.perkinelmer.read_perkinelmer`` for PerkinElmer ``.sp`` IR files

The Carroucell reader is currently part of the official plugin set but does not
yet have a separate user page. Use ``scp.carroucell.read_carroucell(...)`` once
the plugin is installed.

Examples and gallery convention
===============================

Examples remain organized by scientific topic in the central SpectroChemPy
gallery. Plugin-dependent examples live with the plugin source and are staged
into the central gallery from the plugin ``examples/gallery.toml`` manifest.
They should be clearly marked with their required plugin, but they do not need
a separate plugin-only gallery. See :doc:`examples` for the user-facing example
convention.
