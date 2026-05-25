.. _plugins:

=======
Plugins
=======

SpectroChemPy plugins add optional scientific features while keeping the core
installation lighter. Once a plugin is installed, it is discovered
automatically; normal user code does not need a manual loading step.

Install official plugins with extras:

.. code-block:: bash

    pip install spectrochempy[nmr]
    pip install spectrochempy[iris]
    pip install spectrochempy[cantera]
    pip install spectrochempy[nmr,hypercomplex]

or install plugin packages directly:

.. code-block:: bash

    pip install spectrochempy-nmr
    pip install spectrochempy-iris
    pip install spectrochempy-cantera
    pip install spectrochempy-hypercomplex
    pip install spectrochempy-carroucell

Conda users can install from the ``spectrocat`` channel::

    mamba install -c spectrocat -c conda-forge spectrochempy-nmr

Development (pre-release) builds are available on the ``dev`` label::

    mamba install -c spectrocat/label/dev -c conda-forge spectrochempy-nmr

Using plugins
=============

Plugin APIs are exposed through namespaces:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/fid")
    analysis = scp.iris.IRIS()
    reactor = scp.cantera.PFR

Operations that act on an existing dataset use dataset accessors:

.. code-block:: python

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")
    dataset.hyper.set_quaternion(inplace=True)
    ri = dataset.hyper.component("RI")

Some former top-level names remain as compatibility aliases. New code should
prefer namespaced APIs such as ``scp.nmr.read_topspin`` and ``scp.iris.IRIS``.

Inspecting plugins
==================

Use ``scp.plugins()`` to list discovered plugins:

.. code-block:: python

    import spectrochempy as scp

    scp.plugins()
    scp.plugins(verbose=True)

If an official optional feature is missing, SpectroChemPy raises a clear
installation hint instead of failing with an import error.

User roadmap
============

The user-facing direction is simple: official plugins should feel integrated
once installed, examples should state their plugin requirements, and common
core workflows should continue to work without optional plugin dependencies.

For implementation details and architecture planning, see
:ref:`plugin-dev-roadmap`.

Plugin pages
============

.. toctree::
   :maxdepth: 1

   official_plugins
   nmr
   hypercomplex
   iris
   cantera
   examples
   roadmap
