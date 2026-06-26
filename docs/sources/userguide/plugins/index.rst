.. _plugins:

=======
Plugins
=======

SpectroChemPy plugins add optional scientific features while keeping the core
installation lighter. Once a plugin is installed, it is discovered
automatically; normal user code does not need a manual loading step.

This section answers three user-facing questions:

* Which plugins are official and supported?
* How should plugin APIs be used in normal code?
* Where should plugin-dependent workflows and examples be documented?

Quick example
=============

Plugin APIs are usually exposed through namespaces:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/fid")
    dataset = scp.perkinelmer.read("path/to/file.sp")
    analysis = scp.iris.IRIS()
    model = scp.tensor.CP(n_components=2)

Operations that act on an existing dataset use dataset accessors:

.. code-block:: python

    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")
    dataset.hyper.set_quaternion(inplace=True)
    ri = dataset.hyper.component("RI")

Some former top-level names remain as compatibility aliases. New code should
prefer namespaced APIs such as ``scp.nmr.read_topspin``, ``scp.iris.IRIS``,
and ``scp.tensor.CP``.

Install and inspect plugins
===========================

Official plugins can be installed either through SpectroChemPy extras or by
installing the plugin package directly. Once installed, use ``scp.plugins()``
to inspect what SpectroChemPy discovered in the current environment.

Use ``scp.plugins()`` to list discovered plugins:

.. code-block:: python

    import spectrochempy as scp

    scp.plugins()
    scp.plugins(verbose=True)

If an official optional feature is missing, SpectroChemPy raises a clear
installation hint instead of failing with an import error.

For installation details, package lists, and per-plugin summaries, see
:doc:`official_plugins`.

For experimental plugin status, see :doc:`experimental_plugins`.

For example conventions, see :doc:`examples`.

For the user-facing plugin direction, see :doc:`roadmap`.

For implementation details and architecture planning, see :ref:`plugin-devguide`.

Plugin pages
============

.. toctree::
   :maxdepth: 1

   official_plugins
   nmr
   tensor
   hypercomplex
   iris
   perkinelmer
   experimental_plugins
   examples
   roadmap
