.. _nmr-plugin:

==========
NMR plugin
==========

Introduction
============

The ``spectrochempy-nmr`` plugin provides NMR-specific readers and processing
workflows, including the Bruker TopSpin reader.

Installation
============

Install it with:

.. code-block:: bash

    pip install spectrochempy[nmr]

Recommended API
===============

Use the recommended namespaced API:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/fid")

The NMR plugin owns Bruker/TopSpin conventions such as experiment directory
resolution, processed-data defaults, acquisition metadata, and NMR unit
contexts. Core SpectroChemPy remains responsible for generic datasets, units,
plotting, and ordinary FFT operations.

Compatibility aliases
=====================

The plugin-owned TopSpin reader is documented here rather than in the core API
reference because it is provided by ``spectrochempy-nmr``, not by the core
package itself.

Compatibility alias:

``scp.read_topspin(...)`` remains available when the plugin is installed, but
new documentation and examples should prefer ``scp.nmr.read_topspin(...)``.

API Reference
=============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.read_topspin

Examples
========

For phase-sensitive 2D NMR workflows, install hypercomplex support as well:

.. code-block:: bash

    pip install spectrochempy[nmr,hypercomplex]

See also the hypercomplex plugin guide for phase-sensitive 2D NMR workflows
built on TopSpin datasets.

Limitations
===========

Currently, the user-facing documentation for this plugin focuses on TopSpin
datasets and related NMR workflows.
