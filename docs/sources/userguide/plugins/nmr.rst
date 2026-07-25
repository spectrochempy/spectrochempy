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

    dataset = scp.nmr.read("path/to/fid")

The NMR plugin owns Bruker/TopSpin conventions such as experiment directory
resolution, processed-data defaults, acquisition metadata, and NMR unit
contexts. Core SpectroChemPy remains responsible for generic datasets, units,
plotting, and ordinary FFT operations.

TopSpin Processing Profile
==========================

TopSpin datasets may expose a descriptive vendor processing profile in
``dataset.meta.nmr_processing``:

.. code-block:: python

    dataset = scp.nmr.read("path/to/fid")
    profile = dataset.meta.nmr_processing["vendor_profile"]

This profile records values read from the vendor ``procs`` file. It is
descriptive only:

- it is not applied automatically when reading a raw FID;
- it is not replayed automatically by ``scp.nmr.Experiment(...).process()``;
- it does not establish the exact historical processing sequence of an already
  processed vendor spectrum such as TopSpin ``1r``;
- SpectroChemPy-owned ``requested`` / ``applied`` processing traces are handled
  separately and are not yet part of this metadata.

Compatibility aliases
=====================

Compatibility aliases:

- ``scp.nmr.read_topspin(...)`` remains available as the explicit historical
  namespaced form.
- ``scp.read_topspin(...)`` remains available as the root-level compatibility
  alias when the plugin is installed.

New documentation and examples should prefer the shorter
``scp.nmr.read(...)`` form.

API Reference
=============

The generated public API page for the NMR plugin is listed in
:doc:`/reference/plugins`. See :func:`spectrochempy.nmr.read`.

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
