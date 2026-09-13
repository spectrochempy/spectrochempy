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
- SpectroChemPy-owned processing traces are handled separately from the vendor
  profile.

SpectroChemPy Processing Trace
==============================

The result of ``scp.nmr.Experiment(...).process()`` records the
SpectroChemPy-owned processing call in
``result.meta.nmr_processing["scp_processing"]``:

.. code-block:: python

    experiment = scp.nmr.Experiment(fid)
    result = experiment.process(apodization="em", lb=5.0)

    trace = result.meta.nmr_processing["scp_processing"]
    requested = trace["requested"]
    applied = trace["applied"]

This trace is attached to the result only. The source dataset is not mutated.

- ``requested`` contains only the arguments explicitly provided to
  ``Experiment.process()``.
- ``applied`` contains only the operations that SpectroChemPy actually
  executed and the values those operations really consumed.
- ``dataset.meta.nmr_processing["vendor_profile"]`` remains descriptive vendor
  metadata and is never applied automatically.
- the trace records the SpectroChemPy processing associated with the current
  result, not a complete historical processing history for the dataset.
- vendor replay is not implemented.
- ``phase="metadata"`` applies the dataset's current ``meta.phc0`` /
  ``meta.phc1`` state through the existing ``pk()`` path. It does not consult
  ``vendor_profile`` and does not replay TopSpin ``PHC0`` / ``PHC1`` from
  ``procs``. On the validated raw TopSpin 1D oracle path it is currently
  numerically identical to ``phase=None``; on already frequency-domain spectra
  carrying phase metadata, it can still apply a real phase correction.

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

The published public examples for this plugin currently focus on validated 1D
workflows.

Limitations
===========

Currently, the user-facing documentation for this plugin focuses on TopSpin
datasets and related NMR workflows.
