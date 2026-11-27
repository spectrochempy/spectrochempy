.. _carroucell-plugin:

==================
Carroucell plugin
==================

Introduction
============

The ``spectrochempy-carroucell`` plugin provides a reader for carroucell
experiment directories.

Installation
============

Install it directly:

.. code-block:: bash

    pip install spectrochempy-carroucell

Recommended API
===============

Use the recommended namespaced API:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.carroucell.read("path/to/carroucell-directory")

Grouping and temperature labels
===============================

By default, scans are grouped by the sample-holder prefix in their filenames and sorted by
acquisition time within each compatible group. If one holder contains
incompatible retained spectral grids or recognized signal quantities, several
groups are returned with the same holder name. A warning explains the
incompatibility. With ``merge=True``, further assembly across holders is attempted,
but incompatible grids and quantities still remain separate. The reader does
not resample spectra or implicitly convert between different signal quantities.

Compatible scans retain their combined return, including the existing assembly
of compatible single-scan holders. Depending on the groups, the result is one
``NDDataset`` or a list-like ``ScpObjectList``.

When one temperature ``.xls`` file is present, its readings are interpolated at
each retained observation time, accounting for ``delta_clocks``. Temperature
selection covers the earliest through latest imported observation across all
groups. Each group receives one temperature label per spectrum, regardless of
its size; its existing time values, units and source labels are preserved.
The existing linear interpolation and extrapolation behavior is unchanged.

Compatibility aliases
=====================

The explicit historical alias ``scp.carroucell.read_carroucell(...)`` remains
available, but new documentation and examples should prefer the shorter
``scp.carroucell.read(...)`` form.

API Reference
=============

The generated public API page for the Carroucell plugin is listed in
:doc:`/reference/plugins`. See :func:`spectrochempy.carroucell.read`.

Examples
========

Use the same namespaced reader shown above when importing a carroucell
experiment directory.

Limitations
===========

Currently, the user-facing documentation for this plugin focuses on directory
imports for carroucell experiment workflows.
