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

    series = scp.carroucell.read("path/to/carroucell-directory")

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

See :ref:`Carroucell experiment directories <import-carroucell-ir>` in the IR import guide.

Limitations
===========

The reader can return a single dataset or a collection of datasets. See the
IR import guide for the directory layout and the temperature-log constraints.
