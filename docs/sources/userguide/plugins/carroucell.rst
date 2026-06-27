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

    dataset = scp.carroucell.read_carroucell("path/to/carroucell-directory")

Compatibility aliases
=====================

The short namespaced alias ``scp.carroucell.read(...)`` is also available, but
the explicit ``scp.carroucell.read_carroucell(...)`` form is clearer in
documentation and examples.

API Reference
=============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.carroucell.read_carroucell

Examples
========

Use the same namespaced reader shown above when importing a carroucell
experiment directory.

Limitations
===========

Currently, the user-facing documentation for this plugin focuses on directory
imports for carroucell experiment workflows.
