.. _perkinelmer-plugin:

==================
PerkinElmer plugin
==================

Introduction
============

The ``spectrochempy-perkinelmer`` plugin provides a reader for PerkinElmer
``.sp`` binary IR files.

Installation
============

Install it with:

.. code-block:: bash

    pip install spectrochempy[perkinelmer]

Recommended API
===============

Use the recommended namespaced API:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.perkinelmer.read("path/to/file.sp")

Compatibility aliases
=====================

Compatibility aliases are also available for older scripts:

.. code-block:: python

    dataset = scp.perkinelmer.read_perkinelmer("path/to/file.sp")
    dataset = scp.read_perkinelmer("path/to/file.sp")
    dataset = scp.read_sp("path/to/file.sp")

API Reference
=============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.perkinelmer.read

Examples
========

Use the same namespaced reader shown above for ordinary PerkinElmer ``.sp``
import workflows.

Limitations
===========

- Only single-spectrum ``.sp`` files are supported.
- The ``.prf`` format is not supported.
- Metadata extraction depends on the presence of standard PerkinElmer blocks;
  files with incomplete metadata will still load but with reduced ``meta``
  information.
