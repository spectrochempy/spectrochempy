.. _tensor-plugin:

=============
Tensor plugin
=============

Introduction
============

The ``spectrochempy-tensor`` plugin provides tensor decomposition classes backed
by TensorLy. Keeping these algorithms in a plugin lets the core package remain
independent from tensor-specific dependencies while preserving integrated access
when the plugin is installed.

Installation
============

.. code-block:: bash

    pip install spectrochempy[tensor]

or directly:

.. code-block:: bash

    pip install spectrochempy-tensor

Recommended API
===============

.. code-block:: python

    import spectrochempy as scp

    model = scp.tensor.CP(n_components=2)
    model.fit(dataset)
    factors = model.result.factors
    weights = model.result.weights

Compatibility aliases
=====================

The historical ``scp.CP`` alias is kept as a deprecated compatibility path, but
new documentation and examples should use ``scp.tensor.CP``.

Direct accessors such as ``model.A``, ``model.B``, ``model.C``, and
``model.weights`` remain available.

API Reference
=============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.tensor.CP

Examples
========

Use the package-level ``scp.tensor.CP(...)`` estimator shown above for
TensorLy-backed CP/PARAFAC workflows on existing datasets.

Limitations
===========

Tensor decomposition implementations live under
``spectrochempy_tensor.decompositions``. This page currently focuses on the
main CP/PARAFAC workflow exposed through ``scp.tensor.CP(...)``.
