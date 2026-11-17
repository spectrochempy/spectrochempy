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

    python -m pip install "spectrochempy[tensor]"

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

The generated public API page for the Tensor plugin is listed in
:doc:`/reference/plugins`. See :class:`spectrochempy.tensor.CP`.

Examples
========

See the :doc:`CP/PARAFAC tutorial </userguide/analysis/cp>` for organizing
a 3D dataset, selecting the rank, and interpreting the factors.

Limitations
===========

Tensor decomposition implementations live under
``spectrochempy_tensor.decompositions``. This page currently focuses on the
main CP/PARAFAC workflow exposed through ``scp.tensor.CP(...)``.
