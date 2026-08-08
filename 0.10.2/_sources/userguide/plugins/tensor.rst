.. _tensor-plugin:

=============
Tensor plugin
=============

The ``spectrochempy-tensor`` plugin provides tensor decomposition classes backed
by TensorLy. Keeping these algorithms in a plugin lets the core package remain
independent from tensor-specific dependencies while preserving integrated access
when the plugin is installed.

Install
=======

.. code-block:: bash

    pip install spectrochempy[tensor]

or directly:

.. code-block:: bash

    pip install spectrochempy-tensor

Use
===

.. code-block:: python

    import spectrochempy as scp

    model = scp.tensor.CP(n_components=2)
    model.fit(dataset)
    factors = model.result.factors
    weights = model.result.weights

The historical ``scp.CP`` alias is kept as a deprecated compatibility path.
New code should use ``scp.tensor.CP``.
Direct accessors such as ``model.A``, ``model.B``, ``model.C``, and
``model.weights`` remain available.

Extensibility
=============

Tensor decomposition implementations live under
``spectrochempy_tensor.decompositions``. Shared bridges between TensorLy objects
and SpectroChemPy datasets should live under ``spectrochempy_tensor.adapters``
so future classes such as Tucker or TensorTrain can reuse them without adding
tensor concepts back to the core package.
