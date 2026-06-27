.. _iris-plugin:

===========
IRIS plugin
===========

Introduction
============

The ``spectrochempy-iris`` plugin provides 2D-IRIS analysis tools for
spectroscopic adsorption and diffusion studies.

Installation
============

Install it with:

.. code-block:: bash

    pip install spectrochempy[iris]

Recommended API
===============

Use package-level classes from ``scp.iris`` and dataset-bound helpers from
``dataset.iris``:

.. code-block:: python

    import spectrochempy as scp

    analysis = scp.iris.IRIS()
    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")
    analysis.fit(dataset, kernel)

    distribution = analysis.result.f
    residual_sum_squares = analysis.result.RSS

Compatibility aliases
=====================

The compatibility alias ``scp.IRIS`` remains available for older scripts, but
new documentation and examples should prefer ``scp.iris.IRIS``.

The existing direct accessors, such as ``analysis.f``, ``analysis.K``,
``analysis.RSS``, and ``analysis.SM``, remain supported.

API Reference
=============

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.iris.IRIS

Examples
========

Use dataset-bound helpers such as ``dataset.iris.kernel_matrix(...)`` together
with the package-level ``scp.iris.IRIS()`` estimator shown above.

Limitations
===========

Currently, this user-facing page focuses on the main IRIS estimator and the
most common dataset-bound IRIS helper workflow.
