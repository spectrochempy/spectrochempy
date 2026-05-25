.. _iris-plugin:

===========
IRIS plugin
===========

The ``spectrochempy-iris`` plugin provides 2D-IRIS analysis tools for
spectroscopic adsorption and diffusion studies.

Install it with:

.. code-block:: bash

    pip install spectrochempy[iris]

Use package-level classes from ``scp.iris`` and dataset-bound helpers from
``dataset.iris``:

.. code-block:: python

    import spectrochempy as scp

    analysis = scp.iris.IRIS()
    kernel = dataset.iris.kernel_matrix(kernel_type="langmuir")

New code should prefer the namespaced API. Compatibility aliases may be
available for older scripts during the transition.
