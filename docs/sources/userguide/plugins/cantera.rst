.. _cantera-plugin:

==============
Cantera plugin
==============

The ``spectrochempy-cantera`` plugin provides simulation workflows based on
Cantera, currently centered on plug-flow reactor support.

Install it with:

.. code-block:: bash

    pip install spectrochempy[cantera]

Use the Cantera namespace:

.. code-block:: python

    import spectrochempy as scp

    PFR = scp.cantera.PFR

The plugin keeps Cantera and its optional dependencies outside the core
installation until simulation workflows are needed.
