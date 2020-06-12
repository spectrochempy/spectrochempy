.. _api_reference_spectrochempy:

User API reference
==================

The |scpy| API exposes many objects and functions that are described below.

To use the API, one must load it using one of the following syntax :

>>> import spectrochempy as scp

>>> from spectrochempy import *

In the second syntax, as usual in python, access to the objects/functions
may be simplified (*e.g.*, we can use `plot_stack` instead of  `scp.plot_stack` but there is always a risk of
overwriting some variables already in the namespace. Therefore, the first syntax is in general
recommended,
although that, for the examples in this documentation, we have often use the
second one for simplicity.

**Contents**

.. toctree::
    :maxdepth: 1

    NDDataset object <generated/spectrochempy.NDDataset>
    io
    processing
    analysis
    fitting


