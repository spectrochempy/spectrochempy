.. _nmr-plugin:

NMR plugin
==========

.. _introduction:

The independently released ``spectrochempy-nmr`` plugin reads vendor
and exchange formats and processes 1D and supported 2D acquisitions.
SpectroChemPy provides the dataset, units, plotting and generic processing API.

.. _installation:

Install or upgrade the plugin explicitly:

.. code-block:: bash

    python -m pip install "spectrochempy[nmr]>=1.2.1"

The plugin's
`installation and compatibility guide
<https://www.spectrochempy.fr/projects/nmr/latest/getting_started/installation.html>`_
gives the requirements for each workflow.

.. _recommended-api:

.. _compatibility-aliases:

Read data through the namespaced API:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read("path/to/fid")

Use the standalone plugin documentation for the
`reader guide <https://www.spectrochempy.fr/projects/nmr/latest/userguide/reading_data.html>`_,
`1D processing <https://www.spectrochempy.fr/projects/nmr/latest/userguide/processing_1d.html>`_,
and `2D processing and TopSpin replay
<https://www.spectrochempy.fr/projects/nmr/latest/userguide/processing_2d.html>`_.

.. _topspin-processing-profile:
.. _spectrochempy-processing-trace:

Recorded TopSpin recipes are applied when explicitly requested with
``Experiment.process(replay="topspin")`` on supported data.
See the plugin's
`TopSpin processing guide <https://www.spectrochempy.fr/projects/nmr/latest/userguide/processing_2d.html>`_
for supported workflows.

.. _api-reference:

See the standalone
`NMR API reference
<https://www.spectrochempy.fr/projects/nmr/latest/reference/nmr_api.html>`_.


.. _examples:

The plugin's `example gallery
<https://www.spectrochempy.fr/projects/nmr/latest/examples_overview.html>`_
shows how to read, process, and analyze NMR data.

.. _limitations:

See `compatibility and scope
<https://www.spectrochempy.fr/projects/nmr/latest/compatibility.html>`_
for supported workflows and their limits.
