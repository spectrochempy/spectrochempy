.. _nmr-plugin:

NMR plugin
==========

The independently released ``spectrochempy-nmr`` plugin reads vendor
and exchange formats and processes 1D and qualified 2D acquisitions.
SpectroChemPy provides the dataset, units, plotting and generic processing API.

Install or upgrade the plugin explicitly:

.. code-block:: bash

    python -m pip install --upgrade "spectrochempy-nmr>=0.3,<0.4"

This command also works with SpectroChemPy 1.2.0, whose ``nmr`` extra
selects plugin versions below 0.3. The plugin's
`installation and compatibility guide
<https://spectrochempy-nmr.readthedocs.io/latest/getting_started/installation.html>`_
gives the requirements for each workflow.

Read data through the namespaced API:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read("path/to/fid")

``scp.read_topspin()`` remains available for explicit Bruker reads.
Use the standalone plugin documentation for the
`reader guide <https://spectrochempy-nmr.readthedocs.io/latest/userguide/reading_data.html>`_,
`1D processing <https://spectrochempy-nmr.readthedocs.io/latest/userguide/processing_1d.html>`_,
and `2D processing and TopSpin replay
<https://spectrochempy-nmr.readthedocs.io/latest/userguide/processing_2d.html>`_.
Recorded TopSpin recipes are applied when explicitly requested with
``Experiment.process(replay="topspin")`` on supported data.

See the :doc:`core plugin API index </reference/plugins>` and the standalone
`NMR API reference
<https://spectrochempy-nmr.readthedocs.io/latest/reference/nmr_api.html>`_.
