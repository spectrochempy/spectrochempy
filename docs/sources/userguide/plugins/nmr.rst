.. _nmr-plugin:

==========
NMR plugin
==========

The ``spectrochempy-nmr`` plugin provides NMR-specific readers and processing
workflows, including the Bruker TopSpin reader.

Install it with:

.. code-block:: bash

    pip install spectrochempy[nmr]

Use the NMR namespace:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/fid")

For phase-sensitive 2D NMR workflows, install hypercomplex support as well:

.. code-block:: bash

    pip install spectrochempy[nmr,hypercomplex]

The NMR plugin owns Bruker/TopSpin conventions such as experiment directory
resolution, processed-data defaults, acquisition metadata, and NMR unit
contexts. Core SpectroChemPy remains responsible for generic datasets, units,
plotting, and ordinary FFT operations.
