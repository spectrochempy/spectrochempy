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

The plugin-owned TopSpin reader is documented here rather than in the core API
reference because it is provided by ``spectrochempy-nmr``, not by the core
package itself. The recommended public entry point is
``scp.nmr.read_topspin(...)``. The legacy compatibility alias
``scp.read_topspin(...)`` is still available when the plugin is installed, but
new documentation and examples should prefer the namespaced API.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spectrochempy.read_topspin

For phase-sensitive 2D NMR workflows, install hypercomplex support as well:

.. code-block:: bash

    pip install spectrochempy[nmr,hypercomplex]

The NMR plugin owns Bruker/TopSpin conventions such as experiment directory
resolution, processed-data defaults, acquisition metadata, and NMR unit
contexts. Core SpectroChemPy remains responsible for generic datasets, units,
plotting, and ordinary FFT operations.
