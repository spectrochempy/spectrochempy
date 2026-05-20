.. _plugin-dependent-examples:

==========================
Plugin-dependent Examples
==========================

SpectroChemPy keeps one central gallery, organized by scientific topic rather
than by package internals. Plugin-dependent examples remain visible in their
natural sections, such as decomposition, simulation, import/export, or NMR
processing.

When an example needs an official plugin, it should say so near the top of the
example and use the recommended namespaced API:

.. code-block:: python

    from spectrochempy.iris import IRIS
    import spectrochempy as scp

    dataset = scp.nmr.read_topspin("path/to/fid")
    reactor = scp.cantera.PFR

Plugin-dependent examples should use short, consistent notes such as:

.. code-block:: text

    Requires the official spectrochempy-nmr plugin.
    Install with: pip install spectrochempy[nmr]

The current list of plugin-dependent examples is maintained in
:ref:`plugin-examples-list`.

This convention keeps the beginner path uncluttered while still making
specialized workflows easy to find.
