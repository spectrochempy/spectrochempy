.. _plugin-dependent-examples:

==========================
Plugin-dependent Examples
==========================

SpectroChemPy keeps one central gallery, organized by scientific topic rather
than by package internals. Plugin-dependent examples are maintained by the
plugin that provides the feature, then staged into their natural gallery
sections, such as decomposition, simulation, import/export, or NMR processing.

When an example needs a plugin, it should say so near the top of the
example and use the recommended namespaced API:

.. code-block:: python

    import spectrochempy as scp

    analysis = scp.iris.IRIS()
    dataset = scp.nmr.read_topspin("path/to/fid")

Plugin-dependent examples should use short, consistent notes such as:

.. code-block:: text

    Requires the spectrochempy-nmr plugin.
    Install with: pip install spectrochempy[nmr]

Official plugins declare their gallery examples in an ``examples/gallery.toml``
manifest. The current list of plugin-dependent examples is generated from
these manifests in :ref:`plugin-examples-list`.

This convention keeps the beginner path uncluttered while still making
specialized workflows easy to find.
