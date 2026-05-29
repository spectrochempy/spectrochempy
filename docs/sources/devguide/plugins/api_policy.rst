.. _plugin-api-policy:

Plugin API Policy
=================

SpectroChemPy plugins expose public APIs in two complementary places.
Choose the location from the kind of operation, not from the plugin name.

Package/plugin namespaces
-------------------------

Use ``scp.<plugin>.*`` for functions that create objects, read files, or run
global plugin-level workflows.  These functions do not operate on an existing
``NDDataset`` instance as their parent object.

Examples::

    scp.nmr.read_topspin(...)
    scp.iris.IRIS(...)
    scp.iris.batch_iris(...)

The legacy alias ``scp.read_topspin(...)`` is kept as a compatibility layer for
existing code.  When the NMR plugin is installed, it delegates through the
plugin reader registry to ``scp.nmr.read_topspin(...)``.  When the plugin is not
installed, the core stub raises a ``MissingPluginError`` with the install hint
``pip install spectrochempy[nmr]``.

Dataset accessors
-----------------

Use ``dataset.<plugin>.*`` only for operations that genuinely use the parent
``NDDataset`` as input.  The accessor callable receives the dataset as its first
argument.

Examples::

    dataset.iris.kernel_matrix(...)
    # future examples:
    dataset.nmr.phase(...)
    dataset.nmr.apodize(...)

Avoid dataset accessors for I/O and object creation.  In particular, do not add
APIs such as::

    dataset.read_topspin(...)
    dataset.nmr.read_topspin(...)

Current implementation note
---------------------------

The current plugin registry stores namespaced dataset accessors as string keys
such as ``"iris.kernel_matrix"``.  At runtime,
``DatasetPluginAccessor`` exposes this as ``dataset.iris.kernel_matrix(...)``.
This is an incremental mechanism, not a final accessor-class design.

A future cleanup may move mature domains toward real accessor classes, but the
public policy should remain stable:

* I/O, object creation, and global workflows belong at ``scp.<plugin>.*``.
* Operations on an existing dataset belong at ``dataset.<plugin>.*``.
* Legacy aliases are thin compatibility layers, not new primary APIs.

Namespace conventions
---------------------

Official plugin namespaces should be short, stable, and domain-oriented. They
represent the scientific or technical domain exposed to users, for example
``scp.iris`` or ``scp.nmr``. Experimental plugin namespaces such as
``scp.cantera`` follow the same naming convention but are not yet stable.
Avoid creating a second namespace
for the same domain unless there is a clear migration plan.

Documentation and examples should prefer namespace APIs, such as
``scp.iris.IRIS()``, over root-level compatibility
aliases such as ``scp.IRIS``. Compatibility aliases may remain in
tests when they intentionally protect old user code.
