:orphan:

What's New in Revision 0.8.5.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.8.5.dev.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

New Features
~~~~~~~~~~~~
- NDMath internal refactor: extracted focused private helpers (``_prepare_operation_quantities``,
  ``_check_coordinate_compatibility``, ``_resolve_operation_units``, ``_execute_operation``)
  and ``_ExecutionPlan`` class from ``NDMath._op()``, reducing it from ~300 to ~70 lines.
- ``__array_ufunc__`` now explicitly rejects unsupported ufunc methods (``reduce``,
  ``accumulate``, ``outer``, ``at``) by returning ``NotImplemented``.
- ``numpy-quaternion`` is now an optional dependency, imported through
  ``spectrochempy.utils.quaternion`` with a graceful fallback when not installed.
- added NDDataset.reshape()

Bug Fixes
~~~~~~~~~
- refactor of PSD
