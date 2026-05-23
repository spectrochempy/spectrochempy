
:orphan:

What's New in Revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.

.. section

New Features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)
- NDMath internal refactor: extracted focused private helpers (``_prepare_operation_quantities``,
  ``_check_coordinate_compatibility``, ``_resolve_operation_units``, ``_execute_operation``)
  and ``_ExecutionPlan`` class from ``NDMath._op()``, reducing it from ~300 to ~70 lines.
- ``__array_ufunc__`` now explicitly rejects unsupported ufunc methods (``reduce``,
  ``accumulate``, ``outer``, ``at``) by returning ``NotImplemented``.
- ``numpy-quaternion`` is now an optional dependency, imported through
  ``spectrochempy.utils.quaternion`` with a graceful fallback when not installed.
- added NDDataset.reshape()

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)
- refactor of PSD

.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
