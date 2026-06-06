
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


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Fixed preservation of reference-based coordinates when copying
  ``CoordSet`` and ``NDDataset`` objects.


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


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- TEST: Added focused behavioral tests covering reference preservation through
  ``CoordSet.copy()``, ``copy.deepcopy(CoordSet)``, and ``NDDataset.copy()``.

- TEST: Added CoordSet public-contract edge tests covering duplicate-title
  lookup, title/name collisions, synthetic alias collisions, reference lookup,
  and selected non-first default preservation across copy and slicing.

- MAINT: Removed stale commented ``docrep`` residue and the unused commented
  ``numpydoc`` pre-commit hook block.

- TEST: Added a project-wide source-docstring guard to detect stale
  ``docrep``-style placeholders in ``spectrochempy`` source docstrings.

- MAINT: Added private ``CoordSet`` group-model conversion helpers to prepare
  storage migration without changing runtime storage, lookup, serialization, or
  lifecycle behavior.

- TEST: Added focused round-trip tests covering simple, multi-coordinate,
  selected-default, and reference ``CoordSet`` conversion through the private
  group model.

- MAINT: Added an internal ``CoordSet`` lifecycle helper for reduction-related
  dimension cleanup and migrated ``ndmath._reduce_dims()`` to use it,
  preserving existing behavior without changing public APIs, storage, or
  serialization.
- MAINT: Reused the internal ``CoordSet`` reduction lifecycle helper for extrema
  reduction dimension cleanup, preserving existing behavior without changing
  public APIs, storage, or serialization.
- MAINT: Added an internal ``CoordSet`` lifecycle helper for concatenate-related
  coordinate propagation and migrated ``concatenate()`` to use it, preserving
  existing behavior without changing public APIs, storage, or serialization.

- MAINT: Added an internal ``CoordSet`` lifecycle helper for reshape-related
  coordinate handling and migrated ``NDDataset.reshape()`` to use it,
  preserving existing behavior without changing public APIs, storage, or
  serialization.

- MAINT: Added an internal ``CoordSet`` lifecycle API for dimension dropping
  and migrated ``NDDataset.squeeze()`` to use it, preserving existing behavior
  without changing public APIs, storage, or serialization.

- MAINT: Added an internal ``CoordSet`` lifecycle API for coordinate
  assignment/replacement and migrated a focused ``NDDataset`` coordinate
  assignment path to use it, preserving existing behavior while reducing
  direct CoordSet internal coupling.

- MAINT: Introduced an initial internal ``CoordSet`` lifecycle slicing API and
  migrated ``NDDataset`` slicing to use it, preserving existing behavior
  without changing public APIs, storage, or serialization.

- MAINT: Added an internal ``CoordSet`` lifecycle helper for interpolation
  coordinate reconstruction and migrated ``interpolate()`` to use it,
  preserving existing behavior without changing public APIs, storage, or
  serialization.

- TEST: Added public-contract behavioral tests for ``CoordSet`` multi-coord
  construction, attribute access, iteration, keepnames, and additional
  properties (``name``, ``coords``).

- TEST: Added public-contract behavioral tests for ``NDDataset`` ``coord()``,
  ``set_coordset()``, dim attribute access, multi-coord dimensions, and
  ``None``/size-only coordinate entries.

- MAINT: Extracted private resolver helpers ``_resolve_string_lookup``,
  ``_resolve_numeric_lookup``, and ``_resolve_get`` from
  ``CoordSet.__getitem__`` to centralize read-lookup logic before storage
  redesign, preserving existing behavior without changing public APIs,
  storage, or serialization.

- MAINT: Extracted private resolver helpers ``_resolve_set`` and
  ``_resolve_delete`` from ``CoordSet.__setitem__`` and
  ``CoordSet.__delitem__`` to centralize write/delete lookup logic before
  storage redesign, preserving existing behavior without changing public
  APIs, storage, or serialization.
