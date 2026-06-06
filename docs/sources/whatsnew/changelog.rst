
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
