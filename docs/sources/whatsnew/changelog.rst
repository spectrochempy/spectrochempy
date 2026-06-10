
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

- Added the official ``spectrochempy-tensor`` plugin for TensorLy-backed tensor
  decompositions, exposing CP/PARAFAC as ``scp.tensor.CP``.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``read_opus`` now correctly reads assembled / time-resolved OPUS files
  containing data series blocks such as ``a``, ``sm``, ``igsm``, ``phsm``,
  ``tr``, and exposes the new ``TRACE``, ``GCIG``, ``GCSC`` type selectors
  (#1035).

- ``read_opus`` no longer fails when an OPUS file stores a malformed
  acquisition sub-second field (e.g. ``10:31:19.-70``); the timestamp now
  falls back to whole-second precision instead of returning ``None`` (#1036).

- Fixed parsing of the ``a.u.`` (absorbance) and ``K.M.`` (Kubelka-Munk) unit
  symbols from strings, which previously failed because the dots were read as a
  multiplication.

- Fixed native round-trip preservation of selected non-first default
  coordinates in same-dimension multi-coordinate datasets.

- Fixed restoration of reference-based coordinates after native
  save/load round-trips.

- Fixed preservation of reference-based coordinates when copying
  ``CoordSet`` and ``NDDataset`` objects.

- Fixed a crash when concatenating datasets along a dimension where one
  dataset has labels and another does not.  Previously, a ``TypeError``
  was raised on mixed labeled/unlabeled coordinates.

- Fixed structural corruption of same-dimension ``CoordSet`` when setting
  coordinates by name, numeric index, or title.  The group-backed resolution
  path left stale child aliases on the dimension group after replacing
  multiple entries with a single incoming ``CoordSet`` (e.g.
  ``cs["x"] = CoordSet(...)``, ``cs[0] = CoordSet(...)``,
  ``cs["wavenumber"] = CoordSet(...)``), causing ``_groups_to_coordset``
  to double-wrap the inner coordinates under an extra ``CoordSet`` layer.
  This affected concatenation of multi-coordinate datasets.  Also fixed the
  related case of setting a child coordinate by synthetic alias
  (e.g. ``cs["x_2"] = coord``).


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- pint >= 0.24 is now required

.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- ``scp.CP`` and ``spectrochempy.analysis.decomposition.cp.CP`` are now
  deprecated compatibility paths for ``scp.tensor.CP``.


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- MAINT: Moved CP/PARAFAC implementation and TensorLy dependency ownership into
  the new tensor plugin, keeping the core package tensor-agnostic.

- MAINT: Advanced the internal ``CoordSet`` storage migration by consolidating
  lookup, serializer adapters, group conversion, and lifecycle helpers around
  transient group metadata while preserving legacy runtime storage,
  serialization, and public behavior.  Migrated ``_concatenate_dim`` and
  ``_interpolate_dim`` to the lifecycle adapter pattern, completing the
  migration of all dimension manipulation methods.  Migrated ``_resolve_delete``
  to the group-backed architecture following the same projection-resolution-
  reconstruction pattern, covering top-level name and title deletion, synthetic
  alias delegation, and same-dimension fallthrough.  Same-dimension
  multi-coordinate semantics, label metadata, alias and default preservation,
  reference pass-through, and coordinate metadata propagation are maintained.

- MAINT: Removed stale commented ``docrep`` residue and the unused commented
  ``numpydoc`` pre-commit hook block.

- TEST: Added a project-wide source-docstring guard to detect stale
  ``docrep``-style placeholders in ``spectrochempy`` source docstrings.

- MAINT: Harmonized plugin release workflows and maintainer documentation:
  the ``release_plugin.yml`` workflow now gracefully handles first plugin
  releases (where the version is already committed) by skipping the commit
  and push steps instead of failing, and automatically unsets the GitHub
  "Latest" flag on plugin releases so the core release remains the primary
  release on the repository front page.  The maintainer documentation now
  describes the role of ``plugin_version_status.py``, the first-release
  workflow, and the "Latest" flag policy (#1082).
