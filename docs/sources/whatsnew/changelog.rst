
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

- ``interpolate`` now preserves the source coordinate's units and title when the
  target is given as a bare array (e.g. ``np.linspace(...)``) (#1094).  The
  generated coordinate previously lost its units (became unitless) and title
  (became ``<untitled>``); the array values are assumed to be expressed in the
  source coordinate's units, so the units are attached rather than converted.

- ``interpolate`` now returns the result in the order of the requested target
  coordinate.  Interpolating a dataset stored with decreasing coordinates (e.g.
  wavenumbers from 4000 to 400 cm⁻¹) onto an increasing target previously
  flipped the output back to decreasing, ignoring the requested ordering
  (#1100).  The mask and any secondary coordinates are now reordered together
  with the data before interpolation, so they stay aligned when the source
  coordinate is decreasing.

- Fixed ``Project.__str__()`` tree formatting when a project contains both
  sub-projects and sibling datasets or scripts at the same level.  The
  recursive ``_listproj`` helper previously used ``s.strip("\\n")`` which
  stripped the trailing newline from the entire accumulated string, causing
  sibling entries to appear on the same line as the last child of the
  preceding sub-project.

- ``concatenate`` now converts coordinate values expressed in compatible but
  different units to the units of the first dataset instead of silently
  concatenating raw magnitudes, and raises a ``UnitsCompatibilityError`` when
  the coordinate units of the concatenation dimension are incompatible
  (#1101).

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

- Fixed ``CoordSet`` empty-state invariants: ``CoordSet()`` now correctly
  creates a valid empty coordinate set instead of raising ``TypeError``.
  Properties ``default``, ``default_index``, ``data`` return ``None`` and
  ``titles``, ``labels``, ``units``, ``sizes`` return ``[]`` when the
  coordinate set is empty, instead of raising ``IndexError`` or
  ``TypeError``.  The internal ``_coords`` list is never set to ``None``,
  eliminating fragile None-guards in read-side properties.

- Stabilized 1D CSV round-trip support: ``read_csv`` now tolerates header rows
  (e.g., column titles) written by ``write_csv``, and correctly handles both
  single-column (data-only) and multi-column (coordinate + data) CSV files.
  Synthetic tests for CSV reading/writing have been added, removing the dependency
  on external test data for these functionalities (#1077).

- Preserved scientific-context metadata (``meta``, ``author``,
  ``description``, ``origin``, and ``filename``) in wrapper-based processing
  and analysis outputs such as ``Filter(...).transform(...)`` (#1103).


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- pint >= 0.24 is now required

.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- Mixed arithmetic between ``NDDataset`` and ``Coord`` is now rejected
  (e.g. ``dataset + coord`` or ``coord * dataset``).  ``Coord`` is treated as
  axis support, not as a signal-bearing operand.  Workflows needing correction
  vectors, weighting profiles, response curves, or other signal-like 1D
  operands should represent them as 1D ``NDDataset`` objects instead.  This
  clarifies the math semantics under the broader ``#1103`` arithmetic and
  metadata characterization work.

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

- MAINT: Integrated ``Project`` into the common display/representation model:
  added a custom compact ``__repr__`` (``Project: <name>``), a ``_cstr()``
  detailed representation (name, author, description, hierarchy), and migrated
  ``_repr_html_()`` from ad-hoc string replacement to the shared
  ``convert_to_html()`` pipeline.  ``pstr(project)`` now returns the detailed
  ``_cstr()`` output.  Observable behavior is preserved for Coord, CoordSet,
  and NDDataset.  This is PR3 of the Display / Representation Architecture
  (#843).

- MAINT: Harmonized notebook HTML headings — replaced the ad-hoc
  ``obj.__str__() + [obj.name]`` summary line in ``convert_to_html()`` with
  a dedicated ``_html_heading()`` helper.  Headings now follow the pattern
  ``TypeName [name]`` (user-provided name) or just ``TypeName``
  (auto-generated ID), eliminating duplicated project names and exposing no
  internal UUIDs.  ``_repr_html_()`` headings are now decoupled from
  terminal ``__str__()``, aligning with the RFC decision that notebook
  headings may differ from terminal display.  This is PR4 of the Display /
  Representation Architecture (#843).

- MAINT: Added ``has_defined_name`` to ``Project`` — a boolean trait signal
  distinguishing user-provided names from auto-generated fallbacks.  The name
  setter now sets ``_explicit_name = True`` only when ``name is not None``;
  the auto-generated fallback assigns directly to ``_name`` (no recursive
  call) to preserve the distinction.  ``_html_heading()`` now handles Project
  via the ``has_defined_name`` property, so auto-generated names no longer
  appear in notebook headings.  This is PR4.5 of the Display / Representation
  Architecture (#843).

- MAINT: Preserved Project hierarchy in HTML display — ``_process_section()``
  now wraps indented hierarchy lines (prefixed by ``⤷``) in ``<div>`` elements,
  preventing HTML whitespace collapsing from flattening nested sub-projects and
  datasets into a single line.  This is PR6 of the Display / Representation
  Architecture (#843).

- MAINT: Moved CP/PARAFAC implementation and TensorLy dependency ownership into
  the new tensor plugin, keeping the core package tensor-agnostic.

- MAINT: Internal ``CoordSet`` storage redesign — all mutation paths (set,
  delete, append, lifecycle) now resolve through the group-backed
  projection-resolution-reconstruction pipeline instead of legacy in-place
  ``_coords`` mutation.  The migration consolidated lookup, serializer
  adapters, group conversion, and lifecycle helpers around transient group
  metadata while preserving runtime storage, serialization, and public
  behavior.  Same-dimension mutations apply group state directly to legacy
  storage to avoid double-wrapping in ``_groups_to_coordset``, which fixed a
  pre-existing corruption bug in same-dimension title set.  Alias invariants,
  ``default_id`` semantics, label metadata, reference pass-through, and
  coordinate metadata are preserved throughout.

- MAINT: Switched ``CoordSet`` internal storage from the ``_coords``
  ``traitlets.List`` trait (with its ``@validate`` hook) to a plain Python
  ``_storage`` list, completing the migration away from trait-based
  coercion.  The ``_coords`` trait and ``_coords_validate`` method have been
  removed; ``_finalize_child_coordset`` now handles nested ``CoordSet``
  setup explicitly in ``_append`` and mutation paths.  Sorting, copying,
  and name-validation logic that was previously in the validator are now
  performed in ``__init__``, ``_append``, and ``set``.  The internal
  observer ``_coords_update`` was renamed to ``_child_name_changed``.

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
