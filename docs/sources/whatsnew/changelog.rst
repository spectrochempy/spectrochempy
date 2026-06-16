
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

- Restored historical hypercomplex/quaternion dataset display (#1147).  Detailed
  terminal and HTML representations once again show explicit ``RR``/``RI``/``IR``/``II``
  component blocks and preserve complex-dimension shape annotations, instead of
  falling back to raw quaternion scalar dumps.

- JCAMP-DX I/O is more robust (#1080, #1132, #1150).  ``read_jcamp`` now
  handles ``##YUNITS=TRANSMITTANCE`` as the ``transmittance`` unit, accepts
  header values containing ``=``, reports invalid axis metadata with a clear
  formatted error, and keeps the deprecated ``read_jdx`` alias pointed at
  ``read_jcamp``.  ``write_jcamp`` now exports masked samples as JCAMP-DX
  missing values (``?``), excludes them from ``##MAXY``/``##MINY``, and
  preserves masking on round-trip instead of leaking stale values.

- ``interpolate`` now preserves coordinate metadata and target semantics
  (#1093, #1094, #1098, #1100).  Bare-array targets keep the source
  coordinate's units and title; PCHIP interpolation honours ``fill_value``;
  output follows the requested target order even when the source coordinate is
  decreasing; masks and secondary coordinates stay aligned; and labels are
  carried to target points that exactly match original coordinate values while
  genuinely resampled points remain unlabelled.

- ``write_csv`` now exports masked samples as missing values (``NaN``) instead
  of writing their underlying data (#1135).  The writer previously iterated over
  ``dataset.data`` directly, leaking the stored values of points the user had
  explicitly masked; masked samples are now filled with ``NaN`` (mirroring the
  ``write_jcamp`` fix), so they round-trip back as ``NaN`` through ``read_csv``
  and unmasked datasets are unaffected.

- Fixed ``Project.__str__()`` tree formatting when a project contains both
  sub-projects and sibling datasets or scripts at the same level.  The
  recursive ``_listproj`` helper previously used ``s.strip("\\n")`` which
  stripped the trailing newline from the entire accumulated string, causing
  sibling entries to appear on the same line as the last child of the
  preceding sub-project.

- ``concatenate`` now handles coordinate metadata more consistently (#1101).
  Coordinate values expressed in compatible but different units are converted
  to the units of the first dataset, incompatible coordinate units raise a
  ``UnitsCompatibilityError``, and mixed labeled/unlabeled coordinates no
  longer crash during concatenation.

- ``read_opus`` now supports more assembled and time-resolved OPUS files
  (#1035, #1036): data series blocks such as ``a``, ``sm``, ``igsm``,
  ``phsm``, and ``tr`` are read, the ``TRACE``, ``GCIG``, and ``GCSC`` type
  selectors are exposed, and malformed acquisition sub-second fields fall back
  to whole-second precision instead of returning ``None``.

- Fixed parsing of the ``a.u.`` (absorbance) and ``K.M.`` (Kubelka-Munk) unit
  symbols from strings, which previously failed because the dots were read as a
  multiplication.

- ``CoordSet`` and same-dimension coordinate handling are more stable.  Native
  save/load now preserves selected non-first default coordinates and restores
  reference-based coordinates, while copying ``CoordSet`` and ``NDDataset``
  objects keeps reference-based coordinates intact.  Same-dimension
  ``CoordSet`` replacement by name, numeric index, title, or synthetic child
  alias no longer double-wraps inner coordinates, which also improves
  concatenation of multi-coordinate datasets.  Empty ``CoordSet`` objects now
  have consistent empty-state properties instead of raising ``TypeError`` or
  ``IndexError``.

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

- Removed the orphaned ``NDDataset.modeldata`` attribute (#1168).  Fit/model
  outputs should be stored and plotted as explicit ``NDDataset`` objects or
  dedicated fit-result objects rather than hidden structural state on
  ``NDDataset``.  ``plot(plot_model=True)`` now emits a ``FutureWarning``
  explaining the removal.

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

- MAINT: Continued the Display / Representation Architecture work (#843).
  Added the semantic ``DisplayItem`` and ``DisplaySection`` representation
  layer, introduced ``Coord._repr_sections()``, shared array value formatting
  across ``NDArray``, ``NDComplexArray``, and ``Coord``, integrated
  ``Project`` into the common terminal and HTML display model, and harmonized
  notebook headings through ``_html_heading()``.  HTML headings now use stable
  type-specific summaries without exposing internal UUIDs, project hierarchies
  render through the shared ``convert_to_html()`` path, and summary metadata is
  displayed inline under the main heading.

- MAINT: Extended the semantic HTML migration to ``NDDataset`` (#843).
  ``NDDataset._repr_sections()`` builds summary, data, and dimension
  ``DisplaySection`` objects, reusing ``CoordSet._repr_sections()``
  for coordinate dimensions.  ``NDDataset._repr_html_()`` now uses the
  semantic path instead of the sentinel-based ``convert_to_html()``,
  producing clean inline summary metadata, collapsible data and dimension
  sections, and removing exposure of internal UUIDs from the heading.

- MAINT: Extended the semantic HTML migration to ``Project`` (#843).
  ``Project._repr_sections()`` builds summary (name, author, description)
  and data (hierarchy tree) ``DisplaySection`` objects.  ``Project._repr_html_()``
  now uses the semantic path instead of ``convert_to_html()``, producing
  clean inline metadata and ``&nbsp;``-indented collapsible hierarchy
  sections.  The sentinel ``_cstr()`` method is preserved unchanged for
  terminal output.

- MAINT: Extended the semantic HTML migration to ``CoordSet`` (#843).
  Added ``CoordSet._repr_sections()`` which builds one ``DisplaySection``
  per dimension, reusing child ``Coord._repr_sections()`` items for simple
  coordinates and flattening same-dimension multi-coordinate content with
  subgroup separators.  ``CoordSet._repr_html_()`` now uses the semantic
  path (``_repr_sections`` + ``_render_sections``) instead of the
  sentinel-based ``convert_to_html()``, producing cleaner HTML without
  inline sentinel markers.  Same-dimension ``CoordSet`` sections now show
  ``Coord`` headings (e.g. ``Coord \`_1\```) instead of ``Dimension``,
  since synthetic child names like ``_1`` / ``_2`` are coordinates of a
  shared dimension, not dimensions themselves.  The docs cache key was
  updated to invalidate the sphinx-gallery cache when display source
  files change.

- TEST: Added synthetic, offline tests for multi-variable Matlab (``.mat``)
  import and documented the behavior in the ``read_matlab`` docstring: numeric
  variables are converted to ``NDDataset`` objects and then grouped by the
  importer when shapes are compatible (same-shape arrays are stacked into one
  dataset, incompatible ones returned separately), while non-numeric and
  Matlab-internal (``__header__``, ``__version__``, ``__globals__``) variables
  are skipped (#1142).

- MAINT: Moved CP/PARAFAC implementation and TensorLy dependency ownership into
  the new tensor plugin, keeping the core package tensor-agnostic.

- MAINT: Completed the internal ``CoordSet`` storage migration.  Mutation
  paths now resolve through the group-backed
  projection-resolution-reconstruction pipeline, lookup and serializer adapters
  share transient group metadata, and runtime storage uses a plain
  ``_storage`` list instead of the legacy trait-based ``_coords`` validator.
  Nested ``CoordSet`` setup, sorting, copying, and name validation are handled
  explicitly in lifecycle and mutation paths while preserving public behavior,
  serialization, alias invariants, ``default_id`` semantics, label metadata,
  reference pass-through, and coordinate metadata.

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
