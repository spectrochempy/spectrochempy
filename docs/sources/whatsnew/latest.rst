:orphan:

What's new in revision 0.7.2.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.7.2.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

- The `readonly` attribute of Meta object now applies to all nested objects.
- More concise html output in jupyter notebooks with collapsible sections. Use CSS for styling.
- Merging behavior of NDDataset objects has been improved. Now several group of datasets are returned if files are not compatible.
  Note also that merging is the default. To enforce non-merging of compatible files, one must use the `merge` keywords set to False in `read_*` calls.
- `read_opus()` has been revised and now uses the `brukeropus` package developed by Josh Duran
  (`<https://github.com/joshduran/brukeropus>`_). It can read most of the spectra types contained in OPUS files,
  as reference experiment and not only AB type spectra as previously.
- Import/Export tutorials updated to reflect the new `brukeropus` package.
- `read_dir()` (as well as equivalent `read()`) has a new keyword argument `pattern` to filter files to read in a directory.`

Bug fixes
~~~~~~~~~

- Fix the `readonly` attribute of `Meta` objects used to contain dataset metadata.

Dependency updates
~~~~~~~~~~~~~~~~~~

- Removed `brukeropusreader` package dependency.

Breaking changes
~~~~~~~~~~~~~~~~

- New merging behavior of datasets may require some changes in the notebooks and scripts using `read_*` methods.

Deprecations
~~~~~~~~~~~~

- The open/save dialog box functionality in `read_*` methods is deprecated and will be removed in version v0.8.
  This feature has been removed from the documentation.
