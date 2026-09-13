:orphan:

What's new in revision 0.7.2
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.7.2.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

- The `readonly` attribute of the Meta object now applies to all nested objects.
- More concise HTML output in Jupyter notebooks with collapsible sections. Use CSS for styling.
- The merging behavior of NDDataset objects has been improved. Now, several groups of datasets are returned if files are not compatible.
  Note that merging is the default. To enforce non-merging of compatible files, use the `merge` keyword set to False in `read_*` calls.
- `read_opus()` has been revised and now uses the `brukeropus` package developed by Josh Duran
  (`<https://github.com/joshduran/brukeropus>`_). It can read most of the spectra types contained in OPUS files,
  as reference experiments and not only AB type spectra as previously.
- Import/Export tutorials have been updated to reflect the new `brukeropus` package.
- `read_dir()` (as well as the equivalent `read()`) has a new keyword argument `pattern` to filter files to read in a directory.
- Text output of the `plot()` method is now suppressed in Jupyter notebooks, i.e.,

  .. code-block:: ipython

    ds.plot()

  will no longer display the text output of the `plot` method but the plot alone.

Bug fixes
~~~~~~~~~

- Fix the `readonly` attribute of `Meta` objects used to contain dataset metadata.

Dependency updates
~~~~~~~~~~~~~~~~~~

- Removed `brukeropusreader` package dependency.

Breaking changes
~~~~~~~~~~~~~~~~

- The new merging behavior of datasets may require some changes in the notebooks and scripts using `read_*` methods.

Deprecations
~~~~~~~~~~~~

- The open/save dialog box functionality in `read_*` methods is deprecated and will be removed in version 0.8.
  This feature has been removed from the documentation.
