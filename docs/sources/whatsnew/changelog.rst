:orphan:

What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

- The `readonly` attribute of Meta object now applies to all nested objects.
- More concise html output in jupyter notebooks with collapsible sections. Use CSS for styling.
- Merging behavior of NDDataset objects has been improved. Now several group of datasets are returned if files are not compatible.
  Note also that merging is the default. To enforce non-merging of compatible files, one must use the `merge` keywords set to False in `read_*` calls.
- `read_opus()` has been revised and now uses the `brukeropus` package developed by Josh Duran
  (`<https://github.com/joshduran/brukeropus>`_). It can read most of the spectra types contained in OPUS files,
  as reference experiment and not only AB type spectra as previously.
- Import/Export tutorials updated to reflect the new `brukeropus` package.

.. section

Bug fixes
~~~~~~~~~

- Fix the `readonly` attribute of `Meta` objects used to contain dataset metadata.

.. section

Dependency updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- Removed `brukeropusreader` package dependency.

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- New merging behavior of datasets may require some changes in the notebooks and scripts using `read_*` methods.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- The open/save dialog box functionality in `read_*` methods is deprecated and will be removed in version v0.8.
  This feature has been removed from the documentation.
