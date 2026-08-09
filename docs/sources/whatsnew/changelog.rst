
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

- Filter and smoothing outputs (`smooth`, `savgol`, `savgol_filter`, `whittaker`)
  now preserve the source dataset `name` and append a single history entry while
  retaining all prior entries, instead of renaming with a ``_Filter.transform``
  suffix and replacing the history. Savitzky-Golay derivative outputs
  (`deriv > 0`), ``denoise`` and analysis outputs are unchanged.

- JCAMP writing now rejects, before any file is created or truncated, datasets
  without a `y` coordinate (previously producing a partial file) and datasets
  with complex data (previously silently written with corrupted content).
  No partial or corrupt file is left behind and the source dataset is
  unmodified.

- JCAMP `LINK` files now compute `FIRSTY`, `LASTY`, `MAXY` and `MINY`
  separately for each spectrum block instead of repeating the first/global
  extrema across all blocks. The scientific payload, singleton exports and
  unit tags are unchanged.

- JCAMP writing now emits truthful `XUNITS` / `YUNITS` tags instead of always
  claiming `1/CM` and `ABSORBANCE`. The writer preserves numeric values
  unchanged, accepts only exact-scale mappings (`cm^-1`, `um`, `nm`,
  `absorbance`, `transmittance`, or no unit), rejects merely convertible or
  named arbitrary units such as `m^-1`, `dimensionless`, and `count`, and the
  reader now maps `YUNITS=ARBITRARY UNITS` back to `dataset.units = None`.

- CSV export now rejects complex datasets before creating or truncating a file,
  instead of writing complex-valued CSV content that SpectroChemPy cannot read
  back correctly.


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
