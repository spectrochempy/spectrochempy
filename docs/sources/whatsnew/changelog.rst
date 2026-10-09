
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

- Savitzky-Golay derivatives now preserve the source dataset name and retain
  prior history, appending one processing entry per call instead of generating
  a ``_Filter.transform`` name and replacing the history. This applies to
  ``savgol``, ``savgol_filter``, ``differentiate`` and ``Filter.transform``;
  derivative titles and unit scaling are unchanged (:issue:`1756`).

- Filter outputs (``smooth``, ``savgol`` including its derivatives,
  ``whittaker`` and ``Filter.transform``) now retain the acquisition date of
  their source dataset alongside the other single-source context fields
  (description, author, origin, filename and user metadata). The same shared
  context transfer now supplies the acquisition date for the other generic
  outputs it assembles from a single fitted source; role-based analysis
  outputs keep their own date policies. The estimated baseline output is
  excluded: its metadata policy is decided separately
  (:issue:`1756`).

- The baseline-corrected signal produced by the ``Baseline`` class
  (``corrected`` and ``transform``, on both strictly 1D and 2D inputs) now
  keeps the source description, author, origin, acquisition date, filename
  and user metadata, which were previously recreated by the internal input
  coercion. Its history now matches the existing baseline subtraction
  policy used by ``basc`` and 2D corrections (:issue:`1756`).


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
