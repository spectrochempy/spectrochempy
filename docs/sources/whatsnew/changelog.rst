
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

- Added the ``pseudovoigt`` helper and ``pseudovoigtmodel`` curve-fitting
  model, using the existing ``ampl``, ``pos``, ``width``, ``ratio`` and
  ``normalized`` line-shape conventions.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``Baseline`` fitting is now stricter and more predictable: polynomial
  support is validated before fitting, invalid or non-intersecting ranges raise
  clear ``ValueError`` messages, fitted/corrected outputs preserve the input
  shape, and last-axis coordinates with ``NaN``, infinities, duplicates, or
  direction changes are rejected before model-specific internals can produce
  inconsistent results. The Baseline implementation also avoids unnecessary
  ascending-axis sorts and reduces polynomial range-assembly overhead.

- Fixed a false linearity detection in ``Coord`` that could silently replace
  descending coordinates containing duplicated or irregularly spaced values
  with an artificial uniform grid. Such coordinates are now preserved as
  provided, while genuinely regular ascending and descending axes keep being
  detected as linear.

- Stateful preprocessing transformers now reset learned state reliably after
  constructor-parameter changes or failed refits, keep invalid ``set_params()``
  calls transactional, and reject incompatible transform or inverse-transform
  datasets before applying learned statistics. The compatibility guard covers
  feature geometry, coordinate values and units, and data-unit mismatches while
  still allowing new observations along the fitted reduction axis.

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
