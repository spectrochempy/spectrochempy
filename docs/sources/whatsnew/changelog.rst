
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

- ``Baseline.fit()`` now explicitly rejects datasets whose last-axis
  coordinates contain non-finite values (``NaN`` or infinities), duplicated
  values, or direction changes, raising a clear ``ValueError`` instead of
  silently reordering results or failing with model-dependent internal
  errors. Strictly monotonic but irregularly spaced axes remain accepted, in
  both increasing and decreasing orders.

- Fixed a false linearity detection in ``Coord`` that could silently replace
  descending coordinates containing duplicated or irregularly spaced values
  with an artificial uniform grid. Such coordinates are now preserved as
  provided, while genuinely regular ascending and descending axes keep being
  detected as linear.

- Hardened polynomial baseline fitting on short or underspecified spectra by
  validating support size explicitly, reporting non-intersecting ranges with
  clear ``ValueError`` messages, and avoiding internal index failures when
  automatic edge support is enabled.

- Fixed model-dependent shape inconsistencies in ``Baseline.corrected`` so that
  baseline and corrected datasets preserve the input dataset shape and
  dimensions.

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
