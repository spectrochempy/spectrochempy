
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

- Analysis and fitting outputs now follow deterministic derived-metadata rules
  instead of inheriting path-dependent combinations of source identity and
  wrapper defaults. Single-source outputs preserve the captured scientific
  provenance from ``fit()`` time while generating canonical derived
  ``name``/``title``/``description``/``history`` text and clearing
  ``filename``; direct ``NDDataset`` arguments remain authoritative for that
  call only, array-like inputs do not leak stale provenance, and input datasets
  are not mutated. Supervised ``PLSRegression.predict()`` now applies the
  accepted multi-source provenance order (``Xtrain``, ``Ytrain``, then the
  prediction input), ``Optimize.result.residuals`` now follows the same
  canonical derived-output policy as fitted data, and SVD diagnostic
  ``NDDataset`` outputs now use the common diagnostic metadata contract while
  raw ``U``/``s``/``VT`` factors remain unchanged.


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
