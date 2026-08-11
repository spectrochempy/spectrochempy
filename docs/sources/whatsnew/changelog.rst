
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

- Analysis outputs that reuse the stored scientific input (e.g. ``PCA.scores``,
  ``components``, diagnostics, ``result`` outputs, ``PLSRegression`` X-side and
  Y-side surfaces, ``Optimize.fitted``) now preserve the exact ``author`` of the
  scientific source dataset instead of recreating the runtime ``user@host``
  value through the internal ``NDDataset`` input conversion. The snapshot is
  captured once at ``fit`` time and is never overwritten by later direct calls:
  a direct ``NDDataset`` argument remains authoritative for the output of that
  call only, X-side outputs use the X source and Y-side outputs use the Y
  source, array-like inputs are unaffected, and input datasets are not mutated.
  Supervised predictions (``predict``) keep their previous behavior until the
  multi-source provenance policy is implemented.


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
