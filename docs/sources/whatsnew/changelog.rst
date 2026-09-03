
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

- Added ``Pipeline`` for linear, reproducible composition of allowlisted
  SpectroChemPy preprocessing transformers with a final transformer or
  supervised estimator.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- correct OMNIC SRS series time-axis anchor and 84-byte spectrum labels


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

- Added the internal estimator-contract helpers required before a future
  ``Pipeline`` implementation: allowlist-based fitted-state inspection,
  unfitted cloning, canonical not-fitted behavior for supported transformers,
  and lifecycle invalidation for accepted analysis terminal candidates.
