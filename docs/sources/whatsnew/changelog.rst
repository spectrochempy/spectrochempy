
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

- MAINT: Added private cross-validation building blocks: helpers for resolving
  observation dimensions, validating and slicing aligned folds, and restoring
  prediction geometry (#1653), plus unfitted cloning of ``Pipeline`` templates
  and their supported steps (#1654), and per-target regression metric kernels
  with explicit validity reporting (#1655).  No public cross-validation API
  is introduced.
- MAINT: Added an internal structured cross-validation result prototype that
  validates complete out-of-fold coverage and records aligned predictions,
  residuals, metrics, fold positions, configuration snapshots, and optional
  fitted fold estimators (#1656), followed by a private supervised execution
  engine with fold-local cloning, fitting, prediction, and OOF assembly
  (#1657).  The implementation remains private and does not add a public API.
