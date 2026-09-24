
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

- Added ``cross_validate`` and ``CrossValidationResult`` for bounded supervised
  PLS and Pipeline cross-validation with aligned ``NDDataset`` outputs,
  per-target metrics, fold records, and optional fitted fold estimators (#1658).
- Added documented SpectroChemPy adapters for scikit-learn's ``KFold``,
  ``GroupKFold``, and ``LeaveOneOut`` as ``scp.KFold``, ``scp.GroupKFold``, and
  ``scp.LeaveOneOut`` so supported validation protocols can be constructed from
  the SpectroChemPy namespace (#1660).

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Refused NDDataset in-place arithmetic operations now leave the target
  entirely unchanged instead of appending history or partially updating state.
- NDDataset arithmetic now reconstructs dimensions and coordinates after
  positional broadcasting. When a singleton axis expands, the result uses the
  name and coordinate of the operand providing the non-singleton axis.
  Duplicate result dimension names are rejected explicitly (#1667).
- Dataset arithmetic now rejects different last-dimension coordinate grids
  carrying the same unit instead of silently accepting a scientifically
  incompatible pairing (#1665).


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

- MAINT: Added the cross-validation building blocks used by the public API:
  helpers for resolving
  observation dimensions, validating and slicing aligned folds, and restoring
  prediction geometry (#1653), plus unfitted cloning of ``Pipeline`` templates
  and their supported steps (#1654), and per-target regression metric kernels
  with explicit validity reporting (#1655).
- MAINT: Added an internal structured cross-validation result prototype that
  validates complete out-of-fold coverage and records aligned predictions,
  residuals, metrics, fold positions, configuration snapshots, and optional
  fitted fold estimators (#1656), followed by a private supervised execution
  engine with fold-local cloning, fitting, prediction, and OOF assembly
  (#1657).
