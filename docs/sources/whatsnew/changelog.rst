
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

- Added a bounded supervised cross-validation API for PLS and PLS-ending
  Pipelines. ``cross_validate`` fits preprocessing inside each fold and returns
  an aligned ``CrossValidationResult`` with out-of-fold predictions,
  per-target metrics, fold records, and optional fitted estimators. Documented
  ``KFold``, ``GroupKFold``, and ``LeaveOneOut`` adapters are available from the
  SpectroChemPy namespace, together with a user guide and Gallery example
  (#1658, #1659, #1660).

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``PLSRegression.predict`` now verifies that masked feature positions match
  those used at fit time, preventing coefficients from being applied silently
  to different variables. Cross-validation also preserves the original feature
  geometry for preprocessors such as MSC (#1657).
- Refused NDDataset in-place arithmetic operations now roll back data, units,
  masks, titles, history, and other trait replacements made by the operation.
  Side effects performed by custom Traitlets observers remain the observer's
  responsibility (#1668).
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

- NDDataset arithmetic now rejects two ambiguous pairings that 1.0.0 could
  accept: a broadcast result with duplicate dimension names, and different
  same-unit coordinate grids on a non-expanded final axis. Rename colliding
  dimensions or align the coordinate grids explicitly before the operation
  (#1665, #1667).

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- Improved the public units and masks documentation and simplified examples to
  favor SpectroChemPy-native construction, plotting, and arithmetic where that
  keeps the scientific intent clear (#1661, #1662, #1663).
- Corrected development-package version selection so stable core tags sort
  after their older release candidates. Python and Conda builds now derive
  ``1.0.1.devN`` from ``spectrochempy-v1.0.0`` rather than falling back to an
  obsolete RC series (#1666).
