
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

- Added the experimental and opt-in :mod:`spectrochempy.provenance` substrate:
  immutable structured operation/reference models, an append-only ledger,
  context-local capture ownership, and conservative parameter normalization.
  It does not instrument scientific operations and does not export
  reproducibility manifests yet.

- Instrumented out-of-place ``NDDataset`` slicing and ``NDDataset.transpose``
  so an active :class:`~spectrochempy.provenance.ProvenanceCapture` records
  single-source selection and transpose operations, with transient weak object
  tracking and unchanged data and textual history. All other operations - including
  in-place slicing and in-place transpose - remain uninstrumented and create no
  record.

- Added bounded experimental provenance for direct ``CenterTransformer.fit()``
  and ``CenterTransformer.transform()`` calls. Records preserve one transformer
  identity across fitted states, link calibration and transformed datasets,
  summarize learned state without retaining learned arrays, and report failed
  refit mutation explicitly. Composite, functional, inverse, persistence, and
  replay paths, and ``CenterTransformer`` subclasses remain outside this slice.

- Added experimental provenance for out-of-place addition, subtraction,
  multiplication, and true division between two ``NDDataset`` operands.
  Records preserve ordered ``left`` and ``right`` input roles across operator
  and direct NumPy-ufunc dispatch, create one linked result state, and leave
  scalar, external-array, in-place, ``out=``, concatenate/stack, persistence,
  and replay paths outside this bounded slice.

- Added experimental provenance for ``concatenate`` and ``stack`` over
  supported ``NDDataset`` inputs. One record preserves every ordered input
  position, reuses the same state reference for repeated objects, distinguishes
  explicitly requested dimension arguments from their resolved axis and
  dimension, and creates one linked result state without retaining datasets.
  NumPy array-conversion paths and unsupported input forms remain outside this
  bounded slice.

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

- Hardened the experimental provenance instrumentation: an in-place
  modification of a tracked dataset is detected and reported (``partial``
  capture with an ``unrecorded_state_change`` omission) instead of silently
  chaining to a stale state; parameter description failures never break the
  recorded scientific operation; and failed operations keep the bounded
  requested parameters. In addition, in-place slicing
  (``dataset[:, ..., INPLACE]``) is no longer mis-recorded as a complete
  out-of-place slice - it is excluded from capture, matching the in-place
  transpose policy.

- Refused NDDataset in-place arithmetic operations now roll back data, units,
  masks, titles, history, and other trait replacements made by the operation.
  Side effects performed by custom Traitlets observers remain the observer's
  responsibility.
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
