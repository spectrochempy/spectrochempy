
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
