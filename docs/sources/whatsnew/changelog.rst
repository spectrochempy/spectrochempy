
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
  tracking and unchanged data and textual history. All other operations remain
  uninstrumented.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Fixed ``average(..., keepdims=True)`` so reduced dimensions are retained in
  both the data shape and dataset metadata, consistently with other reductions.
  (:pr:`1641`)

- Restored asynchronous update notifications after lazy application startup.
  Stable installations now ignore prereleases, while release-candidate
  installations are notified about newer candidates and the final release.
  (:pr:`1636`)

- Fixed cross-decomposition inverse transforms so Y is reconstructed from its
  own transformed scores instead of reusing the X scores.
  (:pr:`1642`)


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
