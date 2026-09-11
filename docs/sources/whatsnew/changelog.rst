
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

- Restored publication of official plugins (``[tool.spectrochempy]``
  ``official-plugin = true``) on Conda, which was silently skipped by the
  release workflow.  Added a recovery workflow to republish plugin versions
  that are missing on ``spectrocat/main`` and a consistency check between
  GitHub releases, PyPI and Conda. (#1611)

- Corrected the SRS series ``meta.collection_length``: it was the series
  first time (+1002, in minutes) incorrectly converted to seconds; it now
  equals the OMNIC "Total collection time", i.e. the series last time
  (+1006, in minutes) converted to seconds. The time axis is unchanged and
  is still anchored at the series first time. (#1613)

- Added the missing Conda recipe for ``spectrochempy-perkinelmer`` and
  made the repair workflow take a bare ``X.Y.Z`` version (tag derived and
  verified) with a closed plugin list.  Tags without a recipe are now
  recoverable from the canonical ``master`` recipe with a deterministic
  version injection and a core bound aligned with the tag pyproject
  (``recipe_origin=master-fallback``).


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

MAINT: Added ``.github/workflows/scripts/validate_release_artifacts.py``, a
standalone local validator for release artifacts (Python wheel + sdist and
Conda packages) that checks metadata consistency, archive safety, signature
compatibility, installability and smoke-tests, plus its unit tests and the
``.github/workflows/validate_release_artifacts.yml`` workflow. It never
publishes anything, requires no secrets, and is the foundation for a planned
"build once, then validate, then publish" publication architecture.
