
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

- Added :func:`differentiate`, a discoverable public interface for smoothed
  numerical derivatives. It delegates to the existing Savitzky-Golay
  implementation and is available as both ``scp.differentiate(dataset)`` and
  ``dataset.differentiate()``. The derivative order must be a positive integer
  no greater than the fitted polynomial order.

- :func:`read_srs` now exposes the OMNIC ``Collected`` series timestamp of
  SRS series through the standard ``acquisition_date`` dataset property (same
  convention as :func:`read_spa`), and the Y coordinate of dated series gains
  a per-spectrum absolute `datetime` label column derived in full precision
  from the native series fields as ``Collected + timedelta(minutes=time_min +
  i * step)``.  The numeric relative time axis and the spectrum-name labels
  are unchanged.  Series without a valid native anchor (GC variants,
  reprocessed RapidScan files) leave ``acquisition_date`` unset and keep the
  names-only Y labels. (#1617)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Preserved the complete human-readable history when functional preprocessing
  operations are applied in place.  The existing entries and the new
  transformer event are now retained without duplicating timestamps.

- Corrected :func:`download_nist_ir` so its NIST download event actually
  replaces the generic JCAMP import event while retaining subsequent reader
  history entries.

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

- Removed the long-deprecated ``simps`` alias; use ``simpson`` instead.
- Removed the long-deprecated ``force_stack`` compatibility behavior from
  ``concatenate``; use ``stack`` directly.


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- ``Preferences.all()`` now emits the policy-compliant deprecation warning
  that starts its compatibility period in 1.0. Use
  ``Preferences.list_all()`` instead; no premature removal version is promised.


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

MAINT: Added ``.github/workflows/scripts/validate_release_artifacts.py``, a
standalone validator for release artifacts (Python wheel +
sdist and Conda packages) that checks metadata consistency, archive safety,
signature compatibility, installability and smoke-tests, including an exact
check of the installed distribution version, plus its unit tests and the
``.github/workflows/validate_release_artifacts.yml`` workflow. Complete Python
validation requires ``twine``. It never publishes anything or requires secrets.
This first milestone validates locally rebuilt artifacts but does not yet
retain them between jobs or gate the independent publication workflow;
artifact upload is deferred to a later PR.
