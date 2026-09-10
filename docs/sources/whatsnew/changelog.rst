
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
  is still anchored at the series first time.


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
