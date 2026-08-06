
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

- Concurrent writes from `MetaConfigurable` instances such as `MCRALS` no longer
  corrupt shared JSON config files. SpectroChemPy now serializes same-process
  config updates, writes JSON atomically, preserves corrupted files under a
  backup name, and avoids emitting raw persistence exceptions on stdout.

- Internal `MCRALS` revalidation after `_n_components` changes no longer emits
  transient config-file writes for normalized legacy constraint traits during
  `fit()`, while explicit user changes to persisted parameters still behave as
  before.

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
