
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
- Fix ``dim`` keyword argument in Savitzky-Golay, smoothing and Whittaker
  filters (`#1091 <https://github.com/spectrochempy/spectrochempy/issues/1091>`_).
  ``savgol()``, ``smooth()``, ``whittaker()`` and
  ``Filter(...).transform()`` now accept a ``dim`` parameter to select the
  processing axis.  Dimension names (e.g. ``"x"``) and integer indices are
  resolved via the standard dimension selection mechanism.  Invalid types
  (``bool``, ``tuple``, ``list``) and unknown names raise ``TypeError`` or
  ``ValueError``.


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
- Fix ``safe-docs-no-ci`` CI bypass for push events: the workflow now looks up
  the associated PR via ``gh pr list --commit`` to recover its labels when the
  push event context has none (`#1546 <https://github.com/spectrochempy/spectrochempy/issues/1546>`_).
