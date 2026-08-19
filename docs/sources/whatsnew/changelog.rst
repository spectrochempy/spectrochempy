
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

- Savitzky-Golay derivatives now automatically use the signed spacing of a
  uniformly spaced coordinate when ``delta`` is omitted
  (`#1091 <https://github.com/spectrochempy/spectrochempy/issues/1091>`_).
  For ``deriv > 0``, the ``savgol()`` wrapper detects the coordinate along the
  processed axis and derives a signed delta.  An ascending coordinate yields a
  positive delta, a descending one yields a negative delta.  The derivative
  sign therefore conforms to the physical variable carried by the coordinate
  without relying on the unit-based ``_reversed`` heuristic.  An explicit
  ``delta`` has priority and disables auto-detection.  On a non-uniform
  or missing coordinate a warning is emitted and the index-based ``delta=1.0``
  is used as a fallback.  ``deriv=0`` is unchanged.

- Fix sign of Savitzky-Golay derivatives when an explicit ``delta`` is
  provided with ``cm⁻¹`` or ``ppm`` coordinates
  (`#1552 <https://github.com/spectrochempy/spectrochempy/issues/1552>`_).
  The former ``_reversed`` correction applied ``(-1)**deriv`` on the
  Savitzky-Golay path when the coordinate carried ``cm⁻¹`` or ``ppm``
  units, flipping the sign of odd-order derivatives on ascending
  coordinates.  An explicit ``delta`` is now passed to SciPy with its
  sign; no unit-based correction is applied.  For a descending
  coordinate, supply a negative ``delta`` if the derivative should follow
  the physical axis.  This is a numeric correction for affected calls
  with odd-order derivatives.


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
