
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

- The NMR reader now accepts a ``remove_dc_offset`` keyword argument
  (default ``False``). When ``True``, the receiver DC offset is removed
  from the FID before digital filter correction, which eliminates the
  spike at the centre of the spectrum caused by the receiver electronics.
  This matches the ``remove_dc_offset`` parameter added to nmrglue-ng
  (`spectrochempy/nmrglue-ng#50 <https://github.com/spectrochempy/nmrglue-ng/pull/50>`_)
  (:pr:`1745`).

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- The NMR digital filter removal algorithm in the TopSpin reader has been
  rewritten to follow nmrglue-ng's ``rm_dig_filter`` exactly. The previous
  implementation introduced a constant phase rotation
  (``exp(i*pi*phase)``) with no physical justification, added a flat
  pedestal to the spectrum via an incorrect DC subtraction, derived the
  output length from ``TD//2`` instead of the actual data size, and
  mutated the input dictionary. All four defects are fixed. The output is
  now bit-identical to nmrglue-ng on all available Bruker fixtures
  (`spectrochempy/nmrglue-ng#50 <https://github.com/spectrochempy/nmrglue-ng/pull/50>`_,
  `spectrochempy/nmrglue-ng#51 <https://github.com/spectrochempy/nmrglue-ng/pull/51>`_)
  (:pr:`1745`).


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
