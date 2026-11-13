
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

- Add wavelet denoising to ``Filter`` with the optional ``wavelets`` extra. (#1852)
- Add Butterworth low-pass, FTIR fringe notch and Gaussian methods to
  ``Filter``. Measure their effects with ``rms()`` and
  ``characterize_peaks_1d()``, which reports sampled peak position, height and
  apparent full width at half maximum. (#1852)
- Route supported processed 1D NMR JCAMP-DX files through the optional NMR
  plugin when using ``read_jcamp()`` or ``jcamp.read()``.
- Read supported real 2D NMR JCAMP-DX NTUPLES spectra through the same API
  with the NMR plugin and nmrglue-ng 1.3 or later.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Read DIFDUP infrared JCAMP-DX data, handle inline comments, and preserve
  physical spectral axis bounds when line abscissae use an X factor.


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
