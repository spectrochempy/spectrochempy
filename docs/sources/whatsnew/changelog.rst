
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

- **Optical JCAMP-DX:** Read 1D Raman spectra with explicit Raman shift and
  intensity pairs and 1D UV-visible spectra with explicit wavelength and
  intensity pairs. Multi-file reads return these spectra separately, even with
  ``merge=True``. (:pr:`1877`, :pr:`1875`)
- **NMR JCAMP-DX:** Use ``read_jcamp()`` or ``jcamp.read()`` with the optional
  NMR plugin for supported processed 1D spectra and supported real 2D
  NTUPLES spectra. The 2D path requires nmrglue-ng 1.3 or later.
  Use ``jcamp_kind`` to select the spectrum type explicitly when the declaration
  is ambiguous. LINK containers with conflicting declarations raise an
  error. (:pr:`1859`, :pr:`1861`)
- **Spectral filtering:** Use ``Filter`` for Butterworth low-pass or Gaussian
  smoothing, periodic FTIR fringe removal with a notch filter, or wavelet
  denoising with the optional ``wavelets`` extra. Set ``mask_policy="split"``
  to filter contiguous unmasked segments independently. Filtering raises an
  error if an unmasked segment is too short for the selected filter. Compare results with ``rms()`` and ``characterize_peaks_1d()``, which
  reports sampled peak position, height and apparent full width at half
  maximum. (:pr:`1862`, :pr:`1867`, :pr:`1873`)
- **Acquisition-axis despiking:** Remove transient spikes along a selected
  acquisition axis with ``despike(method="neighbors", dim=...)``. (:pr:`1879`)
- **Dataset inspection and plotting:** Use ``NDDataset.argsort()`` to obtain
  the original indices of 1D values in sorted order, plot finite unmasked
  values as counts or unit-aware probability density with
  ``dataset.plot(method="hist")``, and pass a separate ``alpha`` for each
  trace in ``plot_multiple()``. (:pr:`1871`)
- **Cartesian plot axes and colors:** Use ``secondary_x`` to add a linked
  upper x-axis, and ``symmetric_percentile`` to center the color limits of a
  filled contour plot on zero. (:pr:`1879`)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Honor ``figsize`` when ``plot_multiple()`` creates an overlay figure. (:pr:`1871`)
- Read DIFDUP infrared JCAMP-DX data, handle inline comments, and preserve
  physical spectral axis bounds when line abscissae use an X factor. (:pr:`1866`)


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

- DOC: Clarify reader options by format and make import and NMR examples easier
  to find in the gallery, including the PULSAR powder line shape simulation
  and fitting examples. (:pr:`1863`, :pr:`1864`, :pr:`1865`, :pr:`1872`,
  :pr:`1874`)
- TEST: Qualify IR JCAMP-DX decoding against polystyrene reference spectra.
  (:pr:`1869`)
- CI: Add an opt-in Read the Docs build mode that consumes verified
  documentation artifacts produced for the same commit. (:pr:`1876`, :pr:`1878`)
- DEV: Add targeted local HTML previews for documentation changes.
  (:pr:`1868`)
