
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

- Set symmetric color limits on filled contour plots with ``symmetric_percentile``.
- Add a linked upper x-axis to Cartesian plots with the ``secondary_x``
  plotting option.
- Remove transient spikes along a selected acquisition axis with
  ``despike(method="neighbors")``.

- Read 1D Raman JCAMP-DX spectra with explicit Raman shift and intensity pairs.
  (:pr:`1877`)

- Read 1D UV-visible JCAMP-DX spectra with explicit wavelength and intensity pairs.
  (:pr:`1875`)

- Add ``mask_policy="split"`` to ``Filter`` to process contiguous unmasked
  segments independently and report segments too short for the chosen filter. (:pr:`1873`)
- Use ``NDDataset.argsort()`` to obtain the original indices of 1D data sorted
  by value. (:pr:`1871`)
- Plot distributions of finite, unmasked dataset values with
  ``dataset.plot(method="hist")``, with counts or unit-aware probability density. (:pr:`1871`)
- Pass an opacity per trace to ``plot_multiple()`` with ``alpha``. (:pr:`1871`)
- Add wavelet denoising to ``Filter`` with the optional ``wavelets`` extra. (:pr:`1867`)
- Add Butterworth low-pass, FTIR fringe notch and Gaussian methods to
  ``Filter``. Measure their effects with ``rms()`` and
  ``characterize_peaks_1d()``, which reports sampled peak position, height and
  apparent full width at half maximum. (:pr:`1862`)
- Route supported processed 1D NMR JCAMP-DX files through the optional NMR
  plugin when using ``read_jcamp()`` or ``jcamp.read()``. (:pr:`1859`)
- Read supported real 2D NMR JCAMP-DX NTUPLES spectra through the same API
  with the NMR plugin and nmrglue-ng 1.3 or later. (:pr:`1861`)


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

- Link to the PULSAR powder line shape simulation and fitting examples from
  the Modeling and simulation gallery. (:pr:`1874`)

- List NMR examples in the gallery sidebar and include their import cards in
  the main Import / Export grid. (:pr:`1872`)
