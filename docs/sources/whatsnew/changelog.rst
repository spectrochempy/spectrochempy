
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

- Added ``simulate_peaks`` to sum separable Gaussian and Lorentzian peaks
  on a multidimensional coordinate grid with unit-aware parameters.

- Added ``segment_regions`` for threshold-based connected segmentation of 1D
  and 2D spectra, returning coordinate-preserving integer labels and region bounds.

- Added coordinate-aware marginal panels to 2D contour plots, with selectable
  dimensions, calculated projections or external reference spectra, and signed
  contour colors/styles. ``multiplot`` composes these panels, supports mixed
  1D/2D datasets and per-panel options, and saves the completed figure once.

- Added the generic ``fft.validate_input`` plugin hook for read-only FFT/IFFT
  checks before in-place axis changes, resizing or numerical preparation.
  Validators can reject a request without leaving the caller partly modified.

- ``find_peaks`` now supports explicitly selected true 2D spectra with
  deterministic positive-maximum detection, anisotropic point/physical spacing,
  and structured ``PeakTable`` results. The historical 1D behavior is unchanged.

- Added ``correlation_spectroscopy`` for synchronous and asynchronous 2D correlation
  spectroscopy maps, with coordinate-aware validation of real, uniformly spaced
  perturbation coordinates.

- Added a gallery example for 2D-IR peak finding on an experimental spectrum.

- ``PeakFindingResult`` and ``PeakTable`` now display compact text and HTML
  tables with units in column headers and a bounded preview of detected peaks.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Restored the documented 0.1 percent default coordinate linearization
  tolerance in the ``Coord`` constructor. Use ``linearize_below`` explicitly
  when a more permissive tolerance is required. FTIR FFT coordinates convert
  frequency bins to wavenumbers before display rounding to preserve uniform
  bin spacing with the standard tolerance.

- Fixed gallery thumbnail selection by placing Sphinx-Gallery configuration
  after the example docstring so the requested figure is recognized.


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- Remove built-in quaternion handling and the Hypercomplex plugin from the
  official plugin catalog, bundled extras, development environments and CI.
  Specialized NMR representations belong to the standalone NMR plugin; use
  ordinary complex arrays for core numerical operations.


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)
