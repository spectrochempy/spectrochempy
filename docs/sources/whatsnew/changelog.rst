
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

- ``Baseline(model="arpls")`` and ``Baseline(model="aspls")`` add explicitly
  selected reweighted baseline correction using NumPy/SciPy, also through
  ``get_baseline`` and ``basc``. They accept finite real unmasked 1D or rowwise
  2D spectra on regular grids and record parameters in the correction history.
  Existing defaults are unchanged. Inspect broad and negative peak areas;
  convergence alone does not guarantee quantitative preservation (:pr:`1933`).

- ``Baseline(model="distribution")`` adds experimental automatic signal/noise
  classification, local interpolation and mean smoothing for finite real
  unmasked 1D or row-wise 2D spectra on regular grids.
  ``distribution_width`` is a half-width in samples. Inspect the fitted
  baseline and peak areas: broad weak signals can be removed as background.
  Existing models and defaults are unchanged; no NMR dependency is added
  (:pr:`1932`).

- Run-time hints for surprising defaults. ``align(method="inner")`` warns when
  the intersection keeps fewer than half of the points of the sparsest
  coordinate in the common range, or keeps irregularly spaced points from
  regular grids, and suggests ``method="interpolate"``. ``MCRALS.fit`` lists
  the active constraints, defaults included, in its ``INFO`` log.
  ``Baseline`` ``ranges`` that select no points are reported, in the error
  raised when no support remains or in a warning when they are ignored, with a
  reminder that integer limits are indices and float limits coordinates. The
  error for integer ranges outside the coordinate domain carries the same
  reminder (:issue:`1954`).



.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``Optimize.usermodels`` now keeps each registered model in an independent
  adapter. Registering another function or configuring another optimizer no
  longer replaces existing model functions and parameter signatures, preventing
  incorrect fits with multiple user models. Model instances retain their bound
  state and have independent parameter-name lists (:pr:`1941`).

- ``ActionMassKinetics``: the analytical Jacobian (``use_jac=True``) is now correct
  for stoichiometric coefficients greater than one; ``fit_to_concentrations``
  honors ``method`` in ``optimizer_kwargs`` (it was silently ignored, and
  Nelder-Mead was always used); the optimizer ``left_op`` is applied
  once per evaluation instead of accumulating, and the model holds the optimal
  parameters (and returns the matching concentrations) after the fit. With a
  temperature profile ``T(t)``, ``k_dt`` is now optional (rate constants are then
  computed at each step), the gas constant matches the isothermal path
  (8.314462618 instead of 8.314, a relative change of about 1e-3 on rate
  constants), and plain rate constants raise a clear error (:pr:`1923`).


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- ``plot_compare`` (and ``plot_merit``) legends now use "experimental" and
  "calculated" instead of automatic dataset names such as
  ``NDDataset_1a2b3c4d``; names set on the datasets remain the default labels.
  The legend is built from the plotted lines, so it shows their line styles and
  transparency, and lists the other labelled artists of the axes whose label is
  not already shown (:issue:`1954`).

- ``Baseline`` ``ranges`` that select no points now raise a ``ValueError``
  instead of a ``TraitError`` when no baseline support remains
  (:issue:`1954`).


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- Passing ``labels`` to ``plot_compare`` is deprecated: it is ignored and emits
  a ``DeprecationWarning``. Use ``exp_label``, ``calc_label`` and
  ``resid_label`` instead (:issue:`1954`).

- The experimental ``spectrochempy-cantera`` plugin and its ``PFR`` model
  (``scp.cantera.PFR`` and the deprecated alias ``scp.PFR``) will be removed in
  the next minor release. ``PFR`` cannot be built with any Cantera version 3.0 or
  later and has no known users. Cases that need Cantera can use it directly and
  load the results into an ``NDDataset`` (:pr:`1925`).


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)
